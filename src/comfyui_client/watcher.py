"""Folder watcher daemon for ComfyUI job submission"""
import json
import logging
import os
import signal
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .workflow import load_workflow, apply_params

log = logging.getLogger(__name__)


class FolderWatcher:
    """Monitor a directory for JSON job files and submit them to ComfyUI.

    Directory layout created on startup:
        incoming/   - drop job JSON files here
        processing/ - files currently being processed
        completed/  - successfully processed jobs (with _result metadata)
        failed/     - jobs that errored (with _error metadata)
        output/     - downloaded images
    """

    SUBDIRS = ("incoming", "processing", "completed", "failed", "output")

    def __init__(self, watch_dir, client, default_workflow=None, poll_interval=2.0):
        self.watch_dir = Path(watch_dir)
        self.client = client
        self.default_workflow = Path(default_workflow) if default_workflow else None
        self.poll_interval = poll_interval
        self._running = False

    def start(self):
        """Start the blocking main loop."""
        self._setup_dirs()
        self._setup_signals()
        self._recover_processing()

        self._running = True
        log.info("Watching %s (poll every %.1fs)", self.watch_dir / "incoming", self.poll_interval)
        if self.default_workflow:
            log.info("Default workflow: %s", self.default_workflow)

        while self._running:
            files = self._scan_incoming()
            for f in files:
                if not self._running:
                    break
                self._process_file(f)
            if self._running:
                time.sleep(self.poll_interval)

        log.info("Watcher stopped")

    def stop(self):
        """Signal the main loop to stop."""
        self._running = False

    # -- Setup -----------------------------------------------------------

    def _setup_dirs(self):
        for name in self.SUBDIRS:
            (self.watch_dir / name).mkdir(parents=True, exist_ok=True)

    def _setup_signals(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info("Received signal %s, shutting down...", signal.Signals(signum).name)
        self.stop()

    def _recover_processing(self):
        """Move leftover processing/ files back to incoming/ on startup."""
        processing = self.watch_dir / "processing"
        incoming = self.watch_dir / "incoming"
        for entry in sorted(processing.iterdir()):
            if entry.suffix == ".json":
                dest = incoming / entry.name
                shutil.move(str(entry), str(dest))
                log.info("Recovered %s -> incoming/", entry.name)

    # -- Scanning --------------------------------------------------------

    def _scan_incoming(self):
        """Return .json files in incoming/ sorted by mtime (oldest first).

        Skips files with mtime < 1 second ago to avoid reading partially
        written files.
        """
        incoming = self.watch_dir / "incoming"
        now = time.time()
        files = []
        for entry in os.scandir(incoming):
            if entry.name.endswith(".json") and entry.is_file():
                try:
                    stat = entry.stat()
                    if now - stat.st_mtime >= 1.0:
                        files.append((stat.st_mtime, Path(entry.path)))
                except OSError:
                    continue
        files.sort(key=lambda x: x[0])
        return [f for _, f in files]

    # -- Job parsing -----------------------------------------------------

    def _parse_job(self, path):
        """Parse a job file and return a list of job dicts.

        Supported formats:
        - Array: each element is a job dict (batch)
        - Dict with "workflow" key: full job spec
        - Dict without "workflow" key: minimal params-only job
        """
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unexpected JSON type: {type(data).__name__}")

    def _resolve_workflow(self, job):
        """Load the workflow for a job and apply parameter overrides.

        Returns the prepared workflow dict ready for submission.
        """
        if "workflow" in job:
            wf_path = job["workflow"]
            # Resolve relative paths against the watch dir
            wf_path = Path(wf_path)
            if not wf_path.is_absolute():
                wf_path = self.watch_dir / wf_path
            workflow = load_workflow(wf_path)
        elif self.default_workflow:
            workflow = load_workflow(self.default_workflow)
        else:
            raise ValueError("No workflow specified and no default workflow configured")

        workflow = deepcopy(workflow)
        return apply_params(
            workflow,
            prompt=job.get("prompt"),
            negative=job.get("negative"),
            seed=job.get("seed"),
            steps=job.get("steps"),
            cfg=job.get("cfg"),
            width=job.get("width"),
            height=job.get("height"),
            denoise=job.get("denoise"),
            sampler=job.get("sampler"),
            scheduler=job.get("scheduler"),
            image=job.get("image"),
        )

    # -- Processing ------------------------------------------------------

    def _process_file(self, path):
        """Process a single job file: parse, submit, wait, download, move."""
        fname = path.name
        processing_path = self.watch_dir / "processing" / fname
        log.info("Processing %s", fname)

        # Move to processing/
        try:
            shutil.move(str(path), str(processing_path))
        except OSError as e:
            log.error("Failed to move %s to processing/: %s", fname, e)
            return

        try:
            jobs = self._parse_job(processing_path)
        except (json.JSONDecodeError, ValueError) as e:
            log.error("Failed to parse %s: %s", fname, e)
            self._move_failed(processing_path, str(e))
            return

        results = []
        all_ok = True

        for i, job in enumerate(jobs):
            job_label = f"{fname}[{i}]" if len(jobs) > 1 else fname
            try:
                workflow = self._resolve_workflow(job)
            except (ValueError, FileNotFoundError) as e:
                log.error("%s: workflow error: %s", job_label, e)
                results.append({"_error": {
                    "message": str(e),
                    "failed_at": _now_iso(),
                }})
                all_ok = False
                continue

            try:
                prompt_id = self.client.submit(workflow)
                log.info("%s: submitted %s", job_label, prompt_id)
            except httpx.ConnectError as e:
                # Server unreachable — leave in processing/ for retry
                log.warning("%s: ComfyUI unreachable (%s), will retry", job_label, e)
                return  # Don't move the file, leave for next cycle
            except (RuntimeError, httpx.HTTPStatusError) as e:
                log.error("%s: submit error: %s", job_label, e)
                results.append({"_error": {
                    "message": str(e),
                    "failed_at": _now_iso(),
                }})
                all_ok = False
                continue

            try:
                result = self.client.wait_for_completion(prompt_id)
                output_dir = self.watch_dir / "output"
                saved = self.client.download_images(result, output_dir)
                image_names = [p.name for p in saved]
                log.info("%s: completed, %d image(s)", job_label, len(saved))
                results.append({"_result": {
                    "prompt_id": prompt_id,
                    "images": image_names,
                    "completed_at": _now_iso(),
                }})
            except httpx.ConnectError as e:
                log.warning("%s: lost connection during wait (%s), will retry", job_label, e)
                return  # Leave in processing/ for retry
            except RuntimeError as e:
                log.error("%s: execution error: %s", job_label, e)
                results.append({"_error": {
                    "message": str(e),
                    "prompt_id": prompt_id,
                    "failed_at": _now_iso(),
                }})
                all_ok = False

        # Annotate the job file with results and move
        self._annotate_and_move(processing_path, jobs, results, all_ok)

    def _annotate_and_move(self, path, jobs, results, all_ok):
        """Write result metadata into the job file and move to completed/ or failed/."""
        try:
            # Merge results back into job dicts
            for job, res in zip(jobs, results):
                job.update(res)

            data = jobs if len(jobs) > 1 else jobs[0]
            path.write_text(json.dumps(data, indent=2) + "\n")
        except OSError as e:
            log.error("Failed to write results to %s: %s", path.name, e)

        dest_dir = "completed" if all_ok else "failed"
        dest = self.watch_dir / dest_dir / path.name
        try:
            shutil.move(str(path), str(dest))
            log.info("Moved %s -> %s/", path.name, dest_dir)
        except OSError as e:
            log.error("Failed to move %s to %s/: %s", path.name, dest_dir, e)

    def _move_failed(self, path, error_msg):
        """Move a file to failed/ with error metadata."""
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["_error"] = {"message": error_msg, "failed_at": _now_iso()}
        elif isinstance(data, dict):
            data["_error"] = {"message": error_msg, "failed_at": _now_iso()}

        try:
            path.write_text(json.dumps(data, indent=2) + "\n")
        except OSError:
            pass

        dest = self.watch_dir / "failed" / path.name
        try:
            shutil.move(str(path), str(dest))
            log.info("Moved %s -> failed/", path.name)
        except OSError as e:
            log.error("Failed to move %s to failed/: %s", path.name, e)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()
