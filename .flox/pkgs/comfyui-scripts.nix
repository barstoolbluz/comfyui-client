{ lib, writeShellApplication, writeTextFile, symlinkJoin, runCommand }:

let
  # Model types for workflow scripts
  models = [ "sd15" "sdxl" "sd35" "flux" ];

  operations = [ "txt2img" "img2img" "upscale" "inpaint" ];

  # Generate a single workflow wrapper script
  makeScript = model: op:
    let
      name = "${model}-${op}";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOWS_BASE="''${COMFYUI_WORKFLOWS:-''${FLOX_ENV}/share/comfyui-client/workflows}"
        WORKFLOW="$WORKFLOWS_BASE/api/${model}/${model}-${op}.json"

        if [ ! -f "$WORKFLOW" ]; then
          echo "Error: Workflow not found: $WORKFLOW" >&2
          echo "Set COMFYUI_WORKFLOWS to your workflows directory." >&2
          exit 1
        fi

        exec comfyui-submit "$WORKFLOW" --wait "$@"
      '';
    };

  # Generate all workflow scripts
  allScripts = lib.flatten (
    map (model:
      map (op: makeScript model op) operations
    ) models
  );

  # Python source files
  pyproject = writeTextFile {
    name = "pyproject.toml";
    text = ''
      [project]
      name = "comfyui-client"
      version = "0.1.0"
      description = "CLI client for ComfyUI API"
      requires-python = ">=3.11"
      dependencies = [
          "httpx>=0.27",
          "websockets>=12.0",
          "typer>=0.12",
          "rich>=13.0",
      ]

      [project.scripts]
      comfyui-submit = "comfyui_client.cli:submit_cli"
      comfyui-queue = "comfyui_client.cli:queue_cli"
      comfyui-result = "comfyui_client.cli:result_cli"
      comfyui-batch = "comfyui_client.cli:batch_cli"
      comfyui-cancel = "comfyui_client.cli:cancel_cli"
      comfyui-status = "comfyui_client.cli:status_cli"
      comfyui-models = "comfyui_client.cli:models_cli"
      comfyui-info = "comfyui_client.cli:info_cli"

      [build-system]
      requires = ["hatchling"]
      build-backend = "hatchling.build"
    '';
  };

  initPy = writeTextFile {
    name = "__init__.py";
    text = ''
      """ComfyUI API Client"""
      from .client import ComfyUIClient
      from .metadata import extract_comfyui_metadata, summarize_metadata
      from .workflow import (
          load_workflow,
          clean_workflow,
          set_prompt,
          set_seed,
          get_seed,
          set_steps,
          set_cfg,
          set_dimensions,
          set_denoise,
          set_sampler,
          set_scheduler,
          set_input_image,
      )

      __all__ = [
          "ComfyUIClient",
          "extract_comfyui_metadata",
          "summarize_metadata",
          "load_workflow",
          "clean_workflow",
          "set_prompt",
          "set_seed",
          "get_seed",
          "set_steps",
          "set_cfg",
          "set_dimensions",
          "set_denoise",
          "set_sampler",
          "set_scheduler",
          "set_input_image",
      ]
    '';
  };

  clientPy = writeTextFile {
    name = "client.py";
    text = ''
      """ComfyUI API Client"""
      import asyncio
      import json

      import httpx
      import uuid
      import websockets


      class ComfyUIClient:
          def __init__(self, host: str = "localhost", port: int = 8188):
              self.host = host
              self.port = port
              self.base_url = f"http://{host}:{port}"
              self.client_id = str(uuid.uuid4())

          @property
          def ws_url(self) -> str:
              return f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"

          def submit(self, workflow: dict) -> str:
              """Submit workflow, return prompt_id"""
              response = httpx.post(
                  f"{self.base_url}/prompt",
                  json={"prompt": workflow, "client_id": self.client_id}
              )
              if response.status_code >= 400:
                  try:
                      error_data = response.json()
                      error_msg = error_data.get("error", {}).get("message", response.text)
                      node_errors = error_data.get("node_errors", {})
                      if node_errors:
                          error_msg += f"\nNode errors: {node_errors}"
                  except Exception:
                      error_msg = response.text
                  raise RuntimeError(f"ComfyUI error ({response.status_code}): {error_msg}")
              return response.json()["prompt_id"]

          def get_history(self, prompt_id: str) -> dict:
              """Get execution history for prompt_id"""
              response = httpx.get(f"{self.base_url}/history/{prompt_id}")
              response.raise_for_status()
              return response.json()

          def get_queue(self) -> dict:
              """Get current queue status"""
              response = httpx.get(f"{self.base_url}/queue")
              response.raise_for_status()
              return response.json()

          def interrupt(self):
              """Interrupt the currently running workflow"""
              response = httpx.post(f"{self.base_url}/interrupt", json={})
              response.raise_for_status()

          def delete_from_queue(self, prompt_ids: list[str]):
              """Delete specific prompt IDs from the queue"""
              response = httpx.post(f"{self.base_url}/queue", json={"delete": prompt_ids})
              response.raise_for_status()

          def clear_queue(self):
              """Clear all pending items from the queue"""
              response = httpx.post(f"{self.base_url}/queue", json={"clear": True})
              response.raise_for_status()

          def get_system_stats(self) -> dict:
              """Get system stats (versions, RAM, devices)"""
              response = httpx.get(f"{self.base_url}/system_stats")
              response.raise_for_status()
              return response.json()

          def get_model_types(self) -> list[str]:
              """Get available model folder types"""
              response = httpx.get(f"{self.base_url}/models")
              response.raise_for_status()
              return response.json()

          def get_models(self, folder: str) -> list[str]:
              """Get models in a specific folder"""
              response = httpx.get(f"{self.base_url}/models/{folder}")
              response.raise_for_status()
              return response.json()

          def get_image(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
              """Download image from ComfyUI"""
              response = httpx.get(
                  f"{self.base_url}/view",
                  params={"filename": filename, "subfolder": subfolder, "type": type}
              )
              response.raise_for_status()
              return response.content

          async def _ws_wait(self, prompt_id: str, on_progress=None) -> dict:
              """Listen on WebSocket until workflow completes"""
              # Fast path: if already completed (cached/instant execution), return now
              history = self.get_history(prompt_id)
              if prompt_id in history:
                  return history[prompt_id]

              async with websockets.connect(self.ws_url, max_size=2**24) as ws:
                  while True:
                      raw = await asyncio.wait_for(ws.recv(), timeout=1800)
                      if isinstance(raw, bytes):
                          continue
                      msg = json.loads(raw)
                      msg_type = msg.get("type")
                      data = msg.get("data", {})

                      if data.get("prompt_id") != prompt_id:
                          continue

                      if on_progress:
                          on_progress(msg_type, data)

                      if msg_type == "executing" and data.get("node") is None:
                          break
                      elif msg_type == "execution_error":
                          raise RuntimeError(
                              f"Workflow error: {data.get('exception_message', 'unknown')}"
                          )

              history = self.get_history(prompt_id)
              return history[prompt_id]

          def wait_for_completion(self, prompt_id: str, on_progress=None) -> dict:
              """Wait for workflow completion via WebSocket"""
              return asyncio.run(self._ws_wait(prompt_id, on_progress))
    '';
  };

  workflowPy = writeTextFile {
    name = "workflow.py";
    text = ''
      """Workflow loading and modification utilities"""
      import json
      from pathlib import Path


      def clean_workflow(workflow: dict) -> dict:
          """Remove non-node entries from workflow (last_node_id, last_link_id, etc.)"""
          return {
              node_id: node
              for node_id, node in workflow.items()
              if isinstance(node, dict) and "class_type" in node
          }


      def load_workflow(path: Path) -> dict:
          """Load workflow JSON file and clean it for API submission"""
          workflow = json.loads(path.read_text())
          return clean_workflow(workflow)


      def find_node_by_class(workflow: dict, class_type: str) -> tuple[str, dict] | None:
          """Find first node of given class type"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              if node.get("class_type") == class_type:
                  return node_id, node
          return None


      def find_all_nodes_by_class(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
          """Find all nodes of given class type"""
          return [
              (node_id, node)
              for node_id, node in workflow.items()
              if isinstance(node, dict) and node.get("class_type") == class_type
          ]


      def set_prompt(workflow: dict, positive: str, negative: str = "") -> dict:
          """Set positive/negative prompts in workflow"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              if node.get("class_type") == "CLIPTextEncode":
                  title = node.get("_meta", {}).get("title", "").lower()
                  if "positive" in title or "prompt" in title:
                      node["inputs"]["text"] = positive
                  elif "negative" in title:
                      node["inputs"]["text"] = negative
          return workflow


      def set_seed(workflow: dict, seed: int) -> dict:
          """Set seed in KSampler and KSamplerAdvanced nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  if "seed" in node.get("inputs", {}):
                      node["inputs"]["seed"] = seed
                  elif "noise_seed" in node.get("inputs", {}):
                      node["inputs"]["noise_seed"] = seed
          return workflow


      def get_seed(workflow: dict) -> int | None:
          """Read current seed from first KSampler node"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  inputs = node.get("inputs", {})
                  if "seed" in inputs:
                      return inputs["seed"]
                  elif "noise_seed" in inputs:
                      return inputs["noise_seed"]
          return None


      def set_steps(workflow: dict, steps: int) -> dict:
          """Set sampling steps in KSampler nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  if "steps" in node.get("inputs", {}):
                      node["inputs"]["steps"] = steps
          return workflow


      def set_cfg(workflow: dict, cfg: float) -> dict:
          """Set CFG scale in KSampler nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "cfg" in node.get("inputs", {}):
                      node["inputs"]["cfg"] = cfg
          return workflow


      def set_dimensions(workflow: dict, width: int, height: int) -> dict:
          """Set image dimensions in EmptyLatentImage nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              if node.get("class_type") == "EmptyLatentImage":
                  node["inputs"]["width"] = width
                  node["inputs"]["height"] = height
          return workflow


      def set_denoise(workflow: dict, denoise: float) -> dict:
          """Set denoise strength in KSampler nodes (for img2img)"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "denoise" in node.get("inputs", {}):
                      node["inputs"]["denoise"] = denoise
          return workflow


      def set_sampler(workflow: dict, sampler_name: str) -> dict:
          """Set sampler in KSampler nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "sampler_name" in node.get("inputs", {}):
                      node["inputs"]["sampler_name"] = sampler_name
          return workflow


      def set_scheduler(workflow: dict, scheduler: str) -> dict:
          """Set scheduler in KSampler nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "scheduler" in node.get("inputs", {}):
                      node["inputs"]["scheduler"] = scheduler
          return workflow


      def set_input_image(workflow: dict, image_path: str) -> dict:
          """Set input image path in LoadImage nodes"""
          for node_id, node in workflow.items():
              if not isinstance(node, dict):
                  continue
              if node.get("class_type") == "LoadImage":
                  node["inputs"]["image"] = image_path
          return workflow
    '';
  };

  metadataPy = writeTextFile {
    name = "metadata.py";
    text = ''
      """PNG metadata extraction for ComfyUI-generated images (stdlib only)"""
      import json
      import struct
      import zlib
      from pathlib import Path


      def read_png_text_chunks(filepath: Path) -> dict[str, str]:
          """Read tEXt and iTXt chunks from a PNG file.

          PNG chunk format: 4-byte length (big-endian), 4-byte type, data, 4-byte CRC.
          tEXt chunks contain: keyword\\x00text
          iTXt chunks contain: keyword\\x00\\x00\\x00\\x00\\x00text (simplified)
          """
          chunks = {}
          with open(filepath, "rb") as f:
              sig = f.read(8)
              if sig != b"\x89PNG\r\n\x1a\n":
                  return chunks

              while True:
                  header = f.read(8)
                  if len(header) < 8:
                      break
                  length, chunk_type = struct.unpack(">I4s", header)
                  chunk_data = f.read(length)
                  f.read(4)  # skip CRC

                  if chunk_type == b"tEXt":
                      sep = chunk_data.index(b"\x00")
                      key = chunk_data[:sep].decode("latin-1")
                      val = chunk_data[sep + 1:].decode("latin-1")
                      chunks[key] = val
                  elif chunk_type == b"iTXt":
                      sep = chunk_data.index(b"\x00")
                      key = chunk_data[:sep].decode("utf-8")
                      rest = chunk_data[sep + 1:]
                      # compression flag, compression method, language tag\0, translated keyword\0
                      comp_flag = rest[0]
                      # skip compression method (1 byte)
                      rest = rest[2:]
                      lang_end = rest.index(b"\x00")
                      rest = rest[lang_end + 1:]
                      trans_end = rest.index(b"\x00")
                      rest = rest[trans_end + 1:]
                      if comp_flag:
                          val = zlib.decompress(rest).decode("utf-8")
                      else:
                          val = rest.decode("utf-8")
                      chunks[key] = val
                  elif chunk_type == b"IEND":
                      break

          return chunks


      def extract_comfyui_metadata(filepath: Path) -> dict:
          """Extract ComfyUI metadata from a PNG file.

          ComfyUI stores metadata as tEXt chunks:
          - 'prompt': API-format nodes JSON
          - 'workflow': web-UI-format workflow JSON
          """
          chunks = read_png_text_chunks(filepath)
          result = {}

          if "prompt" in chunks:
              try:
                  result["prompt"] = json.loads(chunks["prompt"])
              except json.JSONDecodeError:
                  result["prompt_raw"] = chunks["prompt"]

          if "workflow" in chunks:
              try:
                  result["workflow"] = json.loads(chunks["workflow"])
              except json.JSONDecodeError:
                  result["workflow_raw"] = chunks["workflow"]

          return result


      def summarize_metadata(metadata: dict) -> dict:
          """Extract key generation parameters from ComfyUI prompt metadata."""
          summary = {}
          prompt = metadata.get("prompt", {})

          for node_id, node in prompt.items():
              if not isinstance(node, dict):
                  continue
              class_type = node.get("class_type", "")
              inputs = node.get("inputs", {})

              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  if "seed" in inputs:
                      summary["seed"] = inputs["seed"]
                  elif "noise_seed" in inputs:
                      summary["seed"] = inputs["noise_seed"]
                  if "steps" in inputs:
                      summary["steps"] = inputs["steps"]
                  if "cfg" in inputs:
                      summary["cfg"] = inputs["cfg"]
                  if "sampler_name" in inputs:
                      summary["sampler"] = inputs["sampler_name"]
                  if "scheduler" in inputs:
                      summary["scheduler"] = inputs["scheduler"]
                  if "denoise" in inputs:
                      summary["denoise"] = inputs["denoise"]

              elif class_type == "CLIPTextEncode":
                  title = node.get("_meta", {}).get("title", "").lower()
                  text = inputs.get("text", "")
                  if isinstance(text, str) and text:
                      if "positive" in title or "prompt" in title:
                          summary["positive_prompt"] = text
                      elif "negative" in title:
                          summary["negative_prompt"] = text

              elif class_type == "EmptyLatentImage":
                  if "width" in inputs:
                      summary["width"] = inputs["width"]
                  if "height" in inputs:
                      summary["height"] = inputs["height"]

              elif class_type == "CheckpointLoaderSimple":
                  if "ckpt_name" in inputs:
                      summary["model"] = inputs["ckpt_name"]

              elif class_type == "UNETLoader":
                  if "unet_name" in inputs:
                      summary.setdefault("model", inputs["unet_name"])

          return summary
    '';
  };

  cliPy = writeTextFile {
    name = "cli.py";
    text = ''
      """CLI commands for ComfyUI client"""
      import json
      import typer
      from copy import deepcopy
      from rich.console import Console
      from rich.panel import Panel
      from rich.progress import Progress, SpinnerColumn, TextColumn
      from rich.table import Table
      from pathlib import Path
      import os

      from .client import ComfyUIClient
      from .metadata import extract_comfyui_metadata, summarize_metadata
      from .workflow import (
          load_workflow,
          set_prompt,
          set_seed,
          get_seed,
          set_steps,
          set_cfg,
          set_dimensions,
          set_denoise,
          set_sampler,
          set_scheduler,
          set_input_image,
      )

      app = typer.Typer()
      console = Console()


      def get_client() -> ComfyUIClient:
          return ComfyUIClient(
              host=os.environ.get("COMFYUI_HOST", "localhost"),
              port=int(os.environ.get("COMFYUI_PORT", "8188"))
          )


      def _apply_params(workflow, prompt, negative, seed, steps, cfg, width, height,
                        denoise, sampler, scheduler, image):
          """Apply CLI parameter overrides to a workflow"""
          if prompt:
              workflow = set_prompt(workflow, prompt, negative)
          if seed is not None:
              workflow = set_seed(workflow, seed)
          if steps is not None:
              workflow = set_steps(workflow, steps)
          if cfg is not None:
              workflow = set_cfg(workflow, cfg)
          if width is not None and height is not None:
              workflow = set_dimensions(workflow, width, height)
          if denoise is not None:
              workflow = set_denoise(workflow, denoise)
          if sampler is not None:
              workflow = set_sampler(workflow, sampler)
          if scheduler is not None:
              workflow = set_scheduler(workflow, scheduler)
          if image:
              workflow = set_input_image(workflow, str(image))
          return workflow


      def _wait_and_download(client, prompt_id, output, label="", prefix=None):
          """Wait for completion with progress display, optionally download images"""
          plabel = f"{label} " if label else ""
          with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
              task = progress.add_task(f"{plabel}Queued...", total=None)

              def on_progress(msg_type, data):
                  if msg_type == "executing" and data.get("node"):
                      progress.update(task, description=f"{plabel}Executing node {data['node']}...")
                  elif msg_type == "progress":
                      val = data.get("value", 0)
                      mx = data.get("max", 0)
                      if mx > 0:
                          progress.update(task, description=f"{plabel}Sampling step {val}/{mx}...")

              result = client.wait_for_completion(prompt_id, on_progress=on_progress)

          if output and "outputs" in result:
              output.mkdir(parents=True, exist_ok=True)
              for node_id, node_output in result["outputs"].items():
                  for img in node_output.get("images", []):
                      data = client.get_image(img["filename"], img.get("subfolder", ""))
                      fname = img["filename"]
                      if prefix:
                          fname = f"{prefix}_{fname}"
                      (output / fname).write_bytes(data)
                      console.print(f"[green]Saved:[/green] {output / fname}")

          return result


      @app.command()
      def submit(
          workflow_path: Path,
          prompt: str = typer.Option(None, "--prompt", "-p", help="Positive prompt"),
          negative: str = typer.Option("", "--negative", "-n", help="Negative prompt"),
          seed: int = typer.Option(None, "--seed", "-s", help="Random seed"),
          steps: int = typer.Option(None, "--steps", help="Sampling steps"),
          cfg: float = typer.Option(None, "--cfg", help="CFG scale"),
          width: int = typer.Option(None, "--width", "-W", help="Image width"),
          height: int = typer.Option(None, "--height", "-H", help="Image height"),
          denoise: float = typer.Option(None, "--denoise", "-d", help="Denoise strength (0.0-1.0)"),
          sampler: str = typer.Option(None, "--sampler", help="Sampler name (euler, dpmpp_2m, etc.)"),
          scheduler: str = typer.Option(None, "--scheduler", help="Scheduler (normal, karras, etc.)"),
          image: Path = typer.Option(None, "--image", "-i", help="Input image (for img2img)"),
          wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
          output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
          count: int = typer.Option(1, "--count", "-c", help="Number of images to generate (varies seed)"),
          parallel: bool = typer.Option(False, "--parallel", help="Submit all jobs at once instead of sequentially"),
          prefix: str = typer.Option(None, "--prefix", help="Prefix for output filenames"),
      ):
          """Submit a workflow to ComfyUI"""
          client = get_client()
          workflow = load_workflow(workflow_path)
          workflow = _apply_params(workflow, prompt, negative, seed, steps, cfg,
                                   width, height, denoise, sampler, scheduler, image)

          if count <= 1:
              # Single image — original behavior
              prompt_id = client.submit(workflow)
              console.print(f"[green]Submitted:[/green] {prompt_id}")
              if wait:
                  _wait_and_download(client, prompt_id, output, prefix=prefix)
              return

          # Batch count: determine base seed
          base_seed = seed if seed is not None else get_seed(workflow)
          if base_seed is None:
              base_seed = 0

          if parallel:
              # Submit all at once, then wait
              prompt_ids = []
              for i in range(count):
                  wf = deepcopy(workflow)
                  wf = set_seed(wf, base_seed + i)
                  pid = client.submit(wf)
                  console.print(f"[green]Submitted {i+1}/{count}:[/green] {pid}")
                  prompt_ids.append(pid)
              if wait:
                  for i, pid in enumerate(prompt_ids):
                      _wait_and_download(client, pid, output, label=f"[{i+1}/{count}]", prefix=prefix)
          else:
              # Sequential: submit, wait, download one at a time
              for i in range(count):
                  wf = deepcopy(workflow)
                  wf = set_seed(wf, base_seed + i)
                  pid = client.submit(wf)
                  console.print(f"[green]Submitted {i+1}/{count}:[/green] {pid}")
                  if wait:
                      _wait_and_download(client, pid, output, label=f"[{i+1}/{count}]", prefix=prefix)


      @app.command()
      def queue():
          """Show current queue status"""
          client = get_client()
          q = client.get_queue()
          console.print(f"Running: {len(q.get('queue_running', []))}")
          console.print(f"Pending: {len(q.get('queue_pending', []))}")


      @app.command()
      def result(
          prompt_id: str,
          output: Path = typer.Option(".", "--output", "-o", help="Output directory"),
      ):
          """Get result for a prompt ID"""
          client = get_client()
          history = client.get_history(prompt_id)

          if prompt_id not in history:
              console.print(f"[red]Not found:[/red] {prompt_id}")
              raise typer.Exit(1)

          res = history[prompt_id]
          output.mkdir(parents=True, exist_ok=True)

          for node_id, node_output in res.get("outputs", {}).items():
              for img in node_output.get("images", []):
                  data = client.get_image(img["filename"], img.get("subfolder", ""))
                  (output / img["filename"]).write_bytes(data)
                  console.print(f"[green]Saved:[/green] {output / img['filename']}")


      @app.command()
      def batch(
          batch_file: Path,
          workflow_path: Path = typer.Option(..., "--workflow", "-W", help="Workflow JSON file"),
          output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
          parallel: bool = typer.Option(False, "--parallel", help="Submit all at once"),
          prefix: str = typer.Option(None, "--prefix", help="Prefix for output filenames"),
      ):
          """Run multiple jobs from a JSON batch file"""
          client = get_client()
          base_workflow = load_workflow(workflow_path)
          jobs = json.loads(batch_file.read_text())

          if not isinstance(jobs, list):
              console.print("[red]Error:[/red] Batch file must contain a JSON array")
              raise typer.Exit(1)

          total = len(jobs)

          if parallel:
              prompt_ids = []
              for i, job in enumerate(jobs):
                  wf = deepcopy(base_workflow)
                  wf = _apply_params(
                      wf,
                      prompt=job.get("prompt"),
                      negative=job.get("negative", ""),
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
                  pid = client.submit(wf)
                  console.print(f"[green]Submitted {i+1}/{total}:[/green] {pid}")
                  prompt_ids.append(pid)
              for i, pid in enumerate(prompt_ids):
                  _wait_and_download(client, pid, output, label=f"[{i+1}/{total}]", prefix=prefix)
          else:
              for i, job in enumerate(jobs):
                  wf = deepcopy(base_workflow)
                  wf = _apply_params(
                      wf,
                      prompt=job.get("prompt"),
                      negative=job.get("negative", ""),
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
                  pid = client.submit(wf)
                  console.print(f"[green]Submitted {i+1}/{total}:[/green] {pid}")
                  _wait_and_download(client, pid, output, label=f"[{i+1}/{total}]", prefix=prefix)


      @app.command()
      def cancel(
          prompt_ids: list[str] = typer.Argument(None),
          clear: bool = typer.Option(False, "--clear", help="Clear all pending jobs from the queue"),
          all_: bool = typer.Option(False, "--all", help="Interrupt running job and clear pending queue"),
      ):
          """Cancel running or pending ComfyUI jobs"""
          client = get_client()
          if all_:
              client.interrupt()
              client.clear_queue()
              console.print("[yellow]Interrupted running job and cleared queue[/yellow]")
          elif clear:
              client.clear_queue()
              console.print("[yellow]Cleared pending queue[/yellow]")
          elif prompt_ids:
              client.delete_from_queue(prompt_ids)
              for pid in prompt_ids:
                  console.print(f"[yellow]Removed from queue:[/yellow] {pid}")
          else:
              client.interrupt()
              console.print("[yellow]Interrupted running job[/yellow]")


      @app.command()
      def status():
          """Show ComfyUI server status and system info"""
          client = get_client()
          stats = client.get_system_stats()
          q = client.get_queue()

          sys_info = stats.get("system", {})
          info_lines = []
          if sys_info.get("comfyui_version"):
              info_lines.append(f"ComfyUI:  {sys_info['comfyui_version']}")
          if sys_info.get("pytorch_version"):
              info_lines.append(f"PyTorch:  {sys_info['pytorch_version']}")
          if sys_info.get("python_version"):
              info_lines.append(f"Python:   {sys_info['python_version']}")
          if sys_info.get("os"):
              info_lines.append(f"OS:       {sys_info['os']}")

          ram = sys_info.get("ram_total", 0)
          ram_free = sys_info.get("ram_free", 0)
          if ram > 0:
              info_lines.append(f"RAM:      {ram_free / (1024**3):.1f} GB free / {ram / (1024**3):.1f} GB total")

          if info_lines:
              console.print(Panel("\n".join(info_lines), title="Server Info"))

          devices = stats.get("devices", [])
          if devices:
              table = Table(title="Devices")
              table.add_column("Name")
              table.add_column("Type")
              table.add_column("VRAM Free")
              table.add_column("VRAM Total")
              for dev in devices:
                  vfree = dev.get("vram_free", 0)
                  vtotal = dev.get("vram_total", 0)
                  table.add_row(
                      dev.get("name", "unknown"),
                      dev.get("type", "unknown"),
                      f"{vfree / (1024**3):.1f} GB",
                      f"{vtotal / (1024**3):.1f} GB",
                  )
              console.print(table)

          running = len(q.get("queue_running", []))
          pending = len(q.get("queue_pending", []))
          console.print(f"\nQueue: [bold]{running}[/bold] running, [bold]{pending}[/bold] pending")


      @app.command()
      def models(
          folder: str = typer.Argument(None, help="Model folder (checkpoints, loras, vae, etc.)"),
      ):
          """List available models or model types"""
          client = get_client()
          if folder:
              items = client.get_models(folder)
              if not items:
                  console.print(f"[yellow]No models found in {folder}[/yellow]")
              else:
                  for item in sorted(items):
                      console.print(item)
          else:
              types = client.get_model_types()
              for t in sorted(types):
                  console.print(t)


      @app.command()
      def info(
          image_path: Path = typer.Argument(..., help="Path to a ComfyUI-generated PNG image"),
          json_output: bool = typer.Option(False, "--json", "-j", help="Output full metadata as JSON"),
      ):
          """Display generation metadata from a ComfyUI PNG image"""
          if not image_path.exists():
              console.print(f"[red]Error:[/red] File not found: {image_path}")
              raise typer.Exit(1)

          metadata = extract_comfyui_metadata(image_path)
          if not metadata:
              console.print("[yellow]No ComfyUI metadata found in image[/yellow]")
              raise typer.Exit(1)

          if json_output:
              console.print(json.dumps(metadata, indent=2))
              return

          summary = summarize_metadata(metadata)
          if not summary:
              console.print("[yellow]Could not extract generation parameters[/yellow]")
              console.print("Use --json to see raw metadata")
              raise typer.Exit(1)

          if "model" in summary:
              console.print(f"[bold]Model:[/bold]    {summary['model']}")
          if "positive_prompt" in summary:
              console.print(f"[bold]Prompt:[/bold]   {summary['positive_prompt']}")
          if "negative_prompt" in summary:
              console.print(f"[bold]Negative:[/bold] {summary['negative_prompt']}")
          if "seed" in summary:
              console.print(f"[bold]Seed:[/bold]     {summary['seed']}")
          if "steps" in summary:
              console.print(f"[bold]Steps:[/bold]    {summary['steps']}")
          if "cfg" in summary:
              console.print(f"[bold]CFG:[/bold]      {summary['cfg']}")
          if "sampler" in summary:
              console.print(f"[bold]Sampler:[/bold]  {summary['sampler']}")
          if "scheduler" in summary:
              console.print(f"[bold]Scheduler:[/bold] {summary['scheduler']}")
          if "denoise" in summary:
              console.print(f"[bold]Denoise:[/bold]  {summary['denoise']}")
          if "width" in summary and "height" in summary:
              console.print(f"[bold]Size:[/bold]     {summary['width']}x{summary['height']}")


      def main():
          app()


      # Create separate apps for standalone entry points
      submit_app = typer.Typer()
      queue_app = typer.Typer()
      result_app = typer.Typer()
      batch_app = typer.Typer()
      cancel_app = typer.Typer()
      status_app = typer.Typer()
      models_app = typer.Typer()
      info_app = typer.Typer()

      submit_app.command()(submit)
      queue_app.command()(queue)
      result_app.command()(result)
      batch_app.command()(batch)
      cancel_app.command()(cancel)
      status_app.command()(status)
      models_app.command()(models)
      info_app.command()(info)


      def submit_cli():
          """Entry point for comfyui-submit"""
          submit_app()


      def queue_cli():
          """Entry point for comfyui-queue"""
          queue_app()


      def result_cli():
          """Entry point for comfyui-result"""
          result_app()


      def batch_cli():
          """Entry point for comfyui-batch"""
          batch_app()


      def cancel_cli():
          """Entry point for comfyui-cancel"""
          cancel_app()


      def status_cli():
          """Entry point for comfyui-status"""
          status_app()


      def models_cli():
          """Entry point for comfyui-models"""
          models_app()


      def info_cli():
          """Entry point for comfyui-info"""
          info_app()
    '';
  };

  # Bundle Python source into a directory structure
  pythonSource = runCommand "comfyui-client-source" {} ''
    mkdir -p $out/share/comfyui-client/src/comfyui_client
    cp ${pyproject} $out/share/comfyui-client/pyproject.toml
    cp ${initPy} $out/share/comfyui-client/src/comfyui_client/__init__.py
    cp ${clientPy} $out/share/comfyui-client/src/comfyui_client/client.py
    cp ${workflowPy} $out/share/comfyui-client/src/comfyui_client/workflow.py
    cp ${metadataPy} $out/share/comfyui-client/src/comfyui_client/metadata.py
    cp ${cliPy} $out/share/comfyui-client/src/comfyui_client/cli.py

    cat > $out/share/comfyui-client/.flox-build-v2 << 'FLOX_BUILD'
    FLOX_BUILD_RUNTIME_VERSION=2
    description: Client tools release
    date: 2026-02-23
    change:
      Add --prefix output naming to submit and batch.
      Add comfyui-cancel (interrupt/clear/delete from queue).
      Add comfyui-status (server info, RAM, GPU, queue counts).
      Add comfyui-models (list model folders and models).
      Add comfyui-info (PNG metadata reader, no server needed).
      Add bash tab completions for all commands and wrappers.
    FLOX_BUILD
  '';

  # Man pages
  manSubmit = writeTextFile {
    name = "comfyui-submit.1";
    text = ''
      .TH COMFYUI-SUBMIT 1 "2026-02-21" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-submit \- submit a workflow to ComfyUI
      .SH SYNOPSIS
      .B comfyui-submit
      .I workflow_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-submit
      submits a ComfyUI workflow JSON file to a running ComfyUI server for execution.
      The workflow can be modified on-the-fly using command-line options to set
      prompts, seeds, dimensions, and other generation parameters.
      .SH ARGUMENTS
      .TP
      .I workflow_path
      Path to the ComfyUI workflow JSON file. This should be a workflow exported
      in API format (not the standard web UI format).
      .SH OPTIONS
      .TP
      .BR \-p ", " \-\-prompt " " \fITEXT\fR
      Set the positive prompt text. This replaces the text in CLIPTextEncode nodes
      that have "positive" or "prompt" in their title.
      .TP
      .BR \-n ", " \-\-negative " " \fITEXT\fR
      Set the negative prompt text. This replaces the text in CLIPTextEncode nodes
      that have "negative" in their title.
      .TP
      .BR \-s ", " \-\-seed " " \fIINT\fR
      Set the random seed for generation. Applied to KSampler, KSamplerAdvanced,
      and SamplerCustom nodes.
      .TP
      .BR \-\-steps " " \fIINT\fR
      Set the number of sampling steps.
      .TP
      .BR \-\-cfg " " \fIFLOAT\fR
      Set the CFG (Classifier-Free Guidance) scale. Higher values follow the prompt
      more closely. Typical range is 5.0-15.0.
      .TP
      .BR \-W ", " \-\-width " " \fIINT\fR
      Set the image width in pixels. Must be used together with \fB\-\-height\fR.
      .TP
      .BR \-H ", " \-\-height " " \fIINT\fR
      Set the image height in pixels. Must be used together with \fB\-\-width\fR.
      .TP
      .BR \-d ", " \-\-denoise " " \fIFLOAT\fR
      Set the denoise strength for img2img workflows. Range is 0.0 (no change) to
      1.0 (full regeneration). Typical values for img2img are 0.4-0.8.
      .TP
      .BR \-\-sampler " " \fINAME\fR
      Set the sampler algorithm. Common values: euler, euler_ancestral, heun,
      dpmpp_2m, dpmpp_2m_sde, dpmpp_3m_sde, uni_pc, ddim.
      .TP
      .BR \-\-scheduler " " \fINAME\fR
      Set the scheduler type. Values: normal, karras, exponential, sgm_uniform,
      simple, ddim_uniform, beta.
      .TP
      .BR \-i ", " \-\-image " " \fIPATH\fR
      Set the input image path for img2img workflows. This sets the image in
      LoadImage nodes.
      .TP
      .BR \-w ", " \-\-wait
      Wait for the workflow to complete before exiting. Uses a WebSocket connection
      to monitor execution progress in real time, showing which node is running
      and sampling step counts. Without this flag, the command exits immediately
      after submission.
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading generated images. Only effective when used
      with \fB\-\-wait\fR. Images are saved with their original filenames.
      .TP
      .BR \-c ", " \-\-count " " \fIN\fR
      Number of images to generate. The seed is auto-incremented for each
      variation (base, base+1, base+2, ...). The base seed is taken from
      \fB\-\-seed\fR if specified, otherwise from the workflow's current seed.
      Default: 1
      .TP
      .BR \-\-parallel
      When used with \fB\-\-count\fR, submit all jobs at once instead of
      sequentially. Without this flag, each job is submitted, waited on (if
      \fB\-\-wait\fR), and downloaded before the next one starts.
      .TP
      .BR \-\-prefix " " \fITEXT\fR
      Prefix string prepended to output filenames. When set, downloaded images
      are saved as \fIprefix\fR_\fIoriginal_filename\fR instead of just
      \fIoriginal_filename\fR.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH REMOTE OPERATION
      This command uses HTTP for workflow submission and image retrieval, and
      WebSocket for real-time completion monitoring (with \fB\-\-wait\fR). When using
      \fB\-\-output\fR, generated images are downloaded from the server via the
      ComfyUI \fB/view\fR endpoint. No filesystem access to the ComfyUI server
      is required.
      .PP
      This means you can run the client on a separate machine from ComfyUI:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-submit workflow.json \\
          --wait -o ./local-output
      .fi
      .RE
      .PP
      The workflow executes on the remote server, and the resulting images are
      transferred to your local \fB./local-output\fR directory over HTTP.
      .SH EXAMPLES
      Submit a workflow and download results:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json --wait -o ./output
      .fi
      .RE
      .PP
      Generate with custom prompt, seed, and dimensions:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a serene mountain landscape at sunset" \\
          -n "blurry, low quality" \\
          -s 42 -W 1024 -H 768 --wait -o ./output
      .fi
      .RE
      .PP
      Img2img with input image and denoise strength:
      .PP
      .RS
      .nf
      comfyui-submit img2img.json \\
          -i input.png -d 0.6 --wait -o ./output
      .fi
      .RE
      .PP
      Fine-tune sampling parameters:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "detailed portrait" --steps 30 --cfg 7.5 \\
          --sampler dpmpp_2m --scheduler karras \\
          --wait -o ./output
      .fi
      .RE
      .PP
      Submit without waiting (async), retrieve later:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json -p "async job"
      # Outputs: Submitted: a1b2c3d4-e5f6-7890-abcd-ef1234567890

      # Later, retrieve results:
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./output
      .fi
      .RE
      .PP
      Generate 5 seed variations sequentially:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a cat in space" -s 100 --count 5 --wait -o ./output
      .fi
      .RE
      .PP
      Generate 10 variations in parallel:
      .PP
      .RS
      .nf
      comfyui-submit workflow.json \\
          -p "a cat in space" --count 10 --parallel --wait -o ./output
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-batch (1),
      .BR comfyui-client (7)
    '';
  };

  manQueue = writeTextFile {
    name = "comfyui-queue.1";
    text = ''
      .TH COMFYUI-QUEUE 1 "2026-02-21" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-queue \- show ComfyUI queue status
      .SH SYNOPSIS
      .B comfyui-queue
      .SH DESCRIPTION
      .B comfyui-queue
      displays the current status of the ComfyUI execution queue, showing the
      number of running and pending workflows.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH OUTPUT
      The command outputs two lines:
      .PP
      .RS
      .nf
      Running: N
      Pending: M
      .fi
      .RE
      .PP
      Where N is the number of currently executing workflows and M is the number
      of workflows waiting in the queue.
      .SH EXAMPLES
      Check queue status:
      .PP
      .RS
      .nf
      comfyui-queue
      .fi
      .RE
      .PP
      Check queue on a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=server.local comfyui-queue
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-result (1),
      .BR comfyui-client (7)
    '';
  };

  manResult = writeTextFile {
    name = "comfyui-result.1";
    text = ''
      .TH COMFYUI-RESULT 1 "2026-02-21" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-result \- retrieve results from a ComfyUI workflow
      .SH SYNOPSIS
      .B comfyui-result
      .I prompt_id
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-result
      retrieves and downloads the output images from a previously submitted
      ComfyUI workflow. The workflow must have completed successfully.
      .SH ARGUMENTS
      .TP
      .I prompt_id
      The prompt ID returned by
      .BR comfyui-submit (1)
      when the workflow was submitted. This is a UUID that uniquely identifies
      the workflow execution.
      .SH OPTIONS
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading images. Default: current directory (.)
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH REMOTE OPERATION
      Images are downloaded from the ComfyUI server via the \fB/view\fR HTTP
      endpoint. No filesystem access to the server is required. See
      .BR comfyui-submit (1)
      for details.
      .SH EXAMPLES
      Download results to current directory:
      .PP
      .RS
      .nf
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890
      .fi
      .RE
      .PP
      Download to a specific directory:
      .PP
      .RS
      .nf
      comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./images
      .fi
      .RE
      .PP
      Retrieve from a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local \\
          comfyui-result a1b2c3d4-e5f6-7890-abcd-ef1234567890 -o ./images
      .fi
      .RE
      .SH EXIT STATUS
      .TP
      .B 0
      Success
      .TP
      .B 1
      Prompt ID not found in history
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-client (7)
    '';
  };

  manOverview = writeTextFile {
    name = "comfyui-client.7";
    text = ''
      .TH COMFYUI-CLIENT 7 "2026-02-21" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-client \- command-line interface for ComfyUI
      .SH DESCRIPTION
      The comfyui-client package provides command-line tools for interacting with
      a ComfyUI server. It includes core commands for workflow submission and
      management, as well as convenient wrapper scripts for common operations.
      .SH COMMANDS
      The package provides these core commands:
      .TP
      .BR comfyui-submit (1)
      Submit a workflow JSON file to ComfyUI with optional parameter overrides.
      Supports \fB\-\-count\fR for generating multiple seed variations.
      .TP
      .BR comfyui-batch (1)
      Run multiple jobs with different parameters from a JSON batch file.
      .TP
      .BR comfyui-queue (1)
      Display the current queue status.
      .TP
      .BR comfyui-result (1)
      Retrieve output images from a completed workflow.
      .TP
      .BR comfyui-cancel (1)
      Cancel running or pending jobs.
      .TP
      .BR comfyui-status (1)
      Show server status, versions, RAM, and GPU info.
      .TP
      .BR comfyui-models (1)
      List available models by folder type.
      .TP
      .BR comfyui-info (1)
      Display generation metadata from ComfyUI PNG images.
      .SH WRAPPER SCRIPTS
      For convenience, the package includes wrapper scripts that automatically
      select the appropriate workflow file for common model/operation combinations.
      Each wrapper calls
      .B comfyui-submit
      with the \fB\-\-wait\fR flag and passes through all other arguments.
      .SS Stable Diffusion 1.5
      .TP
      .B sd15-txt2img
      Text-to-image generation using SD 1.5
      .TP
      .B sd15-img2img
      Image-to-image generation using SD 1.5
      .TP
      .B sd15-upscale
      Upscaling using SD 1.5
      .TP
      .B sd15-inpaint
      Inpainting using SD 1.5
      .SS Stable Diffusion XL
      .TP
      .B sdxl-txt2img
      Text-to-image generation using SDXL
      .TP
      .B sdxl-img2img
      Image-to-image generation using SDXL
      .TP
      .B sdxl-upscale
      Upscaling using SDXL
      .TP
      .B sdxl-inpaint
      Inpainting using SDXL
      .SS Stable Diffusion 3.5
      .TP
      .B sd35-txt2img
      Text-to-image generation using SD 3.5
      .TP
      .B sd35-img2img
      Image-to-image generation using SD 3.5
      .TP
      .B sd35-upscale
      Upscaling using SD 3.5
      .TP
      .B sd35-inpaint
      Inpainting using SD 3.5
      .SS FLUX
      .TP
      .B flux-txt2img
      Text-to-image generation using FLUX
      .TP
      .B flux-img2img
      Image-to-image generation using FLUX
      .TP
      .B flux-upscale
      Upscaling using FLUX
      .TP
      .B flux-inpaint
      Inpainting using FLUX
      .SH WORKFLOW FILES
      The package bundles workflow JSON files for all 16 model/operation combinations.
      Wrapper scripts find them automatically at:
      .PP
      .RS
      .nf
      $FLOX_ENV/share/comfyui-client/workflows/api/<model>/<model>-<operation>.json
      .fi
      .RE
      .PP
      For example,
      .B sdxl-txt2img
      loads:
      .PP
      .RS
      .nf
      $FLOX_ENV/share/comfyui-client/workflows/api/sdxl/sdxl-txt2img.json
      .fi
      .RE
      .PP
      To use custom workflows instead, set \fBCOMFYUI_WORKFLOWS\fR to a directory
      containing your own workflow tree. This overrides the bundled workflows.
      .PP
      Workflow files must be in ComfyUI API format (exported via "Save (API Format)"
      in the ComfyUI web interface or converted from standard format).
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .TP
      .B COMFYUI_WORKFLOWS
      Base directory containing workflow files. Overrides the bundled workflows.
      Default: $FLOX_ENV/share/comfyui-client/workflows
      .SH REMOTE OPERATION
      Commands use HTTP for workflow submission, queue queries, and image retrieval.
      The \fB\-\-wait\fR flag uses a WebSocket connection for real-time progress
      monitoring. The client can run on a different machine from the ComfyUI server.
      When downloading images with the \fB\-o\fR option, images are fetched via the
      ComfyUI \fB/view\fR endpoint and saved locally. No filesystem access to the
      server is required.
      .PP
      This architecture supports using ComfyUI as a remote image generation
      service, where the GPU server runs ComfyUI and clients submit workflows
      from separate machines.
      .SH EXAMPLES
      Generate an image with SDXL:
      .PP
      .RS
      .nf
      sdxl-txt2img -p "a beautiful sunset over mountains" -o ./output
      .fi
      .RE
      .PP
      Img2img with SD 1.5 (modify existing image):
      .PP
      .RS
      .nf
      sd15-img2img -i photo.png -p "oil painting style" -d 0.5 -o ./output
      .fi
      .RE
      .PP
      Upscale an image with FLUX:
      .PP
      .RS
      .nf
      flux-upscale -i lowres.png -o ./upscaled
      .fi
      .RE
      .PP
      Generate on a remote GPU server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local \\
          sdxl-txt2img -p "portrait photo" -s 12345 -o ./output
      .fi
      .RE
      .PP
      Use a custom workflow directly:
      .PP
      .RS
      .nf
      comfyui-submit ~/workflows/custom.json \\
          -p "my prompt" --wait -o ./output
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-batch (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-cancel (1),
      .BR comfyui-status (1),
      .BR comfyui-models (1),
      .BR comfyui-info (1)
    '';
  };

  manBatch = writeTextFile {
    name = "comfyui-batch.1";
    text = ''
      .TH COMFYUI-BATCH 1 "2026-02-22" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-batch \- run multiple ComfyUI jobs from a batch file
      .SH SYNOPSIS
      .B comfyui-batch
      .I batch_file
      .B \-\-workflow
      .I workflow_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-batch
      reads a JSON file containing an array of job objects and submits each one
      to a running ComfyUI server. Each job can override prompt, seed, steps, and
      other generation parameters. All jobs share the same base workflow.
      .PP
      By default, jobs run sequentially: each is submitted, waited on, and its
      images downloaded before the next job starts. With \fB\-\-parallel\fR, all
      jobs are submitted at once and then awaited in order.
      .SH ARGUMENTS
      .TP
      .I batch_file
      Path to a JSON file containing an array of job objects. See
      .B BATCH FILE FORMAT
      below.
      .SH OPTIONS
      .TP
      .BR \-W ", " \-\-workflow " " \fIPATH\fR
      Path to the base workflow JSON file (required). This should be a workflow
      exported in API format.
      .TP
      .BR \-o ", " \-\-output " " \fIDIR\fR
      Output directory for downloading generated images.
      .TP
      .BR \-\-parallel
      Submit all jobs at once instead of sequentially.
      .TP
      .BR \-\-prefix " " \fITEXT\fR
      Prefix string prepended to output filenames.
      .SH BATCH FILE FORMAT
      The batch file must contain a JSON array of objects. Each object can have
      the following optional keys:
      .PP
      .RS
      .nf
      prompt      Positive prompt text (string)
      negative    Negative prompt text (string)
      seed        Random seed (integer)
      steps       Number of sampling steps (integer)
      cfg         CFG scale (float)
      width       Image width in pixels (integer)
      height      Image height in pixels (integer)
      denoise     Denoise strength 0.0-1.0 (float)
      sampler     Sampler algorithm name (string)
      scheduler   Scheduler type (string)
      image       Input image path for img2img (string)
      .fi
      .RE
      .PP
      Any key not present in a job object will use the workflow default.
      .SH EXAMPLES
      Simple batch file with three prompts:
      .PP
      .RS
      .nf
      [
        {"prompt": "oil painting of mountains", "seed": 42},
        {"prompt": "watercolor of ocean", "steps": 50},
        {"prompt": "digital art of forest"}
      ]
      .fi
      .RE
      .PP
      Run a batch file:
      .PP
      .RS
      .nf
      comfyui-batch jobs.json -W workflow.json -o ./output
      .fi
      .RE
      .PP
      Run in parallel:
      .PP
      .RS
      .nf
      comfyui-batch jobs.json -W workflow.json --parallel -o ./output
      .fi
      .RE
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
      .BR comfyui-client (7)
    '';
  };

  manStatus = writeTextFile {
    name = "comfyui-status.1";
    text = ''
      .TH COMFYUI-STATUS 1 "2026-02-23" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-status \- show ComfyUI server status and system info
      .SH SYNOPSIS
      .B comfyui-status
      .SH DESCRIPTION
      .B comfyui-status
      displays the ComfyUI server status including software versions, RAM usage,
      GPU devices with VRAM, and current queue counts.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Check local server status:
      .PP
      .RS
      .nf
      comfyui-status
      .fi
      .RE
      .PP
      Check remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-status
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-queue (1),
      .BR comfyui-client (7)
    '';
  };

  manModels = writeTextFile {
    name = "comfyui-models.1";
    text = ''
      .TH COMFYUI-MODELS 1 "2026-02-23" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-models \- list available ComfyUI models
      .SH SYNOPSIS
      .B comfyui-models
      .RI [ FOLDER ]
      .SH DESCRIPTION
      .B comfyui-models
      lists available model types or models within a specific folder. Without
      arguments, it lists all model folder types (checkpoints, loras, vae, etc.).
      With a folder argument, it lists all models in that folder.
      .SH ARGUMENTS
      .TP
      .I FOLDER
      Model folder type to list. Common values: checkpoints, loras, vae,
      controlnet, clip, clip_vision, upscale_models, embeddings.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      List all model folder types:
      .PP
      .RS
      .nf
      comfyui-models
      .fi
      .RE
      .PP
      List available checkpoints:
      .PP
      .RS
      .nf
      comfyui-models checkpoints
      .fi
      .RE
      .PP
      List LoRA models on a remote server:
      .PP
      .RS
      .nf
      COMFYUI_HOST=gpu-server.local comfyui-models loras
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-client (7)
    '';
  };

  manInfo = writeTextFile {
    name = "comfyui-info.1";
    text = ''
      .TH COMFYUI-INFO 1 "2026-02-23" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-info \- display generation metadata from ComfyUI PNG images
      .SH SYNOPSIS
      .B comfyui-info
      .I image_path
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-info
      reads ComfyUI metadata embedded in PNG images and displays the generation
      parameters. ComfyUI stores workflow and prompt data as PNG text chunks.
      This is a local command and does not require a running ComfyUI server.
      .SH ARGUMENTS
      .TP
      .I image_path
      Path to a ComfyUI-generated PNG image file.
      .SH OPTIONS
      .TP
      .BR \-j ", " \-\-json
      Output the full metadata (prompt and workflow) as JSON instead of the
      human-readable summary.
      .SH OUTPUT
      By default, displays a summary of generation parameters:
      .PP
      .RS
      .nf
      Model:     sd_xl_base_1.0.safetensors
      Prompt:    a beautiful landscape
      Seed:      42
      Steps:     20
      CFG:       7.0
      Sampler:   euler
      Scheduler: normal
      Size:      1024x1024
      .fi
      .RE
      .PP
      With \fB\-\-json\fR, outputs the complete ComfyUI prompt and workflow
      metadata as a JSON object.
      .SH EXAMPLES
      Show generation info for an image:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png
      .fi
      .RE
      .PP
      Dump full metadata as JSON:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png --json
      .fi
      .RE
      .PP
      Pipe JSON to jq for inspection:
      .PP
      .RS
      .nf
      comfyui-info output/ComfyUI_00001_.png -j | jq '.prompt'
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-client (7)
    '';
  };

  manCancel = writeTextFile {
    name = "comfyui-cancel.1";
    text = ''
      .TH COMFYUI-CANCEL 1 "2026-02-23" "comfyui-client 0.1.0" "ComfyUI Client Manual"
      .SH NAME
      comfyui-cancel \- cancel running or pending ComfyUI jobs
      .SH SYNOPSIS
      .B comfyui-cancel
      .RI [ ID... ]
      .RI [ OPTIONS ]
      .SH DESCRIPTION
      .B comfyui-cancel
      cancels ComfyUI jobs. With no arguments, it interrupts the currently
      running workflow. Specific pending jobs can be removed by prompt ID.
      .SH ARGUMENTS
      .TP
      .I ID...
      One or more prompt IDs to remove from the pending queue.
      .SH OPTIONS
      .TP
      .BR \-\-clear
      Clear all pending jobs from the queue (does not interrupt the running job).
      .TP
      .BR \-\-all
      Interrupt the running job and clear all pending jobs.
      .SH ENVIRONMENT
      .TP
      .B COMFYUI_HOST
      Hostname of the ComfyUI server. Default: localhost
      .TP
      .B COMFYUI_PORT
      Port of the ComfyUI server. Default: 8188
      .SH EXAMPLES
      Interrupt the currently running job:
      .PP
      .RS
      .nf
      comfyui-cancel
      .fi
      .RE
      .PP
      Remove specific jobs from the queue:
      .PP
      .RS
      .nf
      comfyui-cancel a1b2c3d4-... e5f6a7b8-...
      .fi
      .RE
      .PP
      Clear the entire pending queue:
      .PP
      .RS
      .nf
      comfyui-cancel --clear
      .fi
      .RE
      .PP
      Stop everything:
      .PP
      .RS
      .nf
      comfyui-cancel --all
      .fi
      .RE
      .SH SEE ALSO
      .BR comfyui-submit (1),
      .BR comfyui-queue (1),
      .BR comfyui-client (7)
    '';
  };

  # Bundle workflow files
  workflowFiles = runCommand "comfyui-workflows" {} ''
    mkdir -p $out/share/comfyui-client/workflows/api
    cp -r ${./../../workflows/api}/* $out/share/comfyui-client/workflows/api/
  '';

  # Bundle man pages
  manPages = runCommand "comfyui-client-man" {} ''
    mkdir -p $out/share/man/man1 $out/share/man/man7
    cp ${manSubmit} $out/share/man/man1/comfyui-submit.1
    cp ${manQueue} $out/share/man/man1/comfyui-queue.1
    cp ${manResult} $out/share/man/man1/comfyui-result.1
    cp ${manBatch} $out/share/man/man1/comfyui-batch.1
    cp ${manCancel} $out/share/man/man1/comfyui-cancel.1
    cp ${manStatus} $out/share/man/man1/comfyui-status.1
    cp ${manModels} $out/share/man/man1/comfyui-models.1
    cp ${manInfo} $out/share/man/man1/comfyui-info.1
    cp ${manOverview} $out/share/man/man7/comfyui-client.7
  '';

  # Setup script to install Python package into venv
  setupScript = writeShellApplication {
    name = "comfyui-client-setup";
    text = ''
      VENV="''${1:-$FLOX_ENV_CACHE/venv}"
      SOURCE_DIR="''${FLOX_ENV}/share/comfyui-client"

      # Print build recipe version marker
      flox_build_marker=$(find "$SOURCE_DIR" -maxdepth 1 -name '.flox-build-v*' -print -quit)
      if [ -n "$flox_build_marker" ]; then
        echo "=============================================="
        echo "FLOX_BUILD_RUNTIME_VERSION: $(basename "$flox_build_marker" | sed 's/.flox-build-v//')"
        echo "Source: $SOURCE_DIR"
        echo "=============================================="
      fi

      if [ ! -d "$VENV" ]; then
        echo "Error: venv not found at $VENV" >&2
        exit 1
      fi

      if [ ! -d "$SOURCE_DIR" ]; then
        echo "Error: source not found at $SOURCE_DIR" >&2
        exit 1
      fi

      # Install using uv/pip from the venv
      if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VENV/bin/python" "$SOURCE_DIR" --quiet
      else
        "$VENV/bin/pip" install "$SOURCE_DIR" --quiet
      fi
    '';
  };

  bashCompletions = writeTextFile {
    name = "comfyui-completions.bash";
    text = ''
      # Bash completions for comfyui-client commands

      _comfyui_submit_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"

          opts="--prompt --negative --seed --steps --cfg --width --height --denoise --sampler --scheduler --image --wait --output --count --parallel --prefix --help"

          case "$prev" in
              --sampler)
                  COMPREPLY=( $(compgen -W "euler euler_ancestral heun dpmpp_2m dpmpp_2m_sde dpmpp_3m_sde uni_pc ddim" -- "$cur") )
                  return 0
                  ;;
              --scheduler)
                  COMPREPLY=( $(compgen -W "normal karras exponential sgm_uniform simple ddim_uniform beta" -- "$cur") )
                  return 0
                  ;;
              --image|--output|-i|-o)
                  COMPREPLY=( $(compgen -f -- "$cur") )
                  return 0
                  ;;
              --prompt|--negative|--seed|--steps|--cfg|--width|--height|--denoise|--count|--prefix|-p|-n|-s|-W|-H|-d|-c)
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -- "$cur") )
          fi
      }

      _comfyui_cancel_completions() {
          local cur opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          opts="--clear --all --help"

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          fi
      }

      _comfyui_models_completions() {
          local cur folders
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          folders="checkpoints loras vae controlnet clip clip_vision upscale_models embeddings hypernetworks"

          if [[ "$cur" != -* ]]; then
              COMPREPLY=( $(compgen -W "$folders" -- "$cur") )
          else
              COMPREPLY=( $(compgen -W "--help" -- "$cur") )
          fi
      }

      _comfyui_info_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          opts="--json --help"

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -X '!*.png' -- "$cur") )
          fi
      }

      _comfyui_batch_completions() {
          local cur prev opts
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          prev="''${COMP_WORDS[COMP_CWORD-1]}"
          opts="--workflow --output --parallel --prefix --help"

          case "$prev" in
              --workflow|-W|--output|-o)
                  COMPREPLY=( $(compgen -f -- "$cur") )
                  return 0
                  ;;
          esac

          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
          else
              COMPREPLY=( $(compgen -f -- "$cur") )
          fi
      }

      _comfyui_simple_completions() {
          local cur
          COMPREPLY=()
          cur="''${COMP_WORDS[COMP_CWORD]}"
          if [[ "$cur" == -* ]]; then
              COMPREPLY=( $(compgen -W "--help" -- "$cur") )
          fi
      }

      # Register completions
      complete -F _comfyui_submit_completions comfyui-submit
      complete -F _comfyui_batch_completions comfyui-batch
      complete -F _comfyui_cancel_completions comfyui-cancel
      complete -F _comfyui_models_completions comfyui-models
      complete -F _comfyui_info_completions comfyui-info
      complete -F _comfyui_simple_completions comfyui-queue
      complete -F _comfyui_simple_completions comfyui-status
      complete -F _comfyui_simple_completions comfyui-result

      # Register wrapper scripts (same options as submit)
      complete -F _comfyui_submit_completions sd15-txt2img
      complete -F _comfyui_submit_completions sd15-img2img
      complete -F _comfyui_submit_completions sd15-upscale
      complete -F _comfyui_submit_completions sd15-inpaint
      complete -F _comfyui_submit_completions sdxl-txt2img
      complete -F _comfyui_submit_completions sdxl-img2img
      complete -F _comfyui_submit_completions sdxl-upscale
      complete -F _comfyui_submit_completions sdxl-inpaint
      complete -F _comfyui_submit_completions sd35-txt2img
      complete -F _comfyui_submit_completions sd35-img2img
      complete -F _comfyui_submit_completions sd35-upscale
      complete -F _comfyui_submit_completions sd35-inpaint
      complete -F _comfyui_submit_completions flux-txt2img
      complete -F _comfyui_submit_completions flux-img2img
      complete -F _comfyui_submit_completions flux-upscale
      complete -F _comfyui_submit_completions flux-inpaint
    '';
  };

  completionFiles = runCommand "comfyui-completions" {} ''
    mkdir -p $out/share/bash-completion/completions
    cp ${bashCompletions} $out/share/bash-completion/completions/comfyui-submit
    for cmd in comfyui-batch comfyui-cancel comfyui-status comfyui-models comfyui-info \
               comfyui-queue comfyui-result \
               sd15-txt2img sd15-img2img sd15-upscale sd15-inpaint \
               sdxl-txt2img sdxl-img2img sdxl-upscale sdxl-inpaint \
               sd35-txt2img sd35-img2img sd35-upscale sd35-inpaint \
               flux-txt2img flux-img2img flux-upscale flux-inpaint; do
      ln -s comfyui-submit $out/share/bash-completion/completions/$cmd
    done
  '';

in
symlinkJoin {
  name = "comfyui-scripts";
  paths = allScripts ++ [ pythonSource setupScript manPages workflowFiles completionFiles ];
  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows (sd15, sdxl, sd35, flux)";
  };
}
