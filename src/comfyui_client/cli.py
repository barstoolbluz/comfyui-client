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
