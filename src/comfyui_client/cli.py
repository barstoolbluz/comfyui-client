"""CLI commands for ComfyUI client"""
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import os

from .client import ComfyUIClient
from .workflow import (
    load_workflow,
    set_prompt,
    set_seed,
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
):
    """Submit a workflow to ComfyUI"""
    client = get_client()
    workflow = load_workflow(workflow_path)

    # Apply all specified parameters
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

    prompt_id = client.submit(workflow)
    console.print(f"[green]Submitted:[/green] {prompt_id}")

    if wait:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Queued...", total=None)

            def on_progress(msg_type, data):
                if msg_type == "executing" and data.get("node"):
                    progress.update(task, description=f"Executing node {data['node']}...")
                elif msg_type == "progress":
                    val = data.get("value", 0)
                    mx = data.get("max", 0)
                    if mx > 0:
                        progress.update(task, description=f"Sampling step {val}/{mx}...")

            result = client.wait_for_completion(prompt_id, on_progress=on_progress)

        # Download images if output specified
        if output and "outputs" in result:
            output.mkdir(parents=True, exist_ok=True)
            for node_id, node_output in result["outputs"].items():
                for img in node_output.get("images", []):
                    data = client.get_image(img["filename"], img.get("subfolder", ""))
                    (output / img["filename"]).write_bytes(data)
                    console.print(f"[green]Saved:[/green] {output / img['filename']}")


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


def main():
    app()


# Create separate apps for standalone entry points
submit_app = typer.Typer()
queue_app = typer.Typer()
result_app = typer.Typer()

submit_app.command()(submit)
queue_app.command()(queue)
result_app.command()(result)


def submit_cli():
    """Entry point for comfyui-submit"""
    submit_app()


def queue_cli():
    """Entry point for comfyui-queue"""
    queue_app()


def result_cli():
    """Entry point for comfyui-result"""
    result_app()
