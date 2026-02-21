{ lib, writeShellApplication, writeTextFile, symlinkJoin, runCommand }:

let
  # Model configurations: script prefix -> directory name
  models = {
    sd15 = "SD15";
    sdxl = "SDXL";
    sd35 = "SD35";
    flux = "FLUX";
  };

  operations = [ "txt2img" "img2img" "upscale" ];

  # Generate a single workflow wrapper script
  makeScript = model: op:
    let
      dir = models.${model};
      name = "${model}-${op}";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOWS_BASE="''${COMFYUI_WORKFLOWS:-$HOME/comfyui-work/user/default/workflows}"
        WORKFLOW="$WORKFLOWS_BASE/${dir}/${model}-${op}.json"

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
    lib.mapAttrsToList (model: _:
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

      __all__ = [
          "ComfyUIClient",
          "load_workflow",
          "set_prompt",
          "set_seed",
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
      import httpx
      import uuid


      class ComfyUIClient:
          def __init__(self, host: str = "localhost", port: int = 8188):
              self.base_url = f"http://{host}:{port}"
              self.client_id = str(uuid.uuid4())

          def submit(self, workflow: dict) -> str:
              """Submit workflow, return prompt_id"""
              response = httpx.post(
                  f"{self.base_url}/prompt",
                  json={"prompt": workflow, "client_id": self.client_id}
              )
              response.raise_for_status()
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

          def get_image(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
              """Download image from ComfyUI"""
              response = httpx.get(
                  f"{self.base_url}/view",
                  params={"filename": filename, "subfolder": subfolder, "type": type}
              )
              response.raise_for_status()
              return response.content

          def wait_for_completion(self, prompt_id: str, timeout: float = 300) -> dict:
              """Poll until workflow completes"""
              import time
              start = time.time()
              while time.time() - start < timeout:
                  history = self.get_history(prompt_id)
                  if prompt_id in history:
                      return history[prompt_id]
                  time.sleep(1)
              raise TimeoutError(f"Workflow {prompt_id} did not complete in {timeout}s")
    '';
  };

  workflowPy = writeTextFile {
    name = "workflow.py";
    text = ''
      """Workflow loading and modification utilities"""
      import json
      from pathlib import Path


      def load_workflow(path: Path) -> dict:
          """Load workflow JSON file"""
          return json.loads(path.read_text())


      def find_node_by_class(workflow: dict, class_type: str) -> tuple[str, dict] | None:
          """Find first node of given class type"""
          for node_id, node in workflow.items():
              if node.get("class_type") == class_type:
                  return node_id, node
          return None


      def find_all_nodes_by_class(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
          """Find all nodes of given class type"""
          return [
              (node_id, node)
              for node_id, node in workflow.items()
              if node.get("class_type") == class_type
          ]


      def set_prompt(workflow: dict, positive: str, negative: str = "") -> dict:
          """Set positive/negative prompts in workflow"""
          for node_id, node in workflow.items():
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
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  if "seed" in node.get("inputs", {}):
                      node["inputs"]["seed"] = seed
                  elif "noise_seed" in node.get("inputs", {}):
                      node["inputs"]["noise_seed"] = seed
          return workflow


      def set_steps(workflow: dict, steps: int) -> dict:
          """Set sampling steps in KSampler nodes"""
          for node_id, node in workflow.items():
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                  if "steps" in node.get("inputs", {}):
                      node["inputs"]["steps"] = steps
          return workflow


      def set_cfg(workflow: dict, cfg: float) -> dict:
          """Set CFG scale in KSampler nodes"""
          for node_id, node in workflow.items():
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "cfg" in node.get("inputs", {}):
                      node["inputs"]["cfg"] = cfg
          return workflow


      def set_dimensions(workflow: dict, width: int, height: int) -> dict:
          """Set image dimensions in EmptyLatentImage nodes"""
          for node_id, node in workflow.items():
              if node.get("class_type") == "EmptyLatentImage":
                  node["inputs"]["width"] = width
                  node["inputs"]["height"] = height
          return workflow


      def set_denoise(workflow: dict, denoise: float) -> dict:
          """Set denoise strength in KSampler nodes (for img2img)"""
          for node_id, node in workflow.items():
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "denoise" in node.get("inputs", {}):
                      node["inputs"]["denoise"] = denoise
          return workflow


      def set_sampler(workflow: dict, sampler_name: str) -> dict:
          """Set sampler in KSampler nodes"""
          for node_id, node in workflow.items():
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "sampler_name" in node.get("inputs", {}):
                      node["inputs"]["sampler_name"] = sampler_name
          return workflow


      def set_scheduler(workflow: dict, scheduler: str) -> dict:
          """Set scheduler in KSampler nodes"""
          for node_id, node in workflow.items():
              class_type = node.get("class_type", "")
              if class_type in ("KSampler", "KSamplerAdvanced"):
                  if "scheduler" in node.get("inputs", {}):
                      node["inputs"]["scheduler"] = scheduler
          return workflow


      def set_input_image(workflow: dict, image_path: str) -> dict:
          """Set input image path in LoadImage nodes"""
          for node_id, node in workflow.items():
              if node.get("class_type") == "LoadImage":
                  node["inputs"]["image"] = image_path
          return workflow
    '';
  };

  cliPy = writeTextFile {
    name = "cli.py";
    text = ''
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
                  progress.add_task("Waiting for completion...", total=None)
                  result = client.wait_for_completion(prompt_id)

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
    '';
  };

  # Bundle Python source into a directory structure
  pythonSource = runCommand "comfyui-client-source" {} ''
    mkdir -p $out/share/comfyui-client/src/comfyui_client
    cp ${pyproject} $out/share/comfyui-client/pyproject.toml
    cp ${initPy} $out/share/comfyui-client/src/comfyui_client/__init__.py
    cp ${clientPy} $out/share/comfyui-client/src/comfyui_client/client.py
    cp ${workflowPy} $out/share/comfyui-client/src/comfyui_client/workflow.py
    cp ${cliPy} $out/share/comfyui-client/src/comfyui_client/cli.py
  '';

  # Setup script to install Python package into venv
  setupScript = writeShellApplication {
    name = "comfyui-client-setup";
    text = ''
      VENV="''${1:-$FLOX_ENV_CACHE/venv}"
      SOURCE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")/share/comfyui-client"

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

in
symlinkJoin {
  name = "comfyui-scripts";
  paths = allScripts ++ [ pythonSource setupScript ];
  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows (sd15, sdxl, sd35, flux)";
  };
}
