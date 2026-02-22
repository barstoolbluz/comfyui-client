{ lib, writeShellApplication, writeTextFile, symlinkJoin, runCommand }:

let
  # Model types for workflow scripts
  models = [ "sd15" "sdxl" "sd35" "flux" ];

  operations = [ "txt2img" "img2img" "upscale" ];

  # Generate a single workflow wrapper script
  makeScript = model: op:
    let
      name = "${model}-${op}";
    in
    writeShellApplication {
      inherit name;
      text = ''
        WORKFLOWS_BASE="''${COMFYUI_WORKFLOWS:-$HOME/comfyui-work/user/default/workflows}"
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
          clean_workflow,
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
          "clean_workflow",
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
      .SH SEE ALSO
      .BR comfyui-queue (1),
      .BR comfyui-result (1),
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
      .TP
      .BR comfyui-queue (1)
      Display the current queue status.
      .TP
      .BR comfyui-result (1)
      Retrieve output images from a completed workflow.
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
      .SH WORKFLOW FILES
      Wrapper scripts look for workflow files in a standard location:
      .PP
      .RS
      .nf
      $COMFYUI_WORKFLOWS/api/<model>/<model>-<operation>.json
      .fi
      .RE
      .PP
      For example,
      .B sdxl-txt2img
      loads:
      .PP
      .RS
      .nf
      $COMFYUI_WORKFLOWS/api/sdxl/sdxl-txt2img.json
      .fi
      .RE
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
      Base directory containing workflow files. Default: $HOME/comfyui-work/user/default/workflows
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
      .BR comfyui-queue (1),
      .BR comfyui-result (1)
    '';
  };

  # Bundle man pages
  manPages = runCommand "comfyui-client-man" {} ''
    mkdir -p $out/share/man/man1 $out/share/man/man7
    cp ${manSubmit} $out/share/man/man1/comfyui-submit.1
    cp ${manQueue} $out/share/man/man1/comfyui-queue.1
    cp ${manResult} $out/share/man/man1/comfyui-result.1
    cp ${manOverview} $out/share/man/man7/comfyui-client.7
  '';

  # Setup script to install Python package into venv
  setupScript = writeShellApplication {
    name = "comfyui-client-setup";
    text = ''
      VENV="''${1:-$FLOX_ENV_CACHE/venv}"
      SOURCE_DIR="''${FLOX_ENV}/share/comfyui-client"

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
  paths = allScripts ++ [ pythonSource setupScript manPages ];
  meta = {
    description = "CLI wrapper scripts for ComfyUI workflows (sd15, sdxl, sd35, flux)";
  };
}
