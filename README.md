# ComfyUI Client

A complete toolkit for interacting with ComfyUI servers — CLI tools, an HTTP API server, and a watch-folder daemon — packaged as a reproducible [Flox](https://flox.dev) environment.

## Quick Start

```bash
git clone https://github.com/barstoolbluz/comfyui-client
cd comfyui-client
flox activate

# Generate an image (ComfyUI must be running on localhost:8188)
sd15-txt2img -p "a beautiful sunset over mountains" -w

# Or start the API server and watch folder as services
flox services start
```

## Prerequisites

- [Flox](https://flox.dev) installed
- A running ComfyUI server (default: `localhost:8188`)

## Three Ways to Generate Images

### 1. CLI Commands

Sixteen convenience scripts cover every model and operation:

```
sd15-txt2img    sdxl-txt2img    sd35-txt2img    flux-txt2img
sd15-img2img    sdxl-img2img    sd35-img2img    flux-img2img
sd15-upscale    sdxl-upscale    sd35-upscale    flux-upscale
sd15-inpaint    sdxl-inpaint    sd35-inpaint    flux-inpaint
```

Each is a thin wrapper around `comfyui-submit` with the correct workflow pre-selected:

```bash
# Text-to-image
flux-txt2img -p "cyberpunk cityscape, neon rain" --steps 25 -w

# Image-to-image
sdxl-img2img -p "oil painting style" -i photo.png --denoise 0.6 -w

# Upscale
sd15-upscale -p "sharp details" -i small.png -w

# Inpaint
sd35-inpaint -p "a red door" -i house.png -w
```

The CLI supports prompt, seed, steps, CFG, dimensions, denoise, sampler, scheduler, and input image. For advanced parameters like `upscale_by` or separate mask images, use the [API server](#2-http-api-server) template endpoints which expose the full parameter set for each operation.

### 2. HTTP API Server

A FastAPI server with typed endpoints for every model and operation, plus workflow submission with optional webhook callbacks.

```bash
# Start the server
comfyui-serve                         # default: 0.0.0.0:3000
comfyui-serve --port 8080             # custom port

# Or as a Flox service
flox services start comfyui-api
```

**Template endpoints** accept validated JSON and return a ready-to-submit workflow:

```bash
curl -X POST http://localhost:3000/workflow/flux/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat astronaut", "steps": 20, "seed": 42}'
```

**Prompt endpoint** submits a workflow to ComfyUI and returns generated images (base64-encoded):

```bash
# Synchronous — blocks until images are ready
curl -X POST http://localhost:3000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": <workflow-dict>}'

# Asynchronous — returns immediately, delivers results to webhook
curl -X POST http://localhost:3000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": <workflow-dict>, "webhook_url": "https://example.com/hook"}'
```

Pipe them together for end-to-end generation:

```bash
curl -s -X POST localhost:3000/prompt \
  -H "Content-Type: application/json" \
  -d "$(curl -s -X POST localhost:3000/workflow/sd15/txt2img \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "a lighthouse at sunset", "seed": 42, "steps": 20}')"
```

### 3. Watch Folder

Drop JSON job files into a folder and the watcher submits them to ComfyUI automatically.

```bash
# Start the watcher
comfyui-watch -d ~/jobs -w workflows/api/sd15/sd15-txt2img.json

# Or as a Flox service (uses $FLOX_ENV_CACHE/watch by default)
flox services start comfyui-watch
```

The watcher creates this directory structure:

```
watch/
  incoming/     # Drop job files here
  processing/   # Currently running
  completed/    # Finished (with _result metadata)
  failed/       # Errors (with _error metadata)
  output/       # Generated images
```

Job files support three formats:

```bash
# Minimal — uses the default workflow (-w flag)
echo '{"prompt": "a sunset", "seed": 42, "steps": 8}' > incoming/job.json

# Full — specifies its own workflow
echo '{"workflow": "path/to/workflow.json", "prompt": "a sunset"}' > incoming/job.json

# Batch — array of jobs processed sequentially
echo '[{"prompt": "cats", "seed": 1}, {"prompt": "dogs", "seed": 2}]' > incoming/batch.json
```

Completed jobs are annotated with result metadata:

```json
{
  "prompt": "a sunset",
  "_result": {
    "prompt_id": "abc-123",
    "images": ["output_00001_.png"],
    "completed_at": "2026-03-19T20:03:08+00:00"
  }
}
```

## API Server Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/ready` | ComfyUI connectivity check |
| `GET` | `/queue` | Running and pending job counts |
| `GET` | `/status` | System stats, devices, and queue info |
| `GET` | `/models` | List model type categories |
| `GET` | `/models/{type}` | List models in a category |
| `POST` | `/prompt` | Submit workflow, return images |
| `POST` | `/cancel` | Cancel a specific job |
| `POST` | `/cancel/all` | Interrupt running + clear queue |
| `POST` | `/workflow/{model}/{op}` | Generate workflow from template |

### Template Routes

Sixteen `POST /workflow/{model}/{operation}` endpoints are auto-discovered from Python modules in `workflows/templates/`. Each returns `{"prompt": <workflow>}` — the same shape that `POST /prompt` accepts.

| | txt2img | img2img | upscale | inpaint |
|------|---------|---------|---------|---------|
| **sd15** | /workflow/sd15/txt2img | /workflow/sd15/img2img | /workflow/sd15/upscale | /workflow/sd15/inpaint |
| **sdxl** | /workflow/sdxl/txt2img | /workflow/sdxl/img2img | /workflow/sdxl/upscale | /workflow/sdxl/inpaint |
| **sd35** | /workflow/sd35/txt2img | /workflow/sd35/img2img | /workflow/sd35/upscale | /workflow/sd35/inpaint |
| **flux** | /workflow/flux/txt2img | /workflow/flux/img2img | /workflow/flux/upscale | /workflow/flux/inpaint |

All template endpoints are fully documented in the OpenAPI schema at `/docs`.

### Webhook Delivery

When `webhook_url` is provided to `POST /prompt`, the server returns `202 Accepted` immediately and delivers results asynchronously. Webhooks follow the [Standard Webhooks](https://www.standardwebhooks.com/) spec with HMAC-SHA256 signing.

Set `COMFYUI_WEBHOOK_SECRET` to enable signature verification.

Webhook payload:

```json
{
  "id": "request-id",
  "status": "completed",
  "images": [{"filename": "out.png", "data": "<base64>", "content_type": "image/png"}],
  "stats": {"total_ms": 5432, "prompt_id": "uuid"}
}
```

### Image Conversion

The `POST /prompt` endpoint supports on-the-fly image format conversion:

```json
{
  "prompt": { ... },
  "convert_output": {"format": "webp", "quality": 90}
}
```

Supported formats: `png` (default), `jpeg`, `webp`.

## CLI Reference

### Core Tools

| Command | Description |
|---------|-------------|
| `comfyui-submit` | Submit a workflow JSON with parameter overrides |
| `comfyui-batch` | Process multiple jobs from a JSON batch file |
| `comfyui-queue` | Show running and pending job counts |
| `comfyui-cancel` | Cancel jobs (specific, current, or all) |
| `comfyui-result` | Download images for a completed prompt ID |
| `comfyui-status` | Show server info, devices, and queue status |
| `comfyui-models` | List available models by category |
| `comfyui-info` | Extract generation metadata from ComfyUI PNG files |
| `comfyui-watch` | Watch a folder for job files |
| `comfyui-serve` | Start the HTTP API server |

### comfyui-submit

The core submission tool. All 16 convenience scripts (`sd15-txt2img`, `flux-upscale`, etc.) are wrappers around this.

```bash
comfyui-submit <workflow.json> [OPTIONS]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-p, --prompt` | Positive prompt text |
| `-n, --negative` | Negative prompt text |
| `-s, --seed` | Random seed (default: random) |
| `--steps` | Sampling steps |
| `--cfg` | CFG guidance scale |
| `-W, --width` | Image width in pixels |
| `-H, --height` | Image height in pixels |
| `-d, --denoise` | Denoise strength 0.0-1.0 (img2img/inpaint/upscale) |
| `--sampler` | Sampler algorithm (euler, dpmpp_2m, etc.) |
| `--scheduler` | Scheduler (normal, karras, simple, etc.) |
| `-i, --image` | Input image path |
| `-w, --wait` | Wait for completion and download results |
| `-o, --output` | Output directory for images |
| `-c, --count` | Generate multiple images with varied seeds |
| `--parallel` | Submit all batch jobs simultaneously |
| `--prefix` | Filename prefix for outputs |

**Batch generation:**

```bash
# Sequential (one at a time)
sdxl-txt2img -p "landscape" -c 5 -w -o ./output

# Parallel (all at once)
sdxl-txt2img -p "landscape" -c 5 --parallel -w -o ./output
```

### comfyui-info

Extracts generation metadata embedded in ComfyUI PNG files:

```bash
comfyui-info output.png           # Human-readable summary
comfyui-info output.png --json    # Raw JSON metadata
```

Shows: model, prompt, negative prompt, seed, steps, CFG, sampler, scheduler, dimensions.

### comfyui-watch

```bash
comfyui-watch [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `-d, --dir` | Watch directory (default: `$COMFYUI_WATCH_DIR`) |
| `-w, --workflow` | Default workflow for jobs without a `workflow` key |
| `-p, --poll` | Poll interval in seconds (default: 2.0) |

### comfyui-serve

```bash
comfyui-serve [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--host` | Bind address (default: `$COMFYUI_SERVE_HOST` or 0.0.0.0) |
| `--port` | Bind port (default: `$COMFYUI_SERVE_PORT` or 3000) |
| `--log-level` | Log level (default: info) |

## Model Defaults

Each model family has tuned defaults:

| Model | Default Resolution | CFG | Steps | Notes |
|-------|-------------------|-----|-------|-------|
| SD 1.5 | 512 x 512 | 7.0 | 20 | Fastest, good for prototyping |
| SDXL | 1024 x 1024 | 7.0 | 25 | High quality, good all-rounder |
| SD 3.5 | 1024 x 1024 | 4.5 | 28 | Triple CLIP encoder (clip_l, clip_g, t5xxl) |
| FLUX | 1024 x 1024 | 1.0 | 20 | Guidance-free, highest quality |

## Bundled Workflows

Sixteen API-format workflow JSONs under `workflows/api/`:

```
workflows/api/
  sd15/   sd15-txt2img.json  sd15-img2img.json  sd15-upscale.json  sd15-inpaint.json
  sdxl/   sdxl-txt2img.json  sdxl-img2img.json  sdxl-upscale.json  sdxl-inpaint.json
  sd35/   sd35-txt2img.json  sd35-img2img.json  sd35-upscale.json  sd35-inpaint.json
  flux/   flux-txt2img.json  flux-img2img.json  flux-upscale.json  flux-inpaint.json
```

These are standard ComfyUI API-format workflows. The CLI modifies node inputs directly — no template variables or placeholders. You can export your own workflows from ComfyUI (Settings > Save API Format) and use them with `comfyui-submit`.

## Flox Services

The manifest defines two services:

```toml
[services.comfyui-watch]
command = "comfyui-watch"

[services.comfyui-api]
command = "comfyui-serve"
```

```bash
flox activate --start-services        # Start both on activation
flox services start                   # Start all services
flox services start comfyui-api       # Start just the API server
flox services stop                    # Stop all services
```

## Environment Variables

### ComfyUI Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_HOST` | `localhost` | ComfyUI server hostname |
| `COMFYUI_PORT` | `8188` | ComfyUI server port |

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_SERVE_HOST` | `0.0.0.0` | Bind address |
| `COMFYUI_SERVE_PORT` | `3000` | Bind port |
| `COMFYUI_WEBHOOK_SECRET` | _(none)_ | HMAC-SHA256 secret for webhook signing |

### Watch Folder

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_WATCH_DIR` | `$FLOX_ENV_CACHE/watch` | Directory to monitor |
| `COMFYUI_WATCH_WORKFLOW` | _(none)_ | Default workflow for minimal job files |
| `COMFYUI_WATCH_POLL` | `2.0` | Poll interval in seconds |

### Workflow Discovery

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_WORKFLOW_DIR` | _(auto)_ | Custom template directory for API server |

The API server resolves templates from (in order): `COMFYUI_WORKFLOW_DIR`, `$FLOX_ENV/share/comfyui-client/workflows/templates`, or the bundled `workflows/templates/` directory.

## Man Pages

Every command has a full man page:

```bash
man comfyui-submit
man comfyui-serve
man comfyui-watch
man comfyui-client     # Overview (section 7)
```

## Project Structure

```
comfyui-client/
  .flox/
    env/manifest.toml                    # Flox environment (packages, services, vars)
    pkgs/comfyui-scripts.nix             # Nix derivation (scripts, man pages, completions)
  src/comfyui_client/
    __init__.py                          # Public API exports
    cli.py                               # CLI commands (Typer + Rich)
    client.py                            # ComfyUI WebSocket/HTTP client
    server.py                            # FastAPI server
    templates.py                         # Template discovery and route registration
    watcher.py                           # Watch folder daemon
    webhooks.py                          # Webhook delivery with Standard Webhooks signing
    workflow.py                          # Workflow JSON manipulation
    conversion.py                        # Image format conversion (PNG/JPEG/WebP)
    metadata.py                          # PNG metadata extraction (stdlib-only)
  workflows/
    api/{sd15,sdxl,sd35,flux}/           # 16 API-format workflow JSONs
    templates/{sd15,sdxl,sd35,flux}/     # 16 Python template modules (Pydantic schemas)
  pyproject.toml                         # Python package config (version 0.9.0)
```

## Development

```bash
# Edit source
vim src/comfyui_client/server.py

# Reinstall into venv
pip install -e .

# Build the Nix package
flox build comfyui-scripts

# Run the build output
./result-comfyui-scripts/bin/sd15-txt2img -p "test" -w
```

### Custom Templates

Add your own template by creating a Python module under `workflows/templates/{model}/{operation}.py`:

```python
"""My custom workflow template."""
import random
from pathlib import Path
from pydantic import BaseModel, Field
from comfyui_client.workflow import apply_params, load_workflow

_JSON = Path(__file__).resolve().parents[2] / "api" / "sd15" / "sd15-txt2img.json"

description = "My custom generation workflow"

class Request(BaseModel):
    prompt: str
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=20, ge=1, le=150)

def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    return apply_params(workflow, prompt=params.prompt, seed=params.seed, steps=params.steps)
```

The API server auto-discovers and registers it as `POST /workflow/{model}/{operation}`.

## Troubleshooting

**Cannot connect to ComfyUI:**

```bash
# Verify ComfyUI is reachable
curl http://localhost:8188/system_stats

# Check host/port settings
COMFYUI_HOST=192.168.1.100 flox activate
```

**Rebuild the virtual environment:**

```bash
rm -rf $FLOX_ENV_CACHE/venv
flox activate    # Recreates automatically
```

**Check available models:**

```bash
comfyui-models checkpoints
comfyui-models upscale_models
```

## Resources

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Flox](https://flox.dev/docs)
- [Issue Tracker](https://github.com/barstoolbluz/comfyui-client/issues)

## License

See `pyproject.toml` for Python package license. ComfyUI workflows are community-contributed. Flox environment: MIT.
