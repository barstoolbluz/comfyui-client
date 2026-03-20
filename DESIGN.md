# comfyui-client: Multi-Mode Architecture Design

## Overview

Refactor `comfyui-client` from a CLI-only tool into a multi-mode client that supports three operational patterns, while incorporating production-grade features inspired by [SaladTechnologies/comfyui-api](https://github.com/SaladTechnologies/comfyui-api).

### Current State (v0.6.9)

- 8 CLI commands (`comfyui-submit`, `comfyui-batch`, `comfyui-queue`, `comfyui-status`, `comfyui-cancel`, `comfyui-result`, `comfyui-models`, `comfyui-info`)
- 16 convenience wrappers (`sd15-txt2img`, `flux-txt2img`, etc.)
- Python package: `httpx` + `websockets` + `typer` + `rich`
- Connects to remote ComfyUI via HTTP/WebSocket
- Batch support via JSON arrays on the command line

### Build Architecture (important context)

The source of truth for the Python package is the **Nix expression** at `.flox/pkgs/comfyui-scripts.nix` (2156 lines). This generates the full Python package at build time using `writeTextFile`. The `src/` directory contains a **subset** of the code (3 of 8 CLI commands, fewer client methods) and is out of sync with the Nix-generated version.

Key divergences between `src/` and the Nix expression:

| Component | `src/` | Nix-generated |
|-----------|--------|---------------|
| CLI commands | 3 (submit, queue, result) | 8 (+ batch, cancel, status, models, info) |
| `client.py` methods | 5 (submit, get_history, get_queue, get_image, wait_for_completion) | 10 (+ interrupt, delete_from_queue, clear_queue, get_system_stats, get_model_types, get_models) |
| `workflow.py` | No `clean_workflow`, no `get_seed` | Has both |
| `metadata.py` | Does not exist | Full implementation (stdlib-only PNG chunk parsing) |
| `pyproject.toml` | 3 entry points | 8 entry points |

**Decision required before implementation:** Either (a) consolidate the Nix-generated code back into `src/` so the Nix expression just packages it rather than generating it, or (b) add new modules to the Nix expression. Option (a) is strongly recommended — it makes development, testing, and code review tractable.

### Target State

Three concurrent modes of operation, selectable independently:

| Mode | Trigger | Description |
|------|---------|-------------|
| **CLI** | Direct invocation | Existing behavior, unchanged |
| **Watch** | `comfyui-watch` or Flox service | Monitor a folder for dropped JSON job files |
| **API** | `comfyui-serve` or Flox service | HTTP endpoint accepting workflow submissions |

All three modes share the same core client library (`comfyui_client/client.py`, `workflow.py`). The Nix-generated versions of these files (which have the full API surface) are the baseline.

---

## Mode A: CLI (Existing)

No changes to the existing CLI interface. All current commands remain as-is.

---

## Mode B: Watch Folder

### Concept

A long-running process that monitors a directory for JSON job files. When a file appears, it's parsed, submitted to ComfyUI, and results are saved to an output directory. The job file is then moved to a "completed" or "failed" subdirectory.

### Directory Layout

```
$COMFYUI_WATCH_DIR/                  # default: $FLOX_ENV_CACHE/watch
  incoming/                          # Drop JSON jobs here
  processing/                        # Jobs currently being executed
  completed/                         # Successfully processed jobs (moved here)
  failed/                            # Failed jobs with error metadata
  output/                            # Generated images
```

### Configuration

| Variable | Default | Source |
|----------|---------|--------|
| `COMFYUI_WATCH_DIR` | `$FLOX_ENV_CACHE/watch` | env var or `[vars]` in manifest |
| `COMFYUI_WATCH_POLL` | `2` | Polling interval in seconds (fallback when inotify unavailable) |
| `COMFYUI_WATCH_WORKFLOW` | _(none)_ | Default base workflow for simple prompt files |

### Job File Formats

**Full workflow** (existing ComfyUI API format):
```json
{
  "workflow": "path/to/base.json",
  "prompt": "a mountain landscape",
  "negative": "blurry",
  "seed": 42,
  "steps": 25,
  "output_prefix": "mountain"
}
```

**Batch** (array of jobs, same as `comfyui-batch` format):
```json
[
  {"prompt": "a cat", "seed": 100},
  {"prompt": "a dog", "seed": 200}
]
```

**Minimal** (just a prompt string, requires `COMFYUI_WATCH_WORKFLOW`):
```json
{"prompt": "a sunset over the ocean"}
```

### Processing Flow

```
1. File lands in incoming/
2. Watcher detects it (inotify or polling)
3. File moved to processing/ (atomic rename)
4. Parse JSON:
   a. If array → batch mode (iterate)
   b. If object with "workflow" key → load base workflow, apply overrides
   c. If object without "workflow" → use $COMFYUI_WATCH_WORKFLOW
5. Submit to ComfyUI via existing client.submit()
6. Wait for completion via WebSocket
7. Download images to output/
8. On success: move job file to completed/, append result metadata
9. On failure: move job file to failed/, append error details
```

### Implementation

New module: `src/comfyui_client/watcher.py`

```python
# Core watcher - uses watchdog library (cross-platform inotify/kqueue/polling)
class FolderWatcher:
    def __init__(self, watch_dir, client, default_workflow=None): ...
    def start(self): ...   # blocking main loop
    def stop(self): ...    # graceful shutdown
```

New CLI entry point: `comfyui-watch`

```
comfyui-watch [--dir PATH] [--workflow PATH] [--poll SECONDS]
```

### Flox Service Integration

```toml
[services.comfyui-watch]
command = "comfyui-watch"
is-daemon = false
```

Flox manages the process lifecycle (SIGTERM on shutdown). No custom shutdown command needed.

---

## Mode C: Local API Endpoint

### Concept

A lightweight HTTP server that accepts workflow submissions and returns results, inspired by the [SaladTechnologies/comfyui-api](https://github.com/SaladTechnologies/comfyui-api) pattern. This is **not** a full reimplementation of that project -- it's a focused adapter that exposes the comfyui-client functionality over HTTP.

### Why Not Use Salad's Wrapper Directly?

Salad's wrapper is a Node.js/TypeScript Fastify application designed to **manage** a ComfyUI process as a child. Our use case is different: we're a **remote client** that connects to an already-running ComfyUI instance (typically in Kubernetes). We don't need child process management, apt/pip installation, or ComfyUI lifecycle control. We need the API surface.

### Technology Choice

**FastAPI + Pydantic** -- the natural Python equivalent of Salad's Fastify + Zod:
- Auto-generated OpenAPI/Swagger docs at `/docs`
- Pydantic models for request/response validation
- Async support (pairs well with existing `websockets` usage)
- Lightweight, no heavy framework overhead

**Async caveat:** The existing `client.wait_for_completion()` calls `asyncio.run()`, which cannot be used inside FastAPI's already-running event loop. The server must call `client._ws_wait()` (the async version) directly via `await`. This means the server module needs the `ComfyUIClient` to expose `_ws_wait` as a public async method (e.g., rename to `async_wait_for_completion`).

### API Endpoints

#### Core Endpoints (modeled after Salad's contract)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/prompt` | Submit a full ComfyUI workflow |
| `POST` | `/workflow/{name}` | Submit to a named workflow template |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks ComfyUI connectivity) |
| `GET` | `/queue` | Current queue status |
| `GET` | `/models` | List model types |
| `GET` | `/models/{type}` | List models in a category |
| `POST` | `/cancel` | Cancel a job by ID |
| `POST` | `/cancel/all` | Interrupt + clear queue |
| `GET` | `/status` | Server and GPU diagnostics |

#### `POST /prompt` -- Full Workflow Submission

**Request:**
```json
{
  "prompt": { "1": {"class_type": "...", "inputs": {...}}, ... },
  "id": "optional-caller-id",
  "webhook_url": "https://example.com/callback",
  "convert_output": {"format": "webp", "quality": 85}
}
```

**Synchronous response (200)** -- when no `webhook_url`:
```json
{
  "id": "uuid",
  "images": [
    {"filename": "output_00001_.png", "data": "<base64>", "content_type": "image/png"}
  ],
  "stats": {
    "comfy_execution_ms": 4200,
    "total_ms": 4800
  }
}
```

**Async response (202)** -- when `webhook_url` provided:
```json
{
  "id": "uuid",
  "status": "queued"
}
```

#### `POST /workflow/{name}` -- Named Workflow Templates

This is the equivalent of Salad's auto-mounted `/workflow/*` routes. Instead of runtime TypeScript transpilation, we use Python modules.

**Request** (simplified -- the schema is defined by the workflow):
```json
{
  "prompt": "a cat sitting on a windowsill",
  "negative": "blurry, low quality",
  "seed": 42,
  "steps": 25,
  "checkpoint": "v1-5-pruned-emaonly.safetensors"
}
```

**Workflow definition** (Python module in `workflows/` directory):
```python
# workflows/sd15/txt2img.py
from pydantic import BaseModel, Field

class Request(BaseModel):
    prompt: str
    negative: str = ""
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32))
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=7.0, ge=0, le=20)
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    checkpoint: str = "v1-5-pruned-emaonly.safetensors"

description = "Standard SD 1.5 text-to-image generation"

def generate(params: Request) -> dict:
    """Returns a full ComfyUI API-format workflow dict."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": params.checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["1", 1]}},
        # ... full node graph
    }
```

**Auto-discovery:** At startup, the server walks `$COMFYUI_WORKFLOW_DIR` (or `workflows/api/` in the package), imports each `.py` file, and registers it as a `POST /workflow/{path}` route with the module's Pydantic schema for validation and auto-docs.

### Configuration

| Variable | Default | Source |
|----------|---------|--------|
| `COMFYUI_SERVE_HOST` | `0.0.0.0` | env var or `[vars]` |
| `COMFYUI_SERVE_PORT` | `3000` | env var or `[vars]` |
| `COMFYUI_WORKFLOW_DIR` | _(bundled)_ | Directory of workflow `.py` templates |
| `COMFYUI_WEBHOOK_SECRET` | _(none)_ | HMAC secret for signing webhook payloads |

### Implementation

New module: `src/comfyui_client/server.py`

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="comfyui-client", description="Local API for remote ComfyUI")

# Routes defined here, delegating to existing client.py
```

New CLI entry point: `comfyui-serve`

```
comfyui-serve [--host HOST] [--port PORT] [--workflow-dir PATH]
```

### Flox Service Integration

```toml
[services.comfyui-api]
command = "comfyui-serve"
```

---

## Mode D: Features Adopted from SaladTechnologies/comfyui-api

### Adopted (high value, fits our architecture)

| Feature | Rationale | Implementation |
|---------|-----------|----------------|
| **Health/ready probes** | Essential for K8s. `/health` = always 200; `/ready` = checks ComfyUI reachability | `server.py` -- `GET /health`, `GET /ready` with `client.get_system_stats()` check |
| **Webhook delivery** | Enables async fire-and-forget from external callers | `webhooks.py` -- HMAC-signed POST (Svix-compatible format), retry with backoff |
| **Sync/async response modes** | No webhook = sync (wait + return base64); webhook = async (202 + callback) | Already supported by `client.wait_for_completion()`; add base64 encoding of results |
| **Image format conversion** | Callers can request JPEG/WebP instead of PNG | `Pillow` (already available in ecosystem) -- convert before base64 encoding |
| **Named workflow endpoints** | Simplified API for common operations without sending full node graphs | Auto-mount Python modules from `workflows/` directory with Pydantic schemas |
| **OpenAPI/Swagger docs** | Self-documenting API at `/docs` | Free with FastAPI |
| **Execution statistics** | Timing data in responses (`comfy_execution_ms`, `total_ms`) | Wrap `client.wait_for_completion()` with timing instrumentation |

### Adapted (modified to fit our model)

| Feature | Salad's Version | Our Version |
|---------|----------------|-------------|
| **Credential resolver** | Per-request glob-pattern credentials for model downloads | Not needed -- we don't download models (remote ComfyUI manages its own). Reserve for future. |
| **Node preprocessing** | Download URLs in LoadImage/model-loader nodes → local files | Not applicable -- we submit to a remote server, not localhost. ComfyUI handles its own file access. |
| **Model manifest** | YAML for apt/pip/node/model installation at startup | Flox manifest handles our dependencies. Not needed. |
| **Warmup workflow** | Run a workflow at startup to preload models into VRAM | Useful for the API server -- optional `COMFYUI_WARMUP_WORKFLOW` that runs on `comfyui-serve` startup |

### Not Adopted (doesn't fit our architecture)

| Feature | Reason |
|---------|--------|
| Child process management | We connect to a **remote** ComfyUI, not a local child process |
| apt/pip package installation | Handled by Flox environment |
| Custom node git cloning | Server-side concern, not client-side |
| S3/Azure/HF upload providers | Out of scope -- if needed later, add as separate module |
| LRU file cache | No local model storage needed |
| SaladCloud-specific metadata | Vendor-specific |

---

## New Dependencies

| Package | Purpose | Mode |
|---------|---------|------|
| `fastapi` | HTTP API framework (includes Pydantic) | API |
| `uvicorn` | ASGI server | API |
| `watchdog` | Cross-platform filesystem events | Watch |
| `Pillow` | Image format conversion | API |

These are added to `pyproject.toml` dependencies and installed into the venv. Note: `pydantic` is a transitive dependency of `fastapi` and does not need to be listed separately.

---

## Updated Package Structure

**Prerequisite:** Consolidate the Nix-generated code back into `src/` so all Python modules live in the source tree. The Nix expression should package the source, not generate it.

```
src/comfyui_client/
  __init__.py              # (existing, sync from Nix) Public exports
  client.py                # (existing, sync from Nix) Full HTTP/WebSocket client (10 methods)
  cli.py                   # (existing, sync from Nix) All 8 CLI commands
  workflow.py              # (existing, sync from Nix) Workflow JSON manipulation
  metadata.py              # (existing, sync from Nix) PNG metadata extraction
  watcher.py               # (new) Watch folder mode
  server.py                # (new) FastAPI application
  webhooks.py              # (new) Webhook delivery with HMAC signing
  conversion.py            # (new) Image format conversion (Pillow)

workflows/api/             # (existing) JSON workflow templates -- used by CLI wrappers + watch mode
  sd15/sd15-txt2img.json
  sd15/sd15-img2img.json
  ...                      # 16 files total (4 models x 4 operations)

workflows/templates/       # (new) Python workflow modules -- used by API /workflow/{name} routes
  __init__.py              #   Auto-discovery loader
  sd15/
    txt2img.py             #   Pydantic schema + generate() -> ComfyUI node graph
    img2img.py
  sdxl/
    txt2img.py
    img2img.py
  flux/
    txt2img.py
    img2img.py
```

The existing JSON workflows under `workflows/api/` remain the source of truth for CLI wrappers and watch folder mode. The new Python modules under `workflows/templates/` are used by the API server for validated, schema-documented endpoints. The Python modules can load and parameterize the JSON templates internally via `workflow.load_workflow()` + `set_*()` functions.

---

## Updated CLI Entry Points

| Command | Module | Description |
|---------|--------|-------------|
| `comfyui-submit` | `cli.py` | _(existing)_ Submit workflow |
| `comfyui-batch` | `cli.py` | _(existing)_ Batch processing |
| `comfyui-queue` | `cli.py` | _(existing)_ Queue status |
| `comfyui-status` | `cli.py` | _(existing)_ Server diagnostics |
| `comfyui-cancel` | `cli.py` | _(existing)_ Cancel jobs |
| `comfyui-result` | `cli.py` | _(existing)_ Retrieve results |
| `comfyui-models` | `cli.py` | _(existing)_ List models |
| `comfyui-info` | `cli.py` | _(existing)_ PNG metadata |
| `comfyui-watch` | `watcher.py` | **(new)** Watch folder daemon |
| `comfyui-serve` | `server.py` | **(new)** API server |

---

## Flox Runtime Environment Changes

### manifest.toml additions

**Note:** Flox `[vars]` values are static strings -- they cannot reference other env vars like `$FLOX_ENV_CACHE`. Dynamic defaults must be set in `[hook]`.

```toml
[vars]
# COMFYUI_HOST and COMFYUI_PORT are already set in the hook with
# ${VAR:-default} pattern to respect pre-existing env vars.

# Watch mode -- set a static default or leave empty and let the hook default it
# COMFYUI_WATCH_DIR = "/home/user/comfyui-jobs"
# COMFYUI_WATCH_WORKFLOW = ""           # optional default workflow

# API mode
COMFYUI_SERVE_HOST = "0.0.0.0"
COMFYUI_SERVE_PORT = "3000"
# COMFYUI_WEBHOOK_SECRET = ""           # optional HMAC secret

[services.comfyui-watch]
command = "comfyui-watch"
is-daemon = false

[services.comfyui-api]
command = "comfyui-serve"
is-daemon = false
```

The `[hook]` on-activate should set the dynamic default for `COMFYUI_WATCH_DIR`:

```bash
export COMFYUI_WATCH_DIR="${COMFYUI_WATCH_DIR:-$FLOX_ENV_CACHE/watch}"
```

Users can enable either or both services with `flox activate -s`, or run them manually. CLI mode is always available regardless of services.

---

## Migration Path

### Phase 0: Source Consolidation (prerequisite)
- Extract all Nix-generated Python code from `comfyui-scripts.nix` into `src/`
- Sync `client.py` (add the 5 missing methods), `cli.py` (add 5 missing commands), `workflow.py` (add `clean_workflow`, `get_seed`), and create `metadata.py`
- Update `pyproject.toml` with all 8 entry points
- Refactor `comfyui-scripts.nix` to package `src/` instead of generating code inline
- Verify `flox build comfyui-scripts` still produces identical output
- Make `_ws_wait` a public async method (`async_wait_for_completion`) on `ComfyUIClient`

### Phase 1: Watch Folder
- Add `watcher.py` with `watchdog`-based monitoring
- Add `comfyui-watch` entry point to `pyproject.toml`
- Add `COMFYUI_WATCH_DIR` hook default in runtime environment
- No changes to existing commands

### Phase 2: API Server
- Add `server.py` with FastAPI app
- Add `webhooks.py` for async callback delivery
- Add `conversion.py` for image format conversion
- Add `comfyui-serve` entry point to `pyproject.toml`
- No changes to existing commands

### Phase 3: Workflow Templates
- Create Python workflow template modules for the 16 existing model/operation combinations
- Each module loads the corresponding JSON template from `workflows/api/` and parameterizes it
- Auto-discovery in `server.py` registers as `POST /workflow/{model}/{operation}` routes
- OpenAPI docs auto-generated from Pydantic schemas

---

## Reference

- [SaladTechnologies/comfyui-api](https://github.com/SaladTechnologies/comfyui-api) -- Production ComfyUI API wrapper (Node.js/TypeScript, Fastify + Zod, MIT license). Key patterns adopted: health probes, sync/async response modes, webhook delivery, named workflow endpoints, execution stats.
- [ComfyUI API docs](https://docs.comfy.org/development/comfyui-server/api-key-integration) -- Upstream ComfyUI server API reference.
