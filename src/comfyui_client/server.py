"""FastAPI server for ComfyUI workflow submission and management."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .client import ComfyUIClient
from .conversion import encode_image_for_payload
from .templates import discover_templates, register_template_routes
from .webhooks import WebhookPayload, deliver_webhook
from .workflow import clean_workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ConvertOutput(BaseModel):
    format: Literal["png", "jpeg", "webp"] = "png"
    quality: int = Field(default=85, ge=1, le=100)


class PromptRequest(BaseModel):
    prompt: dict
    id: str | None = None
    webhook_url: str | None = None
    convert_output: ConvertOutput | None = None


class ImageResult(BaseModel):
    filename: str
    data: str
    content_type: str


class PromptSyncResponse(BaseModel):
    id: str
    images: list[ImageResult]
    stats: dict


class PromptAsyncResponse(BaseModel):
    id: str
    status: str = "queued"


class CancelRequest(BaseModel):
    prompt_id: str


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    comfyui: Literal["connected", "unreachable"]


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Lifespan — singleton client + webhook secret
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    host = os.environ.get("COMFYUI_HOST", "localhost")
    port = int(os.environ.get("COMFYUI_PORT", "8188"))
    app.state.client = ComfyUIClient(host=host, port=port)
    app.state.webhook_secret = os.environ.get("COMFYUI_WEBHOOK_SECRET")
    logger.info("ComfyUI client targeting %s:%d", host, port)

    templates = discover_templates()
    register_template_routes(app, templates)
    logger.info("Loaded %d workflow templates", len(templates))

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="comfyui-client", lifespan=lifespan)


# -- Exception handlers ----------------------------------------------------

@app.exception_handler(httpx.ConnectError)
async def connect_error_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"error": "ComfyUI unreachable", "detail": str(exc)},
    )


@app.exception_handler(httpx.HTTPStatusError)
async def http_status_error_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"error": "ComfyUI error", "detail": str(exc)},
    )


# -- Health / readiness -----------------------------------------------------

@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/ready")
async def ready():
    client: ComfyUIClient = app.state.client
    try:
        client.get_queue()
        return ReadyResponse(status="ready", comfyui="connected")
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "comfyui": "unreachable"},
        )


# -- Queue / models / status ------------------------------------------------

@app.get("/queue")
async def queue():
    client: ComfyUIClient = app.state.client
    q = client.get_queue()
    return {
        "running": len(q.get("queue_running", [])),
        "pending": len(q.get("queue_pending", [])),
    }


@app.get("/models")
async def models():
    client: ComfyUIClient = app.state.client
    return client.get_model_types()


@app.get("/models/{model_type}")
async def models_by_type(model_type: str):
    client: ComfyUIClient = app.state.client
    return client.get_models(model_type)


@app.get("/status")
async def status():
    client: ComfyUIClient = app.state.client
    stats = client.get_system_stats()
    q = client.get_queue()
    stats["queue"] = {
        "running": len(q.get("queue_running", [])),
        "pending": len(q.get("queue_pending", [])),
    }
    return stats


# -- Prompt submission -------------------------------------------------------

def _collect_images(
    client: ComfyUIClient, result: dict, convert: ConvertOutput | None
) -> list[dict]:
    """Download output images and optionally convert them."""
    images = []
    fmt = convert.format if convert else "png"
    quality = convert.quality if convert else 85
    for node_output in result.get("outputs", {}).values():
        for img in node_output.get("images", []):
            data = client.get_image(
                img["filename"], img.get("subfolder", "")
            )
            encoded = encode_image_for_payload(
                data, img["filename"], target_format=fmt, quality=quality
            )
            images.append(encoded)
    return images


async def _process_and_webhook(
    client: ComfyUIClient,
    prompt_id: str,
    request_id: str,
    req: PromptRequest,
    start: float,
):
    """Background task: wait for completion, collect images, deliver webhook."""
    secret = app.state.webhook_secret
    try:
        result = await client.async_wait_for_completion(prompt_id)
        images = _collect_images(client, result, req.convert_output)
        elapsed = round((time.monotonic() - start) * 1000)
        payload = WebhookPayload(
            id=request_id,
            status="completed",
            images=images,
            stats={"total_ms": elapsed, "prompt_id": prompt_id},
        )
    except Exception as exc:
        logger.exception("Workflow %s failed", prompt_id)
        elapsed = round((time.monotonic() - start) * 1000)
        payload = WebhookPayload(
            id=request_id,
            status="failed",
            error=str(exc),
            stats={"total_ms": elapsed, "prompt_id": prompt_id},
        )
    await deliver_webhook(req.webhook_url, payload, secret=secret)


@app.post("/prompt")
async def submit_prompt(req: PromptRequest):
    client: ComfyUIClient = app.state.client
    workflow = clean_workflow(req.prompt)
    request_id = req.id or str(uuid.uuid4())
    start = time.monotonic()

    prompt_id = client.submit(workflow)

    if req.webhook_url:
        asyncio.create_task(
            _process_and_webhook(client, prompt_id, request_id, req, start)
        )
        return JSONResponse(
            status_code=202,
            content={"id": request_id, "status": "queued"},
        )

    # Synchronous: wait for completion
    result = await client.async_wait_for_completion(prompt_id)
    images = _collect_images(client, result, req.convert_output)
    elapsed = round((time.monotonic() - start) * 1000)
    return PromptSyncResponse(
        id=request_id,
        images=[ImageResult(**img) for img in images],
        stats={"total_ms": elapsed, "prompt_id": prompt_id},
    )


# -- Cancel ------------------------------------------------------------------

@app.post("/cancel")
async def cancel(req: CancelRequest):
    client: ComfyUIClient = app.state.client
    client.delete_from_queue([req.prompt_id])
    return {"status": "cancelled"}


@app.post("/cancel/all")
async def cancel_all():
    client: ComfyUIClient = app.state.client
    client.interrupt()
    client.clear_queue()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

serve_typer = typer.Typer()


@serve_typer.command()
def serve(
    host: str = typer.Option(None, "--host", help="Bind address"),
    port: int = typer.Option(None, "--port", help="Bind port"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
):
    """Start the comfyui-client API server."""
    _host = host or os.environ.get("COMFYUI_SERVE_HOST", "0.0.0.0")
    _port = port or int(os.environ.get("COMFYUI_SERVE_PORT", "3000"))
    uvicorn.run(app, host=_host, port=_port, log_level=log_level)


def serve_cli():
    serve_typer()
