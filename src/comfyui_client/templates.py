"""Workflow template discovery and FastAPI route registration."""

import importlib.util
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def get_templates_dir() -> Path:
    """Resolve templates directory.

    Checks (in order):
    1. COMFYUI_WORKFLOW_DIR environment variable
    2. $FLOX_ENV/share/comfyui-client/workflows/templates
    3. Bundled location relative to this file (../../workflows/templates)
    """
    env_dir = os.environ.get("COMFYUI_WORKFLOW_DIR")
    if env_dir:
        return Path(env_dir)

    flox_env = os.environ.get("FLOX_ENV")
    if flox_env:
        p = Path(flox_env) / "share" / "comfyui-client" / "workflows" / "templates"
        if p.is_dir():
            return p

    return Path(__file__).resolve().parents[2] / "workflows" / "templates"


def discover_templates() -> dict[str, dict]:
    """Walk templates dir, import each .py module dynamically.

    Returns dict mapping route keys (e.g. "sd15/txt2img") to dicts with:
      - request_model: Pydantic BaseModel class (the module's Request)
      - generate: callable(params) -> dict (workflow)
      - description: str
    """
    templates_dir = get_templates_dir()
    templates = {}

    if not templates_dir.is_dir():
        logger.warning("Templates directory not found: %s", templates_dir)
        return templates

    for model_dir in sorted(templates_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        for py_file in sorted(model_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            model = model_dir.name
            operation = py_file.stem
            route_key = f"{model}/{operation}"

            spec = importlib.util.spec_from_file_location(
                f"templates.{model}.{operation}", py_file
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if not all(hasattr(mod, attr) for attr in ("Request", "generate", "description")):
                logger.warning("Skipping %s: missing Request, generate, or description", route_key)
                continue

            templates[route_key] = {
                "request_model": mod.Request,
                "generate": mod.generate,
                "description": mod.description,
            }

    logger.info("Discovered %d workflow templates", len(templates))
    return templates


def _make_handler(generate_fn, request_model):
    """Factory to create a route handler with correct closure binding."""
    async def handler(req: BaseModel):
        workflow = generate_fn(req)
        return {"prompt": workflow}

    handler.__annotations__["req"] = request_model
    return handler


def register_template_routes(app: FastAPI, templates: dict):
    """Register POST /workflow/{model}/{op} routes from discovered templates."""
    for route_key, tpl in templates.items():
        path = f"/workflow/{route_key}"
        handler = _make_handler(tpl["generate"], tpl["request_model"])
        app.post(path, summary=tpl["description"])(handler)
        logger.debug("Registered template route: POST %s", path)
