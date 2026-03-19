"""SDXL text-to-image workflow template."""

import random
from pathlib import Path

from pydantic import BaseModel, Field

from comfyui_client.workflow import apply_params, load_workflow

_JSON = Path(__file__).resolve().parents[2] / "api" / "sdxl" / "sdxl-txt2img.json"

description = "Generate an image from text using SDXL"


class Request(BaseModel):
    prompt: str
    negative: str = ""
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=25, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=0, le=30)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    sampler: str | None = None
    scheduler: str | None = None


def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    return apply_params(
        workflow, prompt=params.prompt, negative=params.negative,
        seed=params.seed, steps=params.steps, cfg=params.cfg,
        width=params.width, height=params.height,
        sampler=params.sampler, scheduler=params.scheduler,
    )
