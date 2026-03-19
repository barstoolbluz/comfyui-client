"""SDXL upscale workflow template."""

import random
from pathlib import Path

from pydantic import BaseModel, Field

from comfyui_client.workflow import (
    apply_params, load_workflow, set_input_image, set_upscale_params,
)

_JSON = Path(__file__).resolve().parents[2] / "api" / "sdxl" / "sdxl-upscale.json"

description = "Upscale an image using SDXL with UltimateSDUpscale"


class Request(BaseModel):
    prompt: str
    negative: str = ""
    image: str
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=20, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=0, le=30)
    denoise: float = Field(default=0.25, ge=0, le=1)
    upscale_by: float = Field(default=2, ge=1, le=8)
    sampler: str | None = None
    scheduler: str | None = None


def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    apply_params(
        workflow, prompt=params.prompt, negative=params.negative,
    )
    set_input_image(workflow, params.image)
    set_upscale_params(
        workflow, seed=params.seed, steps=params.steps, cfg=params.cfg,
        denoise=params.denoise, upscale_by=params.upscale_by,
        sampler_name=params.sampler, scheduler=params.scheduler,
    )
    return workflow
