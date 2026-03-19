"""SD 3.5 image-to-image workflow template."""

import random
from pathlib import Path

from pydantic import BaseModel, Field

from comfyui_client.workflow import (
    apply_params, load_workflow, set_input_image, set_sd3_prompt,
)

_JSON = Path(__file__).resolve().parents[2] / "api" / "sd35" / "sd35-img2img.json"

description = "Transform an image using Stable Diffusion 3.5"


class Request(BaseModel):
    prompt: str
    negative: str = ""
    image: str
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=4.5, ge=0, le=30)
    denoise: float = Field(default=0.75, ge=0, le=1)
    sampler: str | None = None
    scheduler: str | None = None


def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    set_sd3_prompt(workflow, params.prompt, params.negative)
    set_input_image(workflow, params.image)
    apply_params(
        workflow, seed=params.seed, steps=params.steps, cfg=params.cfg,
        denoise=params.denoise,
        sampler=params.sampler, scheduler=params.scheduler,
    )
    return workflow
