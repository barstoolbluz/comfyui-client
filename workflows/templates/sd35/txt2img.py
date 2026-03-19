"""SD 3.5 text-to-image workflow template."""

import random
from pathlib import Path

from pydantic import BaseModel, Field

from comfyui_client.workflow import (
    apply_params, load_workflow, set_sd3_prompt,
)

_JSON = Path(__file__).resolve().parents[2] / "api" / "sd35" / "sd35-txt2img.json"

description = "Generate an image from text using Stable Diffusion 3.5"


class Request(BaseModel):
    prompt: str
    negative: str = ""
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=4.5, ge=0, le=30)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    sampler: str | None = None
    scheduler: str | None = None


def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    set_sd3_prompt(workflow, params.prompt, params.negative)
    apply_params(
        workflow, seed=params.seed, steps=params.steps, cfg=params.cfg,
        width=params.width, height=params.height,
        sampler=params.sampler, scheduler=params.scheduler,
    )
    return workflow
