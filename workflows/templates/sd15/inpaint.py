"""SD 1.5 inpainting workflow template."""

import random
from pathlib import Path

from pydantic import BaseModel, Field

from comfyui_client.workflow import apply_params, load_workflow, set_inpaint_images

_JSON = Path(__file__).resolve().parents[2] / "api" / "sd15" / "sd15-inpaint.json"

description = "Inpaint a masked region using Stable Diffusion 1.5"


class Request(BaseModel):
    prompt: str
    negative: str = ""
    image: str
    mask: str
    seed: int = Field(default_factory=lambda: random.randint(0, 2**32 - 1))
    steps: int = Field(default=20, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=0, le=30)
    denoise: float = Field(default=0.8, ge=0, le=1)
    sampler: str | None = None
    scheduler: str | None = None


def generate(params: Request) -> dict:
    workflow = load_workflow(_JSON)
    apply_params(
        workflow, prompt=params.prompt, negative=params.negative,
        seed=params.seed, steps=params.steps, cfg=params.cfg,
        denoise=params.denoise,
        sampler=params.sampler, scheduler=params.scheduler,
    )
    set_inpaint_images(workflow, params.image, params.mask)
    return workflow
