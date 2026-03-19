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


def get_seed(workflow: dict) -> int | None:
    """Read current seed from first KSampler node"""
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
            inputs = node.get("inputs", {})
            if "seed" in inputs:
                return inputs["seed"]
            elif "noise_seed" in inputs:
                return inputs["noise_seed"]
    return None


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


def apply_params(workflow, prompt=None, negative=None, seed=None, steps=None,
                 cfg=None, width=None, height=None, denoise=None,
                 sampler=None, scheduler=None, image=None):
    """Apply parameter overrides to a workflow"""
    if prompt:
        workflow = set_prompt(workflow, prompt, negative or "")
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
    return workflow
