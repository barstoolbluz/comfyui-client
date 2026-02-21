"""Workflow loading and modification utilities"""
import json
from pathlib import Path


def load_workflow(path: Path) -> dict:
    """Load workflow JSON file"""
    return json.loads(path.read_text())


def find_node_by_class(workflow: dict, class_type: str) -> tuple[str, dict] | None:
    """Find first node of given class type"""
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    return None


def find_all_nodes_by_class(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
    """Find all nodes of given class type"""
    return [
        (node_id, node)
        for node_id, node in workflow.items()
        if node.get("class_type") == class_type
    ]


def set_prompt(workflow: dict, positive: str, negative: str = "") -> dict:
    """Set positive/negative prompts in workflow"""
    for node_id, node in workflow.items():
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
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
            if "seed" in node.get("inputs", {}):
                node["inputs"]["seed"] = seed
            elif "noise_seed" in node.get("inputs", {}):
                node["inputs"]["noise_seed"] = seed
    return workflow


def set_steps(workflow: dict, steps: int) -> dict:
    """Set sampling steps in KSampler nodes"""
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
            if "steps" in node.get("inputs", {}):
                node["inputs"]["steps"] = steps
    return workflow


def set_cfg(workflow: dict, cfg: float) -> dict:
    """Set CFG scale in KSampler nodes"""
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced"):
            if "cfg" in node.get("inputs", {}):
                node["inputs"]["cfg"] = cfg
    return workflow


def set_dimensions(workflow: dict, width: int, height: int) -> dict:
    """Set image dimensions in EmptyLatentImage nodes"""
    for node_id, node in workflow.items():
        if node.get("class_type") == "EmptyLatentImage":
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
    return workflow


def set_denoise(workflow: dict, denoise: float) -> dict:
    """Set denoise strength in KSampler nodes (for img2img)"""
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced"):
            if "denoise" in node.get("inputs", {}):
                node["inputs"]["denoise"] = denoise
    return workflow


def set_sampler(workflow: dict, sampler_name: str) -> dict:
    """Set sampler in KSampler nodes"""
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced"):
            if "sampler_name" in node.get("inputs", {}):
                node["inputs"]["sampler_name"] = sampler_name
    return workflow


def set_scheduler(workflow: dict, scheduler: str) -> dict:
    """Set scheduler in KSampler nodes"""
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in ("KSampler", "KSamplerAdvanced"):
            if "scheduler" in node.get("inputs", {}):
                node["inputs"]["scheduler"] = scheduler
    return workflow


def set_input_image(workflow: dict, image_path: str) -> dict:
    """Set input image path in LoadImage nodes"""
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadImage":
            node["inputs"]["image"] = image_path
    return workflow
