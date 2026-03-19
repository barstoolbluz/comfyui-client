"""PNG metadata extraction for ComfyUI-generated images (stdlib only)"""
import json
import struct
import zlib
from pathlib import Path


def read_png_text_chunks(filepath: Path) -> dict[str, str]:
    """Read tEXt and iTXt chunks from a PNG file.

    PNG chunk format: 4-byte length (big-endian), 4-byte type, data, 4-byte CRC.
    tEXt chunks contain: keyword\\x00text
    iTXt chunks contain: keyword\\x00\\x00\\x00\\x00\\x00text (simplified)
    """
    chunks = {}
    with open(filepath, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return chunks

        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length, chunk_type = struct.unpack(">I4s", header)
            chunk_data = f.read(length)
            f.read(4)  # skip CRC

            if chunk_type == b"tEXt":
                sep = chunk_data.index(b"\x00")
                key = chunk_data[:sep].decode("latin-1")
                val = chunk_data[sep + 1:].decode("latin-1")
                chunks[key] = val
            elif chunk_type == b"iTXt":
                sep = chunk_data.index(b"\x00")
                key = chunk_data[:sep].decode("utf-8")
                rest = chunk_data[sep + 1:]
                # compression flag, compression method, language tag\0, translated keyword\0
                comp_flag = rest[0]
                # skip compression method (1 byte)
                rest = rest[2:]
                lang_end = rest.index(b"\x00")
                rest = rest[lang_end + 1:]
                trans_end = rest.index(b"\x00")
                rest = rest[trans_end + 1:]
                if comp_flag:
                    val = zlib.decompress(rest).decode("utf-8")
                else:
                    val = rest.decode("utf-8")
                chunks[key] = val
            elif chunk_type == b"IEND":
                break

    return chunks


def extract_comfyui_metadata(filepath: Path) -> dict:
    """Extract ComfyUI metadata from a PNG file.

    ComfyUI stores metadata as tEXt chunks:
    - 'prompt': API-format nodes JSON
    - 'workflow': web-UI-format workflow JSON
    """
    chunks = read_png_text_chunks(filepath)
    result = {}

    if "prompt" in chunks:
        try:
            result["prompt"] = json.loads(chunks["prompt"])
        except json.JSONDecodeError:
            result["prompt_raw"] = chunks["prompt"]

    if "workflow" in chunks:
        try:
            result["workflow"] = json.loads(chunks["workflow"])
        except json.JSONDecodeError:
            result["workflow_raw"] = chunks["workflow"]

    return result


def summarize_metadata(metadata: dict) -> dict:
    """Extract key generation parameters from ComfyUI prompt metadata."""
    summary = {}
    prompt = metadata.get("prompt", {})

    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
            if "seed" in inputs:
                summary["seed"] = inputs["seed"]
            elif "noise_seed" in inputs:
                summary["seed"] = inputs["noise_seed"]
            if "steps" in inputs:
                summary["steps"] = inputs["steps"]
            if "cfg" in inputs:
                summary["cfg"] = inputs["cfg"]
            if "sampler_name" in inputs:
                summary["sampler"] = inputs["sampler_name"]
            if "scheduler" in inputs:
                summary["scheduler"] = inputs["scheduler"]
            if "denoise" in inputs:
                summary["denoise"] = inputs["denoise"]

        elif class_type == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            text = inputs.get("text", "")
            if isinstance(text, str) and text:
                if "positive" in title or "prompt" in title:
                    summary["positive_prompt"] = text
                elif "negative" in title:
                    summary["negative_prompt"] = text

        elif class_type == "EmptyLatentImage":
            if "width" in inputs:
                summary["width"] = inputs["width"]
            if "height" in inputs:
                summary["height"] = inputs["height"]

        elif class_type == "CheckpointLoaderSimple":
            if "ckpt_name" in inputs:
                summary["model"] = inputs["ckpt_name"]

        elif class_type == "UNETLoader":
            if "unet_name" in inputs:
                summary.setdefault("model", inputs["unet_name"])

    return summary
