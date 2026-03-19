"""Image format conversion utilities for API server responses."""

import base64
import io
import os


SUPPORTED_FORMATS = {"png", "jpeg", "webp"}

CONTENT_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def convert_image(
    image_data: bytes, target_format: str, quality: int = 85
) -> tuple[bytes, str]:
    """Convert raw image bytes to target format.

    Returns (converted_bytes, content_type).
    Fast-path: PNG-to-PNG skips conversion entirely.
    """
    target_format = target_format.lower()
    if target_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {target_format!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    content_type = CONTENT_TYPES[target_format]

    # Fast path: PNG input staying as PNG needs no conversion
    if target_format == "png" and image_data[:8] == b"\x89PNG\r\n\x1a\n":
        return image_data, content_type

    from PIL import Image

    img = Image.open(io.BytesIO(image_data))

    # JPEG doesn't support alpha — convert RGBA to RGB
    if target_format == "jpeg" and img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    save_format = "JPEG" if target_format == "jpeg" else target_format.upper()
    img.save(buf, format=save_format, quality=quality)
    return buf.getvalue(), content_type


def encode_image_for_payload(
    image_data: bytes,
    filename: str,
    target_format: str = "png",
    quality: int = 85,
) -> dict[str, str]:
    """Convert image and base64-encode for webhook/API payload.

    Returns {"filename": str, "data": str, "content_type": str}.
    """
    converted, content_type = convert_image(image_data, target_format, quality)

    # Replace extension in filename
    ext = "jpg" if target_format == "jpeg" else target_format
    base = os.path.splitext(filename)[0]
    new_filename = f"{base}.{ext}"

    return {
        "filename": new_filename,
        "data": base64.b64encode(converted).decode("ascii"),
        "content_type": content_type,
    }
