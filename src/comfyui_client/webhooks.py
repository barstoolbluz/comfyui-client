"""Webhook delivery with HMAC-SHA256 signing (Standard Webhooks spec)."""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebhookPayload:
    """Payload delivered to webhook URL after job completion."""

    id: str
    status: str  # "completed" | "failed"
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "status": self.status,
            "images": self.images,
            "stats": self.stats,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


def sign_payload(msg_id: str, timestamp: int, body: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature per Standard Webhooks spec.

    Returns 'v1,{base64-encoded-signature}'.

    If secret starts with 'whsec_', the prefix is stripped and the
    remainder is base64-decoded to obtain the signing key bytes.
    Otherwise the secret string is base64-decoded directly.
    """
    if secret.startswith("whsec_"):
        secret = secret[len("whsec_"):]
    key = base64.b64decode(secret)

    content = f"{msg_id}.{timestamp}.{body}"
    sig = hmac.new(key, content.encode(), hashlib.sha256).digest()
    return f"v1,{base64.b64encode(sig).decode()}"


async def deliver_webhook(
    url: str,
    payload: WebhookPayload,
    secret: str | None = None,
) -> bool:
    """POST webhook with up to 3 retries (1s, 2s, 4s backoff).

    Returns True on any 2xx response, False after all retries exhausted.
    """
    body = json.dumps(payload.to_dict(), separators=(",", ":"))
    msg_id = f"msg_{payload.id}"
    ts = int(time.time())

    headers = {
        "content-type": "application/json",
        "webhook-id": msg_id,
        "webhook-timestamp": str(ts),
    }
    if secret:
        headers["webhook-signature"] = sign_payload(msg_id, ts, body, secret)

    backoffs = [1, 2, 4]
    for attempt, delay in enumerate(backoffs, 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, content=body, headers=headers)
            if 200 <= resp.status_code < 300:
                logger.info("Webhook delivered to %s (attempt %d)", url, attempt)
                return True
            logger.warning(
                "Webhook %s returned %d (attempt %d/%d)",
                url, resp.status_code, attempt, len(backoffs),
            )
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(
                "Webhook %s failed (attempt %d/%d): %s",
                url, attempt, len(backoffs), exc,
            )
        if attempt < len(backoffs):
            await asyncio.sleep(delay)

    logger.error("Webhook delivery to %s exhausted all %d retries", url, len(backoffs))
    return False
