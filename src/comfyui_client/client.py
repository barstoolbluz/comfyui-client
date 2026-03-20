"""ComfyUI API Client"""
import asyncio
import json
from pathlib import Path

import httpx
import uuid
import websockets


class ComfyUIClient:
    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"

    def submit(self, workflow: dict) -> str:
        """Submit workflow, return prompt_id"""
        response = httpx.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id}
        )
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                node_errors = error_data.get("node_errors", {})
                if node_errors:
                    error_msg += f"\nNode errors: {node_errors}"
            except Exception:
                error_msg = response.text
            raise RuntimeError(f"ComfyUI error ({response.status_code}): {error_msg}")
        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for prompt_id"""
        response = httpx.get(f"{self.base_url}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    def get_queue(self) -> dict:
        """Get current queue status"""
        response = httpx.get(f"{self.base_url}/queue")
        response.raise_for_status()
        return response.json()

    def interrupt(self):
        """Interrupt the currently running workflow"""
        response = httpx.post(f"{self.base_url}/interrupt", json={})
        response.raise_for_status()

    def delete_from_queue(self, prompt_ids: list[str]):
        """Delete specific prompt IDs from the queue"""
        response = httpx.post(f"{self.base_url}/queue", json={"delete": prompt_ids})
        response.raise_for_status()

    def clear_queue(self):
        """Clear all pending items from the queue"""
        response = httpx.post(f"{self.base_url}/queue", json={"clear": True})
        response.raise_for_status()

    def get_system_stats(self) -> dict:
        """Get system stats (versions, RAM, devices)"""
        response = httpx.get(f"{self.base_url}/system_stats")
        response.raise_for_status()
        return response.json()

    def get_model_types(self) -> list[str]:
        """Get available model folder types"""
        response = httpx.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def get_models(self, folder: str) -> list[str]:
        """Get models in a specific folder"""
        response = httpx.get(f"{self.base_url}/models/{folder}")
        response.raise_for_status()
        return response.json()

    def get_image(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        """Download image from ComfyUI"""
        response = httpx.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type}
        )
        response.raise_for_status()
        return response.content

    async def async_wait_for_completion(self, prompt_id: str, on_progress=None) -> dict:
        """Async wait for workflow completion via WebSocket.

        Can be awaited directly in async contexts (e.g., FastAPI).
        """
        # Fast path: if already completed (cached/instant execution), return now
        history = self.get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]

        async with websockets.connect(self.ws_url, max_size=2**24) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=1800)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if data.get("prompt_id") != prompt_id:
                    continue

                if on_progress:
                    on_progress(msg_type, data)

                if msg_type == "executing" and data.get("node") is None:
                    break
                elif msg_type == "execution_error":
                    raise RuntimeError(
                        f"Workflow error: {data.get('exception_message', 'unknown')}"
                    )

        history = self.get_history(prompt_id)
        return history[prompt_id]

    def wait_for_completion(self, prompt_id: str, on_progress=None) -> dict:
        """Wait for workflow completion via WebSocket"""
        return asyncio.run(self.async_wait_for_completion(prompt_id, on_progress))

    def download_images(self, result: dict, output_dir: Path, prefix: str = None) -> list[Path]:
        """Download output images from a completed workflow result.

        Returns list of saved file paths.
        """
        saved = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for node_id, node_output in result.get("outputs", {}).items():
            for img in node_output.get("images", []):
                data = self.get_image(img["filename"], img.get("subfolder", ""))
                fname = Path(img["filename"]).name
                if prefix:
                    fname = f"{prefix}_{fname}"
                dest = output_dir / fname
                dest.write_bytes(data)
                saved.append(dest)
        return saved
