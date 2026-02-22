"""ComfyUI API Client"""
import asyncio
import json

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

    def get_image(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        """Download image from ComfyUI"""
        response = httpx.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type}
        )
        response.raise_for_status()
        return response.content

    async def _ws_wait(self, prompt_id: str, on_progress=None) -> dict:
        """Listen on WebSocket until workflow completes"""
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
        return asyncio.run(self._ws_wait(prompt_id, on_progress))
