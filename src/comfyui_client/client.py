"""ComfyUI API Client"""
import httpx
import uuid


class ComfyUIClient:
    def __init__(self, host: str = "localhost", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())

    def submit(self, workflow: dict) -> str:
        """Submit workflow, return prompt_id"""
        response = httpx.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id}
        )
        response.raise_for_status()
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

    def wait_for_completion(self, prompt_id: str, timeout: float = 300) -> dict:
        """Poll until workflow completes"""
        import time
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"Workflow {prompt_id} did not complete in {timeout}s")
