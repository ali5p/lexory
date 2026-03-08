import os
from urllib.parse import urlparse, urlunparse

import requests
from .base import BaseLLM


def _ollama_url() -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    # In Docker, localhost points to the container; use service name instead
    if os.getenv("QDRANT_URL"):
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        if host in ("localhost", "127.0.0.1"):
            netloc = f"ollama:{parsed.port}" if parsed.port else "ollama"
            parsed = parsed._replace(netloc=netloc)
            url = urlunparse(parsed)
    return url


class OllamaAdapter(BaseLLM):
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
        self.url = _ollama_url()

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
            },
            timeout=65,
        )

        response.raise_for_status()
        data = response.json()

        return data["response"]