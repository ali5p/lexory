import os

import requests
from .base import BaseLLM


class OllamaAdapter(BaseLLM):
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2:1.5b")
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

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