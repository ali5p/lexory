import requests
from .base import BaseLLM


class OllamaAdapter(BaseLLM):
    def __init__(
        self,
        model: str = "qwen2:1.5b",
        url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.url = url

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        return data["response"]