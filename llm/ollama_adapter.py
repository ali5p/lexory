import logging
import os
from typing import Any, List, Mapping, Optional
from urllib.parse import urlparse, urlunparse

import requests

from .base import BaseLLM

_log = logging.getLogger(__name__)


def _ollama_generate_url() -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    qdrant_url = os.getenv("QDRANT_URL", "")
    if "qdrant" in qdrant_url and host in ("localhost", "127.0.0.1"):
        netloc = f"ollama:{parsed.port}" if parsed.port else "ollama"
        parsed = parsed._replace(netloc=netloc)
        url = urlunparse(parsed)
    return url


def _chat_url_from_generate(generate_url: str) -> str:
    g = generate_url.strip()
    if "/api/generate" in g:
        return g.replace("/api/generate", "/api/chat", 1)
    parsed = urlparse(g)
    netloc = parsed.netloc or "localhost:11434"
    scheme = parsed.scheme or "http"
    return urlunparse((scheme, netloc, "/api/chat", "", "", ""))


class OllamaAdapter(BaseLLM):
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
        self._generate_url = _ollama_generate_url()
        self._chat_url = _chat_url_from_generate(self._generate_url)
        self._timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            json_schema=None,
        )

    def chat(
        self,
        messages: List[Mapping[str, str]],
        *,
        temperature: float = 0.1,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        structured = (
            json_schema is not None
            and os.getenv("OLLAMA_STRUCTURED_OUTPUT", "1").strip().lower()
            not in ("0", "false", "no")
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "temperature": temperature,
        }
        if structured and json_schema is not None:
            body["format"] = json_schema

        response = requests.post(
            self._chat_url,
            json=body,
            timeout=self._timeout,
        )

        if response.status_code == 400 and structured and json_schema is not None:
            _log.warning(
                "Ollama rejected structured format; retrying without JSON schema "
                "(upgrade Ollama or set OLLAMA_STRUCTURED_OUTPUT=0 to skip format)."
            )
            del body["format"]
            response = requests.post(
                self._chat_url,
                json=body,
                timeout=self._timeout,
            )

        response.raise_for_status()
        data = response.json()
        msg = data.get("message") or {}
        return str(msg.get("content", "") or "")
