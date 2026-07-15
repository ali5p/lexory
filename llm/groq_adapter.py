"""Groq cloud LLM via the OpenAI-compatible chat completions API."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Mapping, Optional

import requests

from llm.base import BaseLLM

_log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqAdapter(BaseLLM):
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq"
            )
        self._api_key = api_key
        base = os.getenv("GROQ_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")
        self._chat_url = f"{base}/chat/completions"
        self.model = os.getenv("GROQ_MODEL", _DEFAULT_MODEL).strip()
        self._timeout = float(os.getenv("GROQ_TIMEOUT", "120"))
        self._structured = os.getenv("GROQ_STRUCTURED_OUTPUT", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,
        }
        use_structured = self._structured and json_schema is not None
        if use_structured:
            body["response_format"] = {"type": "json_object"}

        response = requests.post(
            self._chat_url,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )

        if response.status_code == 400 and use_structured:
            _log.warning(
                "Groq rejected structured response_format; retrying without it."
            )
            body.pop("response_format", None)
            response = requests.post(
                self._chat_url,
                headers=headers,
                json=body,
                timeout=self._timeout,
            )

        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("Groq response missing choices")
        message = choices[0].get("message") or {}
        return str(message.get("content", "") or "")
