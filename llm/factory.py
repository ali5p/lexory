"""Select LLM backend from environment (Ollama local vs Groq cloud)."""

from __future__ import annotations

import os

from llm.base import BaseLLM
from llm.groq_adapter import GroqAdapter
from llm.ollama_adapter import OllamaAdapter

_SUPPORTED = frozenset({"ollama", "groq"})


def build_llm() -> BaseLLM:
    """Return the configured LLM adapter (``LLM_PROVIDER=ollama|groq``)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={provider!r}; use one of: {supported}"
        )
    if provider == "groq":
        return GroqAdapter()
    return OllamaAdapter()
