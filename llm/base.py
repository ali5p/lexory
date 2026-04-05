from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate text from a single user prompt (legacy / simple callers)."""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: List[Mapping[str, str]],
        *,
        temperature: float = 0.1,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Chat completion. `messages` use Ollama roles: system, user, assistant.
        When `json_schema` is set, Ollama constrains output (if supported).
        Return the assistant message content (plain text or JSON string).
        """
        raise NotImplementedError