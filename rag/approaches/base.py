from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import ContextAssembly


class BaseApproach(ABC):
    @abstractmethod
    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_exercises(self, primary_mistake_context: Optional[dict]) -> List[str]:
        raise NotImplementedError


class StubApproachHandler(BaseApproach):
    """Deterministic stub for lesson generation. Used when GENERATOR_MODE=stub."""

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        return f"[STUB] Explanation for topic: {topic or 'unknown'}"

    def generate_exercises(
        self, primary_mistake_context: Optional[object]
    ) -> List[str]:
        if primary_mistake_context is None:
            return []
        mistake_type = (
            getattr(primary_mistake_context, "mistake_type", None)
            or (
                primary_mistake_context.get("mistake_type", "unknown")
                if isinstance(primary_mistake_context, dict)
                else "unknown"
            )
        )
        return [f"Fix the mistake type: {mistake_type}"]
