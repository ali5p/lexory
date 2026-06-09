from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from core.models import ContextAssembly


class BaseApproach(ABC):
    @abstractmethod
    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_exercises(self, primary_mistake_context: Optional[Any]) -> List[str]:
        raise NotImplementedError
