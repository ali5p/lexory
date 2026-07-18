from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ContextAssembly


class BaseApproach(ABC):
    @abstractmethod
    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        raise NotImplementedError
