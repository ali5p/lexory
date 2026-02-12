from __future__ import annotations

from typing import List, Optional

from core.models import ContextAssembly
from .base import BaseApproach


class RuleBasedApproach(BaseApproach):
    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        # Uses explicit pattern description and long-term context; matches prior behavior.
        parts: List[str] = []
        if context.detected_patterns:
            p = context.detected_patterns[0]
            pattern_desc = p.get("description", "")
            rule_message = p.get("rule_message", "")  # LanguageTool message for lesson context
            if rule_message:
                parts.append(f"LanguageTool: {rule_message}")
            elif pattern_desc:
                parts.append(f"Pattern: {pattern_desc}")
        if context.long_term_dynamics:
            summary_content = context.long_term_dynamics[0].get("content", "")
            if summary_content:
                parts.append(f"Context: {summary_content[:200]}")
        if not parts:
            parts.append(f"Topic: {topic}")
        return " ".join(parts)[:500]

    def generate_exercises(self, primary_pattern: Optional[dict]) -> List[str]:
        exercises: List[str] = []
        if primary_pattern:
            examples = primary_pattern.get("examples", [])
            if examples:
                for example in examples[:2]:
                    exercises.append(f"Practice: {example}")
        if not exercises:
            exercises.append("Complete the sentence with the correct form.")
            exercises.append("Identify and correct the mistake in the given text.")
        return exercises[:3]
