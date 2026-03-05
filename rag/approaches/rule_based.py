from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from core.models import ContextAssembly
from .base import BaseApproach

if TYPE_CHECKING:
    from llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are generating a short English grammar lesson.

Input:

Rule message from a grammar checker: {rule_message}

User sentence that contains the mistake: {example_sentence}

Tasks:

1. Identify the grammatical mistake in the sentence using the rule message as a hint.
2. Determine the grammar topic related to this mistake.
3. Write a short lesson explaining this grammar topic.
4. Create one short exercise similar to the user's sentence.
5. If the rule message is vague, focus mainly on the sentence to determine the topic.

Return JSON:

{{
"topic": "",
"lesson": "",
"exercise": ""
}}
"""


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text (handles markdown code blocks or extra text)."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class RuleBasedApproach(BaseApproach):
    def __init__(self, llm: Optional["BaseLLM"] = None):
        self.llm = llm
        self._last_llm_result: dict = {}

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        if self.llm is not None:
            return self._build_explanation_llm(context, topic)
        return self._build_explanation_rule_based(context, topic)

    def _build_explanation_llm(self, context: ContextAssembly, topic: str) -> str:
        self._last_llm_result = {}
        rule_message = ""
        example_sentence = ""
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            rule_message = p.get("rule_message", "") or ""
            examples = p.get("examples", [])
            example_sentence = (examples[0] if examples else "") or ""

        prompt = _PROMPT_TEMPLATE.format(
            rule_message=rule_message,
            example_sentence=example_sentence,
        )

        try:
            response_text = self.llm.generate(prompt)
        except Exception as e:
            err_msg = f"LLM generation failed: {e!s}"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        json_str = _extract_json(response_text)
        if not json_str:
            err_msg = "LLM generation failed: invalid JSON response"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            err_msg = f"LLM generation failed: invalid JSON response ({e!s})"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        required_keys = ("topic", "lesson", "exercise")
        if not all(k in parsed for k in required_keys):
            missing = [k for k in required_keys if k not in parsed]
            err_msg = f"LLM generation failed: missing keys {missing}"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        self._last_llm_result = {
            "topic": str(parsed.get("topic", "")).strip() or topic,
            "explanation": str(parsed.get("lesson", "")).strip(),
            "exercises": [str(parsed.get("exercise", "")).strip()] if parsed.get("exercise") else [],
            "approach_type": "llm",
        }
        return self._last_llm_result["explanation"]

    def _build_explanation_rule_based(self, context: ContextAssembly, topic: str) -> str:
        self._last_llm_result = {}  # Clear so _construct_lesson does not use stale override
        # Uses explicit mistake_type description and long-term context; matches prior behavior.
        parts: List[str] = []
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            mistake_type_desc = p.get("description", "")
            rule_message = p.get("rule_message", "")  # LanguageTool message for lesson context
            if rule_message:
                parts.append(f"LanguageTool: {rule_message}")
            elif mistake_type_desc:
                parts.append(f"Mistake Type: {mistake_type_desc}")
        if context.long_term_dynamics:
            summary_content = context.long_term_dynamics[0].get("content", "")
            if summary_content:
                parts.append(f"Context: {summary_content[:200]}")
        if not parts:
            parts.append(f"Topic: {topic}")
        return " ".join(parts)[:500]

    def generate_exercises(self, primary_mistake_context: Optional[dict]) -> List[str]:
        if self.llm and self._last_llm_result:
            return self._last_llm_result.get("exercises", [])
        exercises: List[str] = []
        if primary_mistake_context:
            examples = primary_mistake_context.get("examples", [])
            if examples:
                for example in examples[:2]:
                    exercises.append(f"Practice: {example}")
        if not exercises:
            exercises.append("Complete the sentence with the correct form.")
            exercises.append("Identify and correct the mistake in the given text.")
        return exercises[:3]
