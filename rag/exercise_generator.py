"""LLM exercise generation (separate from lesson explanation)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Mapping

import requests

from core.exercises import (
    EXERCISES_RESPONSE_JSON_SCHEMA,
    extract_json_object,
    parse_generated_exercises,
    split_generated_exercise,
)
from core.models import ContextAssembly

if TYPE_CHECKING:
    from llm.base import BaseLLM

_log = logging.getLogger(__name__)

_SYSTEM_EXERCISE_INSTRUCTIONS = """You are an English grammar drill writer. Produce 1–2 short practice exercises as structured JSON only.

The user message will contain:
- A mistake category (Lexory taxonomy label).
- A rule message from a grammar checker (hint).
- The learner's sentence with the mistake.
- The lesson topic and explanation already shown to the learner.

Rules:
1. Target ONLY the detected grammar mistake — do not invent unrelated errors.
2. Use the learner's sentence style when possible.
3. Prefer one multiple_choice and one fill_blank when two exercises fit; otherwise one exercise is fine.
4. For multiple_choice: exactly one correct option; distractors must be plausible learner mistakes (include the wrong form from the source sentence when relevant). correct_answer must exactly match one option string.
5. For fill_blank: the sentence MUST contain "___" where the learner fills the answer.
6. Keep instructions short (under 12 words).

Respond with JSON only. Use exactly this shape:
{"exercises": [ ... ]}

Example:
{"exercises": [
  {"type": "multiple_choice", "instruction": "Choose the correct form", "question": "She ___ to school every day.", "options": ["walks", "walk", "walking"], "correct_answer": "walks", "explanation_on_success": "Correct — third person -s.", "explanation_on_error": "Use walks with she."},
  {"type": "fill_blank", "instruction": "Fill in the blank", "sentence": "He ___ tennis on weekends.", "answer": "plays", "hint": "present simple -s", "explanation_on_success": "Good!", "explanation_on_error": "Use plays with he."}
]}
"""


def _primary_fields(context: ContextAssembly) -> tuple[str, str, str]:
    rule_message = example_sentence = mistake_type = ""
    if context.detected_mistake_examples:
        p = context.detected_mistake_examples[0]
        rule_message = p.rule_message or ""
        mistake_type = p.mistake_type or ""
        examples = p.examples
        example_sentence = (examples[0] if examples else "") or ""
    return rule_message, example_sentence, mistake_type


def _user_message(
    *,
    rule_message: str,
    example_sentence: str,
    mistake_type: str,
    topic: str,
    explanation: str,
) -> str:
    return (
        f"Mistake category:\n{mistake_type.strip() or '(none)'}\n\n"
        f"Rule message from grammar checker:\n{rule_message or '(none)'}\n\n"
        f"User sentence with the mistake:\n{example_sentence or '(none)'}\n\n"
        f"Lesson topic:\n{topic or '(none)'}\n\n"
        f"Lesson explanation:\n{explanation or '(none)'}\n"
    )


class ExerciseGenerator:
    def __init__(self, llm: "BaseLLM"):
        self.llm = llm

    def generate(
        self,
        context: ContextAssembly,
        *,
        topic: str,
        explanation: str,
    ) -> List[tuple[dict[str, Any], dict[str, Any]]]:
        """Return list of (payload, answer_key) dict pairs ready for DB insert."""
        rule_message, example_sentence, mistake_type = _primary_fields(context)
        if not mistake_type and not example_sentence:
            return []

        user_message = _user_message(
            rule_message=rule_message,
            example_sentence=example_sentence,
            mistake_type=mistake_type,
            topic=topic,
            explanation=explanation,
        )
        messages: list[Mapping[str, str]] = [
            {"role": "system", "content": _SYSTEM_EXERCISE_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ]

        try:
            response_text = self.llm.chat(
                messages,
                temperature=0.0,
                json_schema=EXERCISES_RESPONSE_JSON_SCHEMA,
            )
        except requests.RequestException as e:
            _log.warning("Exercise LLM call failed: %s", e)
            return []
        except (KeyError, ValueError, TypeError) as e:
            _log.warning("Exercise LLM invalid response: %s", e)
            return []

        parsed = extract_json_object(response_text)
        if not parsed or "exercises" not in parsed:
            _log.warning(
                "Exercise LLM returned unparseable JSON: %r",
                (response_text or "")[:500],
            )
            return []

        generated = parse_generated_exercises(parsed["exercises"])
        if not generated:
            _log.warning(
                "Exercise validation dropped all items from LLM output: %r",
                parsed.get("exercises"),
            )
            return []

        return [split_generated_exercise(item) for item in generated]
