"""LLM exercise generation (separate from lesson explanation)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Mapping

import requests

from core.exercise_rotation import ExerciseType
from core.exercises import (
    exercise_response_json_schema,
    extract_json_object,
    parse_generated_exercises,
    split_generated_exercise,
)
from core.models import ContextAssembly

if TYPE_CHECKING:
    from llm.base import BaseLLM

_log = logging.getLogger(__name__)

_MCQ_INSTRUCTIONS = """You are an English grammar drill writer. Produce exactly ONE multiple_choice exercise as structured JSON.

The user message specifies the required exercise type and the learner sentence to base the drill on.

Rules:
1. Drill ONLY the error from the rule message — the sentence may contain other grammar mistakes; ignore them.
2. Exactly one correct option; correct_answer must exactly match one option string.
3. Distractors must be plausible learner mistakes.
4. Keep instruction short (under 12 words).

Respond with JSON only:
{"exercises": [ { "type": "multiple_choice", ... } ]}
"""

_FB_INSTRUCTIONS = """You are an English grammar drill writer. Produce exactly ONE fill_blank exercise as structured JSON.

The user message specifies the required exercise type and the learner sentence to base the drill on.

Rules:
1. Drill ONLY the error from the rule message — the sentence may contain other grammar mistakes; ignore them.
2. The sentence MUST contain "___" where the learner fills the answer.
3. Keep instruction short (under 12 words).

Respond with JSON only:
{"exercises": [ { "type": "fill_blank", ... } ]}
"""


def _system_prompt(exercise_type: ExerciseType) -> str:
    if exercise_type == "fill_blank":
        return _FB_INSTRUCTIONS
    return _MCQ_INSTRUCTIONS


def _primary_fields(context: ContextAssembly) -> tuple[str, str, str, str]:
    rule_message = example_sentence = mistake_type = rule_id = ""
    if context.detected_mistake_examples:
        p = context.detected_mistake_examples[0]
        rule_message = p.rule_message or ""
        mistake_type = p.mistake_type or ""
        rule_id = p.rule_id or ""
        examples = p.examples
        example_sentence = (examples[0] if examples else "") or ""
    return rule_message, example_sentence, mistake_type, rule_id


def _user_message(
    *,
    rule_message: str,
    example_sentence: str,
    mistake_type: str,
    rule_id: str,
    topic: str,
    explanation: str,
    exercise_type: ExerciseType,
) -> str:
    return (
        f"Required exercise type:\n{exercise_type}\n\n"
        f"Mistake category:\n{mistake_type.strip() or '(none)'}\n\n"
        f"Focus rule id (ONLY drill this error — ignore others in the sentence):\n"
        f"{rule_id.strip() or '(none)'}\n\n"
        f"Rule message from grammar checker:\n{rule_message or '(none)'}\n\n"
        f"Lesson topic:\n{topic or '(none)'}\n\n"
        f"Lesson explanation:\n{explanation or '(none)'}\n\n"
        f"Learner sentence (basis for this exercise):\n{example_sentence or '(none)'}\n"
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
        exercise_type: ExerciseType,
    ) -> List[tuple[dict[str, Any], dict[str, Any]]]:
        """Return zero or one (payload, answer_key) pair for DB insert."""
        rule_message, example_sentence, mistake_type, rule_id = _primary_fields(context)
        if not mistake_type and not example_sentence:
            return []

        user_message = _user_message(
            rule_message=rule_message,
            example_sentence=example_sentence,
            mistake_type=mistake_type,
            rule_id=rule_id,
            topic=topic,
            explanation=explanation,
            exercise_type=exercise_type,
        )
        messages: list[Mapping[str, str]] = [
            {"role": "system", "content": _system_prompt(exercise_type)},
            {"role": "user", "content": user_message},
        ]

        try:
            response_text = self.llm.chat(
                messages,
                temperature=0.0,
                json_schema=exercise_response_json_schema(exercise_type),
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

        generated = parse_generated_exercises(
            parsed["exercises"],
            expected_type=exercise_type,
        )
        if not generated:
            _log.warning(
                "Exercise validation dropped output for type %s: %r",
                exercise_type,
                parsed.get("exercises"),
            )
            return []

        return [split_generated_exercise(generated[0])]
