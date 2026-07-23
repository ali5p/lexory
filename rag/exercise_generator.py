"""LLM exercise generation (separate from lesson explanation)."""

from __future__ import annotations

import logging
import time
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
2. Use a SHORT sentence focused on the error (not necessarily the full learner sentence).
3. sentence MUST contain exactly one "___" blank.
4. options MUST be short words or phrases only (2–4 options). Do NOT use full sentences as options.
5. correct_answer must exactly match one option string.
6. Distractors must be plausible learner mistakes.
7. instruction is optional and short (under 12 words); leave empty if unnecessary.

Respond with JSON only:
{"exercises": [{"type": "multiple_choice", "instruction": "", "sentence": "She ___ to school.", "options": ["goes", "go", "going"], "correct_answer": "goes"}]}
"""

_FB_INSTRUCTIONS = """You are an English grammar drill writer. Produce exactly ONE fill_blank exercise as structured JSON.

The user message specifies the required exercise type and the learner sentence to base the drill on.

Rules:
1. Drill ONLY the error from the rule message — the sentence may contain other grammar mistakes; ignore them.
2. Use a SHORT sentence focused on the error (not necessarily the full learner sentence).
3. sentence MUST contain exactly one "___" blank.
4. answer MUST be the word or phrase that fills the blank (required field).
5. instruction is optional and short (under 12 words); leave empty if unnecessary.

Respond with JSON only:
{"exercises": [{"type": "fill_blank", "instruction": "", "sentence": "She ___ looking everywhere.", "answer": "is"}]}
"""

_MAX_GENERATION_ATTEMPTS = 2
_MAX_TRANSIENT_HTTP_RETRIES = 3
_TRANSIENT_HTTP_BACKOFF_SEC = 2.0

_RETRY_HINT_BY_TYPE = {
    "multiple_choice": (
        "Your previous JSON failed validation. Return one multiple_choice exercise with: "
        "exactly one ___ in sentence, short phrase options (not full sentences), "
        "and correct_answer matching one option."
    ),
    "fill_blank": (
        "Your previous JSON failed validation. Return one fill_blank exercise with: "
        "exactly one ___ in sentence and a non-empty answer field for that blank."
    ),
}


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


def _alternate_exercise_type(exercise_type: ExerciseType) -> ExerciseType:
    return "fill_blank" if exercise_type == "multiple_choice" else "multiple_choice"


def _chat_with_transient_retry(
    llm: "BaseLLM",
    messages: list[Mapping[str, str]],
    *,
    exercise_type: ExerciseType,
) -> str | None:
    """Call LLM; retry briefly on rate-limit / upstream errors."""
    last_error: requests.RequestException | None = None
    for http_attempt in range(_MAX_TRANSIENT_HTTP_RETRIES):
        try:
            return llm.chat(
                messages,
                temperature=0.0,
                json_schema=exercise_response_json_schema(exercise_type),
            )
        except requests.RequestException as e:
            last_error = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (429, 502, 503, 504) and http_attempt + 1 < _MAX_TRANSIENT_HTTP_RETRIES:
                _log.warning(
                    "Exercise LLM HTTP %s (attempt %s/%s); retrying in %.0fs",
                    status,
                    http_attempt + 1,
                    _MAX_TRANSIENT_HTTP_RETRIES,
                    _TRANSIENT_HTTP_BACKOFF_SEC,
                )
                time.sleep(_TRANSIENT_HTTP_BACKOFF_SEC)
                continue
            _log.warning("Exercise LLM call failed: %s", e)
            return None
    if last_error is not None:
        _log.warning("Exercise LLM call failed after retries: %s", last_error)
    return None


def _generate_for_type(
    llm: "BaseLLM",
    *,
    user_message: str,
    exercise_type: ExerciseType,
) -> List[tuple[dict[str, Any], dict[str, Any]]]:
    messages: list[Mapping[str, str]] = [
        {"role": "system", "content": _system_prompt(exercise_type)},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(_MAX_GENERATION_ATTEMPTS):
        response_text = _chat_with_transient_retry(
            llm, messages, exercise_type=exercise_type
        )
        if response_text is None:
            return []

        parsed = extract_json_object(response_text)
        if not parsed or "exercises" not in parsed:
            _log.warning(
                "Exercise LLM returned unparseable JSON: %r",
                (response_text or "")[:500],
            )
            if attempt + 1 < _MAX_GENERATION_ATTEMPTS:
                messages.append(
                    {"role": "user", "content": _RETRY_HINT_BY_TYPE[exercise_type]}
                )
                continue
            return []

        generated = parse_generated_exercises(
            parsed["exercises"],
            expected_type=exercise_type,
        )
        if generated:
            return [split_generated_exercise(generated[0])]

        _log.warning(
            "Exercise validation dropped output for type %s (attempt %s/%s): %r",
            exercise_type,
            attempt + 1,
            _MAX_GENERATION_ATTEMPTS,
            parsed.get("exercises"),
        )
        if attempt + 1 < _MAX_GENERATION_ATTEMPTS:
            messages.append(
                {"role": "user", "content": _RETRY_HINT_BY_TYPE[exercise_type]}
            )

    return []


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

        pairs = _generate_for_type(
            self.llm,
            user_message=user_message,
            exercise_type=exercise_type,
        )
        if pairs:
            return pairs

        fallback_type = _alternate_exercise_type(exercise_type)
        _log.warning(
            "Exercise generation failed for %s; trying fallback type %s",
            exercise_type,
            fallback_type,
        )
        fallback_message = _user_message(
            rule_message=rule_message,
            example_sentence=example_sentence,
            mistake_type=mistake_type,
            rule_id=rule_id,
            topic=topic,
            explanation=explanation,
            exercise_type=fallback_type,
        )
        return _generate_for_type(
            self.llm,
            user_message=fallback_message,
            exercise_type=fallback_type,
        )
