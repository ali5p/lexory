"""Structured exercise models, validation, and LLM schema helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator


def expose_exercise_answers() -> bool:
    """When true, /submit includes dev_answer_key (Swagger/local testing only)."""
    return os.getenv("LEXORY_EXPOSE_EXERCISE_ANSWERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


_MCQ_MAX_OPTION_WORDS = 6
_MCQ_BLANK_MARKERS = ("___", "____")


def _has_blank_marker(text: str) -> bool:
    return any(marker in text for marker in _MCQ_BLANK_MARKERS)


def _blank_marker_count(text: str) -> int:
    """Count blank slots; treat ____ as one marker, not two ___."""
    normalized = text.replace("____", "\0")
    return normalized.count("___")


def _mcq_option_looks_like_full_sentence(option: str) -> bool:
    """Reject whole-sentence options; MCQ choices must be words or short phrases."""
    text = option.strip()
    if not text:
        return True
    if re.search(r"[.!?]", text):
        return True
    return len(text.split()) > _MCQ_MAX_OPTION_WORDS


# --- Public payloads (API → frontend; no answer keys) ---


class MultipleChoicePayload(BaseModel):
    type: Literal["multiple_choice"]
    instruction: str = ""
    sentence: str
    options: list[str]


class FillBlankPayload(BaseModel):
    type: Literal["fill_blank"]
    instruction: str = ""
    sentence: str
    hint: str | None = None


ExercisePayloadBody = Annotated[
    Union[MultipleChoicePayload, FillBlankPayload],
    Field(discriminator="type"),
]


# --- Server-only answer keys ---


class MultipleChoiceAnswerKey(BaseModel):
    type: Literal["multiple_choice"]
    correct_option: str
    explanation_on_success: str = ""
    explanation_on_error: str = ""


class FillBlankAnswerKey(BaseModel):
    type: Literal["fill_blank"]
    accepted_answers: list[str]
    explanation_on_success: str = ""
    explanation_on_error: str = ""


ExerciseAnswerKey = Annotated[
    Union[MultipleChoiceAnswerKey, FillBlankAnswerKey],
    Field(discriminator="type"),
]


class ExercisePayload(BaseModel):
    """User-facing exercise object returned by /submit and exercise endpoints."""

    exercise_id: str
    mistake_type: str = ""
    source_sentence: str = ""
    payload: ExercisePayloadBody
    dev_answer_key: ExerciseAnswerKey | None = Field(
        default=None,
        description=(
            "Present only when LEXORY_EXPOSE_EXERCISE_ANSWERS=1 "
            "(local/Swagger testing; omitted in production UI)."
        ),
    )


_answer_key_adapter: TypeAdapter[ExerciseAnswerKey] = TypeAdapter(ExerciseAnswerKey)
_payload_body_adapter: TypeAdapter[ExercisePayloadBody] = TypeAdapter(ExercisePayloadBody)


# --- LLM-generated exercises (before persistence) ---


class GeneratedMultipleChoice(BaseModel):
    type: Literal["multiple_choice"]
    instruction: str = ""
    sentence: str
    options: list[str]
    correct_answer: str
    explanation_on_success: str = ""
    explanation_on_error: str = ""

    @model_validator(mode="after")
    def _validate_options(self) -> GeneratedMultipleChoice:
        if not _has_blank_marker(self.sentence):
            raise ValueError("multiple_choice sentence must contain a blank marker (___)")
        if _blank_marker_count(self.sentence) != 1:
            raise ValueError("multiple_choice sentence must contain exactly one blank")
        if len(self.options) < 2:
            raise ValueError("multiple_choice requires at least 2 options")
        if any(_mcq_option_looks_like_full_sentence(option) for option in self.options):
            raise ValueError("multiple_choice options must be words or short phrases")
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer must be one of options")
        return self


class GeneratedFillBlank(BaseModel):
    type: Literal["fill_blank"]
    instruction: str = ""
    sentence: str
    answer: str
    hint: str | None = None
    explanation_on_success: str = ""
    explanation_on_error: str = ""

    @model_validator(mode="after")
    def _validate_blank(self) -> GeneratedFillBlank:
        if not _has_blank_marker(self.sentence):
            raise ValueError("fill_blank sentence must contain a blank marker (___)")
        if _blank_marker_count(self.sentence) != 1:
            raise ValueError("fill_blank sentence must contain exactly one blank")
        if not self.answer.strip():
            raise ValueError("fill_blank answer must be non-empty")
        return self


GeneratedExercise = Annotated[
    Union[GeneratedMultipleChoice, GeneratedFillBlank],
    Field(discriminator="type"),
]

_generated_adapter: TypeAdapter[GeneratedExercise] = TypeAdapter(GeneratedExercise)


# --- Answer submission ---


class ExerciseAnswerRequest(BaseModel):
    user_id: str
    selected_option: str | None = None
    answer: str | None = None


class ExerciseAnswerResponse(BaseModel):
    correct: bool
    explanation: str
    exercise_attempt_id: str


EXERCISES_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exercises": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "multiple_choice"},
                            "instruction": {"type": "string"},
                            "sentence": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 4,
                            },
                            "correct_answer": {"type": "string"},
                            "explanation_on_success": {"type": "string"},
                            "explanation_on_error": {"type": "string"},
                        },
                        "required": [
                            "type",
                            "sentence",
                            "options",
                            "correct_answer",
                        ],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "fill_blank"},
                            "instruction": {"type": "string"},
                            "sentence": {"type": "string"},
                            "answer": {"type": "string"},
                            "hint": {"type": "string"},
                            "explanation_on_success": {"type": "string"},
                            "explanation_on_error": {"type": "string"},
                        },
                        "required": ["type", "sentence", "answer"],
                    },
                ],
            },
        }
    },
    "required": ["exercises"],
}


def exercise_response_json_schema(exercise_type: str) -> dict[str, Any]:
    """JSON schema for exactly one exercise of the requested type."""
    items_schema = EXERCISES_RESPONSE_JSON_SCHEMA["properties"]["exercises"]["items"]
    one_of = items_schema["oneOf"]
    selected = next(
        (branch for branch in one_of if branch["properties"]["type"]["const"] == exercise_type),
        one_of[0],
    )
    return {
        "type": "object",
        "properties": {
            "exercises": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": selected,
            }
        },
        "required": ["exercises"],
    }


def _normalize_llm_exercise_item(item: Any) -> dict[str, Any] | None:
    """Fix common small-model mistakes before Pydantic validation."""
    if not isinstance(item, dict):
        return None

    exercise_type = str(item.get("type", "")).strip()
    if exercise_type == "multiple_choice":
        options = [str(o).strip() for o in item.get("options", []) if str(o).strip()]
        correct = str(item.get("correct_answer", "")).strip()
        sentence = str(item.get("sentence", "") or item.get("question", "")).strip()
        if not options or not correct or not sentence:
            return None
        if not _has_blank_marker(sentence):
            return None
        if _blank_marker_count(sentence) != 1:
            return None
        if any(_mcq_option_looks_like_full_sentence(option) for option in options):
            return None
        matched = next((o for o in options if o.lower() == correct.lower()), None)
        if matched:
            correct = matched
        elif correct not in options:
            if len(options) >= 4:
                options = options[:3] + [correct]
            else:
                options = options + [correct]
        return {
            **item,
            "type": "multiple_choice",
            "sentence": sentence,
            "options": options,
            "correct_answer": correct,
        }

    if exercise_type == "fill_blank":
        sentence = str(item.get("sentence", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not sentence or not answer:
            return None
        if not _has_blank_marker(sentence):
            word_pattern = re.compile(rf"\b{re.escape(answer)}\b", re.IGNORECASE)
            if word_pattern.search(sentence):
                sentence = word_pattern.sub("___", sentence, count=1)
            else:
                loose = re.compile(re.escape(answer), re.IGNORECASE)
                if loose.search(sentence):
                    sentence = loose.sub("___", sentence, count=1)
        if not _has_blank_marker(sentence):
            return None
        if _blank_marker_count(sentence) != 1:
            return None
        return {**item, "type": "fill_blank", "sentence": sentence, "answer": answer}

    return None


def parse_generated_exercises(
    raw: Any,
    *,
    expected_type: str | None = None,
) -> list[GeneratedExercise]:
    """Validate LLM exercise array; keeps valid items, drops broken ones."""
    if not isinstance(raw, list):
        raise ValueError("exercises must be a list")
    out: list[GeneratedExercise] = []
    for item in raw:
        normalized = _normalize_llm_exercise_item(item)
        if normalized is None:
            continue
        if expected_type and normalized.get("type") != expected_type:
            continue
        try:
            out.append(_generated_adapter.validate_python(normalized))
        except Exception:
            continue
    return out


def split_generated_exercise(
    generated: GeneratedExercise,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split validated LLM output into public payload and server answer_key dicts."""
    if isinstance(generated, GeneratedMultipleChoice):
        payload = MultipleChoicePayload(
            type="multiple_choice",
            instruction=generated.instruction,
            sentence=generated.sentence,
            options=list(generated.options),
        )
        answer_key = MultipleChoiceAnswerKey(
            type="multiple_choice",
            correct_option=generated.correct_answer,
            explanation_on_success=generated.explanation_on_success,
            explanation_on_error=generated.explanation_on_error,
        )
    else:
        payload = FillBlankPayload(
            type="fill_blank",
            instruction=generated.instruction,
            sentence=generated.sentence,
            hint=generated.hint,
        )
        answer_key = FillBlankAnswerKey(
            type="fill_blank",
            accepted_answers=[generated.answer.strip()],
            explanation_on_success=generated.explanation_on_success,
            explanation_on_error=generated.explanation_on_error,
        )
    return payload.model_dump(), answer_key.model_dump()


def build_exercise_payload(
    *,
    exercise_id: str,
    mistake_type: str,
    source_sentence: str,
    payload: dict[str, Any],
    answer_key: dict[str, Any] | None = None,
) -> ExercisePayload:
    body = _payload_body_adapter.validate_python(payload)
    dev_key = None
    if expose_exercise_answers() and answer_key:
        dev_key = _answer_key_adapter.validate_python(answer_key)
    return ExercisePayload(
        exercise_id=exercise_id,
        mistake_type=mistake_type,
        source_sentence=source_sentence,
        payload=body,
        dev_answer_key=dev_key,
    )


def validate_exercise_answer(
    payload: dict[str, Any],
    answer_key: dict[str, Any],
    request: ExerciseAnswerRequest,
) -> tuple[bool, str]:
    """Deterministic grading. Returns (is_correct, explanation)."""
    body = _payload_body_adapter.validate_python(payload)
    key = _answer_key_adapter.validate_python(answer_key)

    if body.type == "multiple_choice":
        if not isinstance(key, MultipleChoiceAnswerKey):
            raise ValueError("answer_key type mismatch for multiple_choice")
        if request.selected_option is None:
            raise ValueError("selected_option is required for multiple_choice")
        selected = request.selected_option.strip()
        if selected == key.correct_option:
            return True, key.explanation_on_success or "Correct."
        return False, key.explanation_on_error or f"The correct answer is: {key.correct_option}."

    if not isinstance(key, FillBlankAnswerKey):
        raise ValueError("answer_key type mismatch for fill_blank")
    if request.answer is None:
        raise ValueError("answer is required for fill_blank")
    normalized = _normalize_text(request.answer)
    accepted = {_normalize_text(a) for a in key.accepted_answers if a.strip()}
    if normalized in accepted:
        return True, key.explanation_on_success or "Correct."
    primary = key.accepted_answers[0] if key.accepted_answers else ""
    return False, key.explanation_on_error or f"The correct answer is: {primary}."


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object from LLM output (handles fenced or extra text)."""
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
