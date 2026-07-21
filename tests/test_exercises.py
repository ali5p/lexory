"""Tests for structured exercise validation and answer grading."""

import pytest

from core.exercises import (
    ExerciseAnswerRequest,
    GeneratedMultipleChoice,
    build_exercise_payload,
    parse_generated_exercises,
    split_generated_exercise,
    validate_exercise_answer,
)


def test_split_multiple_choice_hides_answer_key():
    generated = GeneratedMultipleChoice(
        type="multiple_choice",
        question="She ___ to school.",
        options=["goes", "go", "going"],
        correct_answer="goes",
        explanation_on_success="Good!",
        explanation_on_error="Use -s with she.",
    )
    payload, answer_key = split_generated_exercise(generated)
    assert "correct_answer" not in payload
    assert payload["options"] == ["goes", "go", "going"]
    assert answer_key["correct_option"] == "goes"


def test_validate_multiple_choice_correct_and_wrong():
    payload = {
        "type": "multiple_choice",
        "question": "He ___ tennis.",
        "options": ["plays", "play", "playing"],
    }
    answer_key = {
        "type": "multiple_choice",
        "correct_option": "plays",
        "explanation_on_success": "Nice!",
        "explanation_on_error": "Try plays.",
    }
    ok, msg = validate_exercise_answer(
        payload,
        answer_key,
        ExerciseAnswerRequest(user_id="u1", selected_option="plays"),
    )
    assert ok is True
    assert msg == "Nice!"

    ok, msg = validate_exercise_answer(
        payload,
        answer_key,
        ExerciseAnswerRequest(user_id="u1", selected_option="play"),
    )
    assert ok is False
    assert "Try plays." in msg


def test_validate_fill_blank_normalizes_case():
    payload = {
        "type": "fill_blank",
        "sentence": "She ___ to school yesterday.",
        "hint": "past simple",
    }
    answer_key = {
        "type": "fill_blank",
        "accepted_answers": ["went"],
        "explanation_on_success": "Correct.",
        "explanation_on_error": "Use went.",
    }
    ok, _ = validate_exercise_answer(
        payload,
        answer_key,
        ExerciseAnswerRequest(user_id="u1", answer="  Went "),
    )
    assert ok is True


def test_parse_generated_exercises_rejects_bad_mcq():
    result = parse_generated_exercises(
        [
            {
                "type": "multiple_choice",
                "question": "Q?",
                "options": ["a", "b"],
                "correct_answer": "c",
            }
        ]
    )
    assert len(result) == 1
    assert result[0].correct_answer == "c"


def test_parse_generated_exercises_keeps_valid_when_sibling_invalid():
    result = parse_generated_exercises(
        [
            {
                "type": "multiple_choice",
                "question": "She ___ to school.",
                "options": ["walks", "walk"],
                "correct_answer": "walks",
            },
            {
                "type": "fill_blank",
                "sentence": "She is best student.",
                "answer": "an",
            },
        ]
    )
    assert len(result) == 1
    assert result[0].type == "multiple_choice"


def test_parse_generated_exercises_normalizes_blank_marker():
    result = parse_generated_exercises(
        [
            {
                "type": "fill_blank",
                "sentence": "He plays tennis on weekends.",
                "answer": "plays",
            }
        ]
    )
    assert len(result) == 1
    assert "___" in result[0].sentence


def test_parse_generated_exercises_normalizes_mcq_case():
    result = parse_generated_exercises(
        [
            {
                "type": "multiple_choice",
                "question": "Q?",
                "options": ["chose", "chosen", "choosed"],
                "correct_answer": "Chose",
            }
        ]
    )
    assert len(result) == 1
    assert result[0].correct_answer == "chose"


def test_build_exercise_payload_includes_dev_answer_key(monkeypatch):
    monkeypatch.setenv("LEXORY_EXPOSE_EXERCISE_ANSWERS", "1")
    payload = build_exercise_payload(
        exercise_id="ex-1",
        mistake_type="sva",
        source_sentence="She walk.",
        payload={
            "type": "multiple_choice",
            "question": "She ___?",
            "options": ["walks", "walk"],
        },
        answer_key={
            "type": "multiple_choice",
            "correct_option": "walks",
        },
    )
    assert payload.dev_answer_key is not None
    assert payload.dev_answer_key.correct_option == "walks"


def test_build_exercise_payload_hides_dev_answer_key_by_default(monkeypatch):
    monkeypatch.delenv("LEXORY_EXPOSE_EXERCISE_ANSWERS", raising=False)
    payload = build_exercise_payload(
        exercise_id="ex-1",
        mistake_type="sva",
        source_sentence="She walk.",
        payload={
            "type": "multiple_choice",
            "question": "She ___?",
            "options": ["walks", "walk"],
        },
        answer_key={
            "type": "multiple_choice",
            "correct_option": "walks",
        },
    )
    assert payload.dev_answer_key is None
