"""Tests for exercise generator prompt wiring."""

from core.models import ContextAssembly, DetectedMistakeExample
from rag.exercise_generator import ExerciseGenerator, _user_message


class _CapturingLLM:
    def __init__(self):
        self.last_messages = []

    def chat(self, messages, *, temperature=0.0, json_schema=None):
        self.last_messages = list(messages)
        return (
            '{"exercises": ['
            '{"type": "multiple_choice", "question": "Q1", "options": ["a", "b"], '
            '"correct_answer": "a"}'
            "]}"
        )


def test_user_message_includes_required_type():
    msg = _user_message(
        rule_message="Use -s with he/she/it.",
        example_sentence="She walk.",
        mistake_type="subject_verb_agreement",
        rule_id="SVA",
        topic="SVA",
        explanation="Add -s.",
        exercise_type="fill_blank",
    )
    assert "Required exercise type:\nfill_blank" in msg
    assert "She walk." in msg


def test_generate_returns_single_exercise_of_requested_type():
    llm = _CapturingLLM()
    gen = ExerciseGenerator(llm=llm)
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="sva",
                examples=["A sentence."],
                rule_message="rule",
                rule_id="R1",
            )
        ]
    )
    pairs = gen.generate(
        context,
        topic="T",
        explanation="E",
        exercise_type="multiple_choice",
    )
    assert len(pairs) == 1
    payload, answer_key = pairs[0]
    assert payload["type"] == "multiple_choice"
    assert answer_key["type"] == "multiple_choice"
