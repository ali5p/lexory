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
            '{"type": "multiple_choice", "sentence": "She ___ to school.", "options": ["a", "b"], '
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


def test_generate_retries_after_validation_failure():
    calls = {"n": 0}

    class _RetryLLM:
        def chat(self, messages, *, temperature=0.0, json_schema=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    '{"exercises": [{"type": "multiple_choice", '
                    '"sentence": "Choose the correct sentence.", '
                    '"options": ["She is looking.", "She looking."], '
                    '"correct_answer": "She is looking."}]}'
                )
            return (
                '{"exercises": [{"type": "multiple_choice", '
                '"sentence": "She ___ looking.", "options": ["is", "was"], '
                '"correct_answer": "is"}]}'
            )

    gen = ExerciseGenerator(_RetryLLM())
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="pronouns",
                examples=["She looking everywhere."],
                rule_message="missing auxiliary",
                rule_id="PRP_VBG",
            )
        ]
    )
    pairs = gen.generate(
        context,
        topic="Progressive tense",
        explanation="Use is/am/are + -ing.",
        exercise_type="multiple_choice",
    )
    assert calls["n"] == 2
    assert len(pairs) == 1


def test_generate_falls_back_to_alternate_type_after_repeated_failure():
    calls = {"n": 0}

    class _FallbackLLM:
        def chat(self, messages, *, temperature=0.0, json_schema=None):
            calls["n"] += 1
            system = messages[0]["content"] if messages else ""
            if "fill_blank" in system:
                return (
                    '{"exercises": [{"type": "fill_blank", '
                    '"sentence": "The ship had ___ in the ocean.", "answer": "sunk"}]}'
                )
            return (
                '{"exercises": [{"type": "multiple_choice", '
                '"sentence": "Choose one.", "options": ["a", "b"], '
                '"correct_answer": "c"}]}'
            )

    gen = ExerciseGenerator(_FallbackLLM())
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="subject_verb_agreement",
                examples=["The ship had sank in the ocean."],
                rule_message="past participle",
                rule_id="HAVE_PART_AGREEMENT",
            )
        ]
    )
    pairs = gen.generate(
        context,
        topic="SVA",
        explanation="Use sunk.",
        exercise_type="multiple_choice",
    )
    assert len(pairs) == 1
    assert pairs[0][0]["type"] == "fill_blank"
    assert calls["n"] >= 3


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
