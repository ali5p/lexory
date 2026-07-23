"""Minimal integration test for lesson construction structural contract."""
import pytest
from unittest.mock import AsyncMock, Mock

from core.models import ContextAssembly, DetectedMistakeExample
from rag.service import RAGService
from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


class _FakeLessonLLM:
    """Returns fixed lesson JSON and type-specific exercise JSON."""

    def chat(self, messages, *, temperature: float = 0.0, json_schema=None) -> str:
        system = messages[0]["content"] if messages else ""
        if "drill writer" in system.lower() or "grammar drill" in system.lower():
            if "fill_blank" in system.lower():
                return (
                    '{"exercises": [{"type": "fill_blank", '
                    '"sentence": "He ___ tennis on weekends.", "answer": "plays"}]}'
                )
            return (
                '{"exercises": [{"type": "multiple_choice", '
                '"sentence": "She ___ to school.", '
                '"options": ["walks", "walk"], "correct_answer": "walks"}]}'
            )
        return (
            '{"topic": "Subject-verb agreement", '
            '"lesson": "Use the -s form with he/she/it."}'
        )


@pytest.fixture
def mock_qdrant():
    qdrant = Mock(spec=QdrantStore)
    qdrant.search = Mock(return_value=[])
    qdrant.upsert = Mock()
    return qdrant


@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=Embedder)
    embedder.embed_single = Mock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def mock_session_factory():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory = Mock()
    factory.return_value = ctx
    return factory


@pytest.fixture
def rag_service(mock_qdrant, mock_embedder, mock_session_factory):
    service = RAGService(mock_qdrant, mock_embedder, mock_session_factory)
    fake = _FakeLessonLLM()
    service._exercise_generator.llm = fake
    for handler in service._approach_registry.values():
        handler.llm = fake
    return service


def test_construct_lesson_returns_valid_structure(rag_service: RAGService) -> None:
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(mistake_type="SUBJECT_VERB_AGREEMENT")
        ],
    )

    lesson = rag_service._construct_lesson(context, "rule_based")

    assert lesson.topic == "Subject-verb agreement"
    assert lesson.explanation
    assert lesson.approach_type == "rule_based"


@pytest.mark.asyncio
async def test_generate_and_persist_exercises_one_per_item(
    rag_service: RAGService,
) -> None:
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="subject_verb_agreement",
                examples=["She walk to school."],
            )
        ],
    )
    lesson = rag_service._construct_lesson(context, "rule_based")

    mcq = await rag_service._generate_and_persist_exercises(
        artifact_id="artifact-1",
        context=context,
        lesson=lesson,
        selection_index=0,
    )
    assert len(mcq) == 1
    assert mcq[0].payload.type == "multiple_choice"

    fb = await rag_service._generate_and_persist_exercises(
        artifact_id="artifact-2",
        context=context,
        lesson=lesson,
        selection_index=2,
    )
    assert len(fb) == 1
    assert fb[0].payload.type == "fill_blank"


def test_construct_lesson_example_based_records_selected_approach(
    rag_service: RAGService,
) -> None:
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="SUBJECT_VERB_AGREEMENT",
                examples=["My brother live in Berlin."],
                rule_message="Singular subject needs -s.",
            )
        ],
        similar_past_examples=[
            {"text": "He walk to school.", "rule_message": "x"},
            {"text": "She play tennis.", "rule_message": "y"},
        ],
    )

    lesson = rag_service._construct_lesson(context, "example_based")

    assert lesson.approach_type == "example_based"
    assert lesson.explanation
