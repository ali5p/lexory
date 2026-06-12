"""Minimal integration test for lesson construction structural contract."""
import pytest
from unittest.mock import AsyncMock, Mock

from core.models import ContextAssembly, DetectedMistakeExample
from rag.service import RAGService
from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


class _FakeLLM:
    """Returns a fixed, schema-valid lesson JSON. No network."""

    def chat(self, messages, *, temperature: float = 0.0, json_schema=None) -> str:
        return (
            '{"topic": "Subject-verb agreement", '
            '"lesson": "Use the -s form with he/she/it.", '
            '"exercise": "Correct: She walk to school."}'
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
    factory = Mock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx
    return factory


@pytest.fixture
def rag_service(mock_qdrant, mock_embedder, mock_session_factory):
    service = RAGService(mock_qdrant, mock_embedder, mock_session_factory)
    # Inject a deterministic LLM so tests never hit the network.
    fake = _FakeLLM()
    for handler in service._approach_registry.values():
        handler.llm = fake
    return service


def test_construct_lesson_returns_valid_structure(rag_service: RAGService) -> None:
    """Validate structural contract of lesson construction (LLM path, mocked LLM)."""
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(mistake_type="SUBJECT_VERB_AGREEMENT")
        ],
        long_term_dynamics=[],
    )

    lesson = rag_service._construct_lesson(context, "rule_based")

    assert lesson.topic == "Subject-verb agreement"
    assert lesson.explanation
    assert lesson.exercises == ["Correct: She walk to school."]
    # Stored approach_type is the selected teaching approach, not generation status.
    assert lesson.approach_type == "rule_based"

    # Empty context must not crash (structural contract)
    empty_context = ContextAssembly(
        detected_mistake_examples=[],
        long_term_dynamics=[],
    )
    empty_lesson = rag_service._construct_lesson(empty_context, "rule_based")
    assert empty_lesson.topic is not None
    assert isinstance(empty_lesson.exercises, list)


def test_construct_lesson_example_based_records_selected_approach(
    rag_service: RAGService,
) -> None:
    """example_based runs through the shared LLM path and stores its own approach_type."""
    context = ContextAssembly(
        detected_mistake_examples=[
            DetectedMistakeExample(
                mistake_type="SUBJECT_VERB_AGREEMENT",
                examples=["My brother live in Berlin."],
                rule_message="Singular subject needs -s.",
            )
        ],
        long_term_dynamics=[],
        similar_past_examples=[
            {"text": "He walk to school.", "rule_message": "x"},
            {"text": "She play tennis.", "rule_message": "y"},
        ],
    )

    lesson = rag_service._construct_lesson(context, "example_based")

    assert lesson.approach_type == "example_based"
    assert lesson.explanation
    assert isinstance(lesson.exercises, list)
