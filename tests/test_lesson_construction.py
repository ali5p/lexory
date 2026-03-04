"""Minimal integration test for lesson construction structural contract."""
import pytest
from unittest.mock import Mock

from core.models import ContextAssembly
from rag.service import RAGService
from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


@pytest.fixture
def mock_qdrant():
    """Mock QdrantStore."""
    qdrant = Mock(spec=QdrantStore)
    qdrant.search = Mock(return_value=[])
    qdrant.upsert = Mock()
    return qdrant


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = Mock(spec=Embedder)
    embedder.embed_single = Mock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def rag_service(mock_qdrant, mock_embedder):
    """Create RAGService with mocked dependencies."""
    return RAGService(mock_qdrant, mock_embedder)


def test_construct_lesson_stub_returns_valid_structure(
    rag_service: RAGService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate structural contract of lesson construction with stub handler."""
    monkeypatch.setenv("GENERATOR_MODE", "stub")

    context = ContextAssembly(
        detected_mistake_examples=[{"mistake_type": "SUBJECT_VERB_AGREEMENT"}],
        recently_used_explanations=[],
        long_term_dynamics=[],
    )

    lesson = rag_service._construct_lesson(context)

    assert lesson.topic is not None
    assert lesson.explanation.startswith("[STUB]")
    assert isinstance(lesson.exercises, list)

    # Empty context must not crash (structural contract)
    empty_context = ContextAssembly(
        detected_mistake_examples=[],
        recently_used_explanations=[],
        long_term_dynamics=[],
    )
    empty_lesson = rag_service._construct_lesson(empty_context)
    assert empty_lesson.topic is not None
    assert isinstance(empty_lesson.exercises, list)
