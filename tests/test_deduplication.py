"""Tests for mistake event deduplication workflow."""
import pytest
from unittest.mock import Mock, MagicMock

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


def test_deduplication_stage1_no_existing_examples(rag_service, mock_qdrant):
    """Test A: No existing examples creates both example and occurrence."""
    event = {
        "mistake_id": "mistake-123",
        "user_id": "user-123",
        "session_id": "session-456",
        "rule_id": "SUBJECT_VERB_AGREEMENT",
        "mistake_type": "subject_verb_agreement",
        "text": "He go to school.",
        "source": "raw_text",
        "weight": 1.0,
        "timestamp": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.1] * 384,
        "extra": {},
    }
    
    # Stage 1: No existing examples
    mock_qdrant.search.return_value = []
    
    example_point, occurrence_point = rag_service._ingest_mistake_event(
        event=event,
        user_text_id="text-123",
    )
    
    assert example_point is not None
    assert occurrence_point is not None
    assert example_point["id"] != event["mistake_id"]  # New example ID
    assert occurrence_point["id"] == event["mistake_id"]  # Occurrence uses mistake_id
    assert "vectors" in example_point
    assert "mistake_logic" in example_point["vectors"]
    assert "context" in example_point["vectors"]
    assert "vectors" in occurrence_point
    assert "mistake_logic" in occurrence_point["vectors"]
    assert "context" not in occurrence_point["vectors"]
    
    # Verify SQL insertion
    assert len(rag_service.occurrence_store.occurrences) == 1
    occ = rag_service.occurrence_store.occurrences[0]
    assert occ["mistake_id"] == event["mistake_id"]
    assert occ["mistake_type"] == "subject_verb_agreement"


def test_deduplication_stage2_high_similarity(rag_service, mock_qdrant):
    """Test B: High similarity (>0.9) creates only occurrence."""
    event = {
        "mistake_id": "mistake-456",
        "user_id": "user-123",
        "session_id": "session-456",
        "rule_id": "SUBJECT_VERB_AGREEMENT",
        "mistake_type": "subject_verb_agreement",
        "text": "He go to school.",
        "source": "raw_text",
        "weight": 1.0,
        "timestamp": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.1] * 384,
        "extra": {},
    }
    
    # Stage 1: Example exists
    mock_qdrant.search.side_effect = [
        [{"id": "example-1", "score": 0.5, "payload": {"mistake_type": "subject_verb_agreement"}}],
        [{"id": "example-1", "score": 0.99, "payload": {"mistake_type": "subject_verb_agreement"}}],  # Stage 2: high similarity
    ]
    
    example_point, occurrence_point = rag_service._ingest_mistake_event(
        event=event,
        user_text_id="text-456",
    )
    
    assert example_point is None  # No new example
    assert occurrence_point is not None  # Only occurrence
    assert occurrence_point["id"] == event["mistake_id"]


def test_deduplication_stage2_low_similarity(rag_service, mock_qdrant):
    """Test C: Low similarity (<=0.9) creates new example + occurrence."""
    event = {
        "mistake_id": "mistake-789",
        "user_id": "user-123",
        "session_id": "session-456",
        "rule_id": "SUBJECT_VERB_AGREEMENT",
        "mistake_type": "subject_verb_agreement",
        "text": "They goes to park.",
        "source": "raw_text",
        "weight": 1.0,
        "timestamp": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.2] * 384,  # Different context vector
        "extra": {},
    }
    
    # Stage 1: Example exists, Stage 2: Low similarity
    mock_qdrant.search.side_effect = [
        [{"id": "example-1", "score": 0.5, "payload": {"mistake_type": "subject_verb_agreement"}}],
        [{"id": "example-1", "score": 0.95, "payload": {"mistake_type": "subject_verb_agreement"}}],  # Below threshold
    ]
    
    example_point, occurrence_point = rag_service._ingest_mistake_event(
        event=event,
        user_text_id="text-789",
    )
    
    assert example_point is not None  # New example created
    assert occurrence_point is not None  # Occurrence also created
    assert example_point["id"] != event["mistake_id"]
    assert occurrence_point["id"] == event["mistake_id"]
