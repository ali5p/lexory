"""Tests for mistake event deduplication workflow."""
import uuid

import pytest
from unittest.mock import Mock, patch

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
        "user_text_id": "text-123",
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
    fixed_example_id = uuid.UUID("00000000-0000-4000-8000-000000000001")

    with patch("rag.service.uuid.uuid4", return_value=fixed_example_id):
        example_point, occurrence_point = rag_service._ingest_mistake_event(
            event=event,
            user_text_id="text-123",
        )
    
    assert example_point is not None
    assert occurrence_point is not None
    assert example_point["id"] == event["mistake_id"]
    assert occurrence_point["id"] == event["mistake_id"]
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
    assert example_point["payload"]["mistake_id"] == event["mistake_id"]
    assert example_point["payload"]["example_id"] == str(fixed_example_id)
    assert occ["example_id"] == str(fixed_example_id)


def test_deduplication_stage2_high_similarity(rag_service, mock_qdrant):
    """Test B: High similarity (>0.9) creates only occurrence."""
    event = {
        "mistake_id": "mistake-456",
        "user_id": "user-123",
        "user_text_id": "text-456",
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
    occ = rag_service.occurrence_store.occurrences[0]
    assert "example_id" not in occ


def test_deduplication_stage2_low_similarity(rag_service, mock_qdrant):
    """Test C: Low similarity (<=0.9) creates new example + occurrence."""
    event = {
        "mistake_id": "mistake-789",
        "user_id": "user-123",
        "user_text_id": "text-789",
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
    
    # Stage 1: Example exists, Stage 2: Low similarity (score must be <= semantic_dedup_threshold 0.9)
    mock_qdrant.search.side_effect = [
        [{"id": "example-1", "score": 0.5, "payload": {"mistake_type": "subject_verb_agreement"}}],
        [{"id": "example-1", "score": 0.85, "payload": {"mistake_type": "subject_verb_agreement"}}],
    ]
    
    fixed_example_id = uuid.UUID("00000000-0000-4000-8000-000000000002")
    with patch("rag.service.uuid.uuid4", return_value=fixed_example_id):
        example_point, occurrence_point = rag_service._ingest_mistake_event(
            event=event,
            user_text_id="text-789",
        )

    assert example_point is not None  # New example created
    assert occurrence_point is not None  # Occurrence also created
    assert example_point["id"] == event["mistake_id"]
    assert occurrence_point["id"] == event["mistake_id"]
    occ = rag_service.occurrence_store.occurrences[0]
    assert example_point["payload"]["example_id"] == str(fixed_example_id)
    assert occ["example_id"] == str(fixed_example_id)
