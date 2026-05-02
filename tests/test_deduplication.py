"""Tests for mistake event deduplication workflow."""
import uuid

import pytest
from unittest.mock import AsyncMock, Mock, patch

from rag.service import RAGService
from rag.embedder import Embedder
from storage.models import MistakeOccurrence, UserScoringEvent
from vectorstore.qdrant_client import QdrantStore


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
def mock_session():
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


def _added_row(mock_session, model_type):
    for call in mock_session.add.call_args_list:
        row = call.args[0]
        if isinstance(row, model_type):
            return row
    raise AssertionError(f"{model_type.__name__} was not added")


@pytest.fixture
def mock_session_factory(mock_session):
    factory = Mock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx
    return factory


@pytest.fixture
def rag_service(mock_qdrant, mock_embedder, mock_session_factory):
    return RAGService(mock_qdrant, mock_embedder, mock_session_factory)


@pytest.mark.asyncio
async def test_deduplication_stage1_no_existing_examples(rag_service, mock_qdrant, mock_session):
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
        "detected_at": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.1] * 384,
        "extra": {},
    }

    mock_qdrant.search.return_value = []
    fixed_example_id = uuid.UUID("00000000-0000-4000-8000-000000000001")

    with patch("rag.service.uuid.uuid4", return_value=fixed_example_id):
        example_point, occurrence_point = await rag_service._ingest_mistake_event(
            session=mock_session,
            event=event,
            user_text_id="text-123",
        )

    assert example_point is not None
    assert occurrence_point is not None
    assert example_point["id"] == event["mistake_id"]
    assert occurrence_point["id"] == event["mistake_id"]
    assert "context" in example_point["vectors"]
    assert "context" not in occurrence_point["vectors"]
    assert example_point["payload"]["mistake_id"] == event["mistake_id"]
    assert example_point["payload"]["example_id"] == str(fixed_example_id)

    assert mock_session.add.call_count == 2
    row = _added_row(mock_session, MistakeOccurrence)
    assert row.mistake_id == event["mistake_id"]
    assert row.example_id == str(fixed_example_id)
    scoring_row = _added_row(mock_session, UserScoringEvent)
    assert scoring_row.delta == 1.0
    assert scoring_row.mistake_type == event["mistake_type"]


@pytest.mark.asyncio
async def test_deduplication_stage2_high_similarity(rag_service, mock_qdrant, mock_session):
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
        "detected_at": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.1] * 384,
        "extra": {},
    }

    mock_qdrant.search.side_effect = [
        [{"id": "example-1", "score": 0.5, "payload": {"mistake_type": "subject_verb_agreement"}}],
        [{"id": "example-1", "score": 0.99, "payload": {"mistake_type": "subject_verb_agreement"}}],
    ]

    example_point, occurrence_point = await rag_service._ingest_mistake_event(
        session=mock_session,
        event=event,
        user_text_id="text-456",
    )

    assert example_point is None
    assert occurrence_point is not None
    assert occurrence_point["id"] == event["mistake_id"]

    row = _added_row(mock_session, MistakeOccurrence)
    assert row.example_id is None
    scoring_row = _added_row(mock_session, UserScoringEvent)
    assert scoring_row.delta == 1.0


@pytest.mark.asyncio
async def test_deduplication_stage2_low_similarity(rag_service, mock_qdrant, mock_session):
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
        "detected_at": "2025-01-30T12:34:56Z",
        "mistake_logic_vector": [1.0] + [0.0] * 63,
        "context_vector": [0.2] * 384,
        "extra": {},
    }

    mock_qdrant.search.side_effect = [
        [{"id": "example-1", "score": 0.5, "payload": {"mistake_type": "subject_verb_agreement"}}],
        [{"id": "example-1", "score": 0.85, "payload": {"mistake_type": "subject_verb_agreement"}}],
    ]

    fixed_example_id = uuid.UUID("00000000-0000-4000-8000-000000000002")
    with patch("rag.service.uuid.uuid4", return_value=fixed_example_id):
        example_point, occurrence_point = await rag_service._ingest_mistake_event(
            session=mock_session,
            event=event,
            user_text_id="text-789",
        )

    assert example_point is not None
    assert occurrence_point is not None
    assert example_point["id"] == event["mistake_id"]
    assert example_point["payload"]["example_id"] == str(fixed_example_id)

    row = _added_row(mock_session, MistakeOccurrence)
    assert row.example_id == str(fixed_example_id)
    scoring_row = _added_row(mock_session, UserScoringEvent)
    assert scoring_row.delta == 1.0
