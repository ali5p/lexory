"""Tests for LanguageTool pipeline."""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from rag.embedder import Embedder
from rag.pipelines.languagetool_pipeline import process_text


class MockLanguageTool:
    """Mock LanguageTool for testing."""

    def __init__(self, lang="en-US"):
        self.lang = lang

    def check(self, text):
        """Return mock matches based on text."""
        matches = []
        if "go to" in text.lower() and "he" in text.lower():
            match = Mock()
            match.rule_id = "SUBJECT_VERB_AGREEMENT"
            match.message = "Did you mean 'goes'?"
            matches.append(match)
        return matches


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic 384-dim vectors."""
    embedder = Mock(spec=Embedder)
    embedder.embed_single = Mock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def mock_lt_tool():
    """Mock LanguageTool instance."""
    return MockLanguageTool()


def test_process_text_no_mistakes(mock_embedder):
    """Text with no mistakes returns empty list (use mock LT to avoid real API)."""
    events = process_text(
        text="He goes to school.",
        user_id="user-123",
        user_text_id="text-001",
        session_id="session-456",
        detected_at=datetime.now(timezone.utc),
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=MockLanguageTool(),
    )
    assert len(events) == 0


def test_process_text_with_mistake(mock_embedder, mock_lt_tool):
    """Test A: Single mistake creates one event with correct structure."""
    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="He go to school.",
        user_id="user-123",
        user_text_id="text-001",
        session_id="session-456",
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt_tool,
    )

    assert len(events) == 1
    event = events[0]

    assert "mistake_id" in event
    assert event["user_id"] == "user-123"
    assert event["user_text_id"] == "text-001"
    assert event["session_id"] == "session-456"
    assert event["rule_id"] == "SUBJECT_VERB_AGREEMENT"
    assert event["mistake_type"] == "subject_verb_agreement"
    assert event["source"] == "raw_text"
    assert event["weight"] == 1.0
    assert event["detected_at"] == test_detected_at.isoformat()
    assert len(event["context_vector"]) == 384
    assert len(event["mistake_logic_vector"]) == 64
    assert event["text"] == "He go to school."


def test_process_text_exercise_weight(mock_embedder, mock_lt_tool):
    """Test D: Exercise attempts have weight 0.5."""
    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="He go to school.",
        user_id="user-123",
        user_text_id="exercise-001",
        session_id=None,
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="exercise_attempt",
        lt_tool=mock_lt_tool,
    )

    assert len(events) == 1
    assert events[0]["weight"] == 0.5
    assert events[0]["source"] == "exercise_attempt"
    assert events[0]["user_text_id"] == "exercise-001"
    assert events[0]["detected_at"] == test_detected_at.isoformat()


def test_process_text_multiple_sentences(mock_embedder, mock_lt_tool):
    """Test that multiple sentences are processed separately."""
    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="He go to school. She go to work.",
        user_id="user-123",
        user_text_id="text-002",
        session_id=None,
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt_tool,
    )

    assert len(events) == 2
    assert all(e["rule_id"] == "SUBJECT_VERB_AGREEMENT" for e in events)
    assert all(e["detected_at"] == test_detected_at.isoformat() for e in events)
    assert all(e["user_text_id"] == "text-002" for e in events)


def test_process_text_bracketed_rule_id_maps_via_normalize(mock_embedder):
    """Mapping keys are base ids; LT may return TOT_HE[1]. Stored rule_id stays bracketed."""
    mock_lt = MockLanguageTool()

    def check_bracket(text):
        match = Mock()
        match.rule_id = "TOT_HE[1]"
        match.message = ""
        return [match]

    mock_lt.check = check_bracket
    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="Some text.",
        user_id="user-123",
        user_text_id="text-bracket",
        session_id=None,
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt,
    )
    assert len(events) == 1
    assert events[0]["mistake_type"] == "unmapped"
    assert events[0]["rule_id"] == "TOT_HE[1]"


def test_process_text_rule_id_mapping_fallback_lowercase_json_key(mock_embedder):
    """LT may return UPPER_SNAKE while the only mapping key is lower_snake (e.g. in_excess_of)."""
    mock_lt = MockLanguageTool()

    def check_in_excess(text):
        match = Mock()
        match.rule_id = "IN_EXCESS_OF"
        match.message = ""
        return [match]

    mock_lt.check = check_in_excess
    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="Some text.",
        user_id="user-123",
        user_text_id="text-in-excess",
        session_id=None,
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt,
    )
    assert len(events) == 1
    assert events[0]["mistake_type"] == "style"
    assert events[0]["rule_id"] == "IN_EXCESS_OF"


def test_process_text_unknown_rule_id(mock_embedder):
    """Unknown rule IDs map to mistake_type unlisted."""
    mock_lt = MockLanguageTool()

    def check_unknown(text):
        match = Mock()
        match.rule_id = "UNKNOWN_RULE"
        match.message = ""
        return [match]

    mock_lt.check = check_unknown

    test_detected_at = datetime.now(timezone.utc)
    events = process_text(
        text="Some text.",
        user_id="user-123",
        user_text_id="text-003",
        session_id=None,
        detected_at=test_detected_at,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt,
    )

    assert len(events) == 1
    assert events[0]["mistake_type"] == "unlisted"
    assert events[0]["user_text_id"] == "text-003"
    assert events[0]["detected_at"] == test_detected_at.isoformat()
