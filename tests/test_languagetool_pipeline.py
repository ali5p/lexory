"""Tests for LanguageTool pipeline."""
import pytest
from unittest.mock import Mock, patch

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
            # Simulate subject-verb agreement error
            match = Mock()
            match.ruleId = "SUBJECT_VERB_AGREEMENT"
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
    """Test A: Text with no mistakes returns empty list."""
    with patch("rag.pipelines.languagetool_pipeline.LanguageTool", MockLanguageTool):
        events = process_text(
            text="He goes to school.",
            user_id="user-123",
            session_id="session-456",
            embedder=mock_embedder,
            source="raw_text",
        )
        assert len(events) == 0


def test_process_text_with_mistake(mock_embedder, mock_lt_tool):
    """Test A: Single mistake creates one event with correct structure."""
    events = process_text(
        text="He go to school.",
        user_id="user-123",
        session_id="session-456",
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt_tool,
    )
    
    assert len(events) == 1
    event = events[0]
    
    assert "mistake_id" in event
    assert event["user_id"] == "user-123"
    assert event["session_id"] == "session-456"
    assert event["rule_id"] == "SUBJECT_VERB_AGREEMENT"
    assert event["mistake_type"] == "subject_verb_agreement"
    assert event["source"] == "raw_text"
    assert event["weight"] == 1.0
    assert len(event["context_vector"]) == 384
    assert len(event["mistake_logic_vector"]) == 64
    assert event["text"] == "He go to school."


def test_process_text_exercise_weight(mock_embedder, mock_lt_tool):
    """Test D: Exercise attempts have weight 0.5."""
    events = process_text(
        text="He go to school.",
        user_id="user-123",
        session_id=None,
        embedder=mock_embedder,
        source="exercise_attempt",
        lt_tool=mock_lt_tool,
    )
    
    assert len(events) == 1
    assert events[0]["weight"] == 0.5
    assert events[0]["source"] == "exercise_attempt"


def test_process_text_multiple_sentences(mock_embedder, mock_lt_tool):
    """Test that multiple sentences are processed separately."""
    events = process_text(
        text="He go to school. She go to work.",
        user_id="user-123",
        session_id=None,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt_tool,
    )
    
    # Should detect mistake in both sentences
    assert len(events) == 2
    assert all(e["rule_id"] == "SUBJECT_VERB_AGREEMENT" for e in events)


def test_process_text_unknown_rule_id(mock_embedder):
    """Test that unknown rule IDs map to 'other'."""
    mock_lt = MockLanguageTool()
    
    # Override check to return unknown rule
    def check_unknown(text):
        match = Mock()
        match.ruleId = "UNKNOWN_RULE"
        return [match]
    
    mock_lt.check = check_unknown
    
    events = process_text(
        text="Some text.",
        user_id="user-123",
        session_id=None,
        embedder=mock_embedder,
        source="raw_text",
        lt_tool=mock_lt,
    )
    
    assert len(events) == 1
    assert events[0]["mistake_type"] == "other"
