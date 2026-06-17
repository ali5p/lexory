"""Tests for lesson artifact selection metadata persistence."""
from datetime import datetime, timezone

from core.lesson_artifact import LessonArtifactRecord
from core.models import DetectedMistakeExample, LessonResponse


def test_lesson_artifact_record_sql_row_includes_selection_metadata():
    record = LessonArtifactRecord.for_lesson(
        artifact_id="artifact-1",
        lesson=LessonResponse(
            topic="Articles",
            explanation="Use a before consonant sounds.",
            exercises=["Fill in: ___ apple."],
            approach_type="rule_based",
        ),
        user_id="user-1",
        session_id="session-1",
        primary_mistake=DetectedMistakeExample(
            mistake_id="m-1",
            rule_id="ARTICLE",
            mistake_type="articles",
            examples=["I ate apple."],
        ),
        created_at=datetime(2026, 6, 14, tzinfo=timezone.utc),
        selection_index=5,
        is_contrast_lesson=True,
    )

    row = record.sql_row()

    assert row["selection_index"] == 5
    assert row["is_contrast_lesson"] is True
