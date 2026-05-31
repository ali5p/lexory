"""Shared lesson artifact schema for SQL, Qdrant, and batch."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from core.models import DetectedMistakeExample, LessonResponse


class LessonArtifactRecord(BaseModel):
    """
    Canonical lesson artifact fields.
    SQL stores the full row.
    """

    artifact_id: str
    user_id: str
    session_id: str = ""
    mistake_id: Optional[str] = None
    rule_id: str = ""
    mistake_type: str = ""
    mistake_context: str = ""
    topic: str = ""
    explanation: str = ""
    exercises: list[str] = Field(default_factory=list)
    approach_type: str = ""
    created_at: str = ""

    @classmethod
    def for_lesson(
        cls,
        *,
        artifact_id: str,
        lesson: LessonResponse,
        user_id: str,
        session_id: Optional[str],
        primary_mistake: Optional[DetectedMistakeExample],
        created_at: datetime,
    ) -> LessonArtifactRecord:
        mistake_context = ""
        if primary_mistake and primary_mistake.examples:
            mistake_context = primary_mistake.examples[0] or ""
        return cls(
            artifact_id=artifact_id,
            user_id=user_id,
            session_id=session_id or "",
            mistake_id=primary_mistake.mistake_id if primary_mistake else None,
            rule_id=primary_mistake.rule_id if primary_mistake else "",
            mistake_type=primary_mistake.mistake_type if primary_mistake else "",
            mistake_context=mistake_context,
            topic=lesson.topic,
            explanation=lesson.explanation,
            exercises=list(lesson.exercises),
            approach_type=lesson.approach_type,
            created_at=created_at.isoformat(),
        )

    def sql_row(self) -> dict:
        return self.model_dump(
            include={
                "artifact_id",
                "user_id",
                "session_id",
                "topic",
                "explanation",
                "exercises",
                "approach_type",
                "mistake_type",
                "created_at",
            }
        )

    def qdrant_payload(self) -> dict:
        return self.model_dump(exclude={"exercises"})
