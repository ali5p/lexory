"""Thin async repository functions for PostgreSQL persistence."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import ExerciseAttempt, LessonArtifact, MistakeOccurrence


async def insert_occurrence(session: AsyncSession, data: dict) -> None:
    row = MistakeOccurrence(
        mistake_id=data["mistake_id"],
        user_id=data["user_id"],
        session_id=data.get("session_id", ""),
        user_text_id=data["user_text_id"],
        detected_at=data["detected_at"],
        source=data["source"],
        mistake_type=data["mistake_type"],
        rule_id=data["rule_id"],
        example_id=data.get("example_id"),
        lesson_artifact_id=data.get("lesson_artifact_id"),
    )
    session.add(row)
    await session.flush()


async def get_most_recent_occurrence_mistake_id(
    session: AsyncSession, user_id: str
) -> Optional[str]:
    """Latest mistake_id for this user among rows that have a mistake_examples Qdrant point (example_id set)."""
    stmt = (
        select(MistakeOccurrence.mistake_id)
        .where(
            MistakeOccurrence.user_id == user_id,
            MistakeOccurrence.example_id.isnot(None),
        )
        .order_by(MistakeOccurrence.detected_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.first()
    return row[0] if row else None


async def upsert_artifact(session: AsyncSession, data: dict) -> None:
    row = LessonArtifact(
        artifact_id=data["artifact_id"],
        user_id=data["user_id"],
        session_id=data.get("session_id", ""),
        content=data.get("content", ""),
        lesson_type=data.get("lesson_type", ""),
        approach_type=data.get("approach_type", ""),
        mistake_types_covered=data.get("mistake_types_covered", []),
        pedagogy_tags=data.get("pedagogy_tags", []),
        created_at=data.get("created_at", ""),
    )
    await session.merge(row)
    await session.flush()


async def insert_exercise_attempt(session: AsyncSession, data: dict) -> None:
    row = ExerciseAttempt(
        exercise_attempt_id=data["exercise_attempt_id"],
        lesson_artifact_id=data["lesson_artifact_id"],
        user_id=data["user_id"],
        attempt_timestamp=data["attempt_timestamp"],
    )
    session.add(row)
    await session.flush()
