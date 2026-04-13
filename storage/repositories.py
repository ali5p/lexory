"""Thin async repository functions for PostgreSQL persistence."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import (
    ExampleImprint,
    ExerciseAttempt,
    LessonArtifact,
    MistakeOccurrence,
)


async def insert_occurrence(session: AsyncSession, data: dict) -> None:
    row = MistakeOccurrence(
        mistake_id=data["mistake_id"],
        user_id=data["user_id"],
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


async def insert_imprint(session: AsyncSession, payload: dict) -> None:
    required = ("mistake_id", "user_id", "session_id", "detected_at")
    for k in required:
        if k not in payload:
            raise ValueError(f"Example imprint requires '{k}' in payload")
    row = ExampleImprint(
        mistake_id=payload["mistake_id"],
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        detected_at=payload["detected_at"],
        user_text_id=payload.get("user_text_id"),
        rule_id=payload.get("rule_id"),
        mistake_type=payload.get("mistake_type"),
    )
    session.add(row)
    await session.flush()


async def get_most_recent_imprint_mistake_id(
    session: AsyncSession, user_id: str
) -> Optional[str]:
    stmt = (
        select(ExampleImprint.mistake_id)
        .where(ExampleImprint.user_id == user_id)
        .order_by(ExampleImprint.detected_at.desc())
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
