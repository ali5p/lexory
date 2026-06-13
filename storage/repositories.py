"""Thin async repository functions for PostgreSQL persistence."""

import uuid
from typing import Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import (
    ExerciseAttempt,
    LessonArtifact,
    MistakeOccurrence,
    UserScoringEvent,
)


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
        topic=data.get("topic", ""),
        explanation=data.get("explanation", ""),
        exercises=data.get("exercises", []),
        approach_type=data.get("approach_type", ""),
        mistake_type=data.get("mistake_type", ""),
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


async def get_lesson_artifact_by_id(
    session: AsyncSession, artifact_id: str
) -> Optional[LessonArtifact]:
    stmt = select(LessonArtifact).where(LessonArtifact.artifact_id == artifact_id)
    r = await session.execute(stmt)
    return r.scalar_one_or_none()


async def insert_user_scoring_event(session: AsyncSession, data: dict) -> None:
    row = UserScoringEvent(
        id=data.get("id") or str(uuid.uuid4()),
        user_id=data["user_id"],
        rule_id=data.get("rule_id"),
        mistake_type=data["mistake_type"],
        mistake_id=data.get("mistake_id"),
        session_or_exercise_id=data["session_or_exercise_id"],
        occurred_at=data["occurred_at"],
        delta=float(data["delta"]),
    )
    session.add(row)
    await session.flush()


async def has_any_user_scoring_event(
    session: AsyncSession, user_id: str
) -> bool:
    stmt = (
        select(UserScoringEvent.id)
        .where(UserScoringEvent.user_id == user_id)
        .limit(1)
    )
    r = await session.execute(stmt)
    return r.first() is not None


async def has_nonzero_user_scoring_event(
    session: AsyncSession, user_id: str
) -> bool:
    """At least one row where delta is not 0 (graded activity)."""
    stmt = (
        select(UserScoringEvent.id)
        .where(
            UserScoringEvent.user_id == user_id,
            UserScoringEvent.delta != 0.0,
        )
        .limit(1)
    )
    r = await session.execute(stmt)
    return r.first() is not None


async def top_mistake_types_by_clamped_score(
    session: AsyncSession, user_id: str, k: int = 2
) -> Sequence[str]:
    """
    mistake_types with positive clamped total, ordered by score desc,
    then latest occurred_at (tie-break) desc.
    """
    agg = (
        select(
            UserScoringEvent.mistake_type.label("mt"),
            func.greatest(0, func.sum(UserScoringEvent.delta)).label("score"),
            func.max(UserScoringEvent.occurred_at).label("last_at"),
        )
        .where(UserScoringEvent.user_id == user_id)
        .group_by(UserScoringEvent.mistake_type)
        .having(func.greatest(0, func.sum(UserScoringEvent.delta)) > 0)
        .subquery()
    )
    stmt = (
        select(agg.c.mt).order_by(agg.c.score.desc(), agg.c.last_at.desc()).limit(k)
    )
    r = await session.execute(stmt)
    return [row[0] for row in r.all()]


async def clamped_scores_by_mistake_type(
    session: AsyncSession,
    user_id: str,
    mistake_types: Sequence[str],
) -> dict[str, tuple[float, str]]:
    """
    Scores for the requested mistake_types only.
    Returns mistake_type -> (clamped_score, latest_occurred_at).
    """
    unique_types = sorted({mt for mt in mistake_types if mt})
    if not unique_types:
        return {}

    stmt = (
        select(
            UserScoringEvent.mistake_type,
            func.greatest(0, func.sum(UserScoringEvent.delta)).label("score"),
            func.max(UserScoringEvent.occurred_at).label("last_at"),
        )
        .where(
            UserScoringEvent.user_id == user_id,
            UserScoringEvent.mistake_type.in_(unique_types),
        )
        .group_by(UserScoringEvent.mistake_type)
    )
    r = await session.execute(stmt)
    return {
        row[0]: (float(row[1] or 0), str(row[2] or ""))
        for row in r.all()
    }


async def clamped_scores_by_mistake_type_and_rule(
    session: AsyncSession,
    user_id: str,
    mistake_types: Sequence[str],
) -> dict[tuple[str, str], tuple[float, str]]:
    """
    Scores grouped by mistake_type + rule_id. Used when a placeholder mistake_type
    (e.g. "unmapped") is too broad to be a lesson target by itself.
    """
    unique_types = sorted({mt for mt in mistake_types if mt})
    if not unique_types:
        return {}

    stmt = (
        select(
            UserScoringEvent.mistake_type,
            UserScoringEvent.rule_id,
            func.greatest(0, func.sum(UserScoringEvent.delta)).label("score"),
            func.max(UserScoringEvent.occurred_at).label("last_at"),
        )
        .where(
            UserScoringEvent.user_id == user_id,
            UserScoringEvent.mistake_type.in_(unique_types),
        )
        .group_by(UserScoringEvent.mistake_type, UserScoringEvent.rule_id)
    )
    r = await session.execute(stmt)
    return {
        (str(row[0] or ""), str(row[1] or "")): (
            float(row[2] or 0),
            str(row[3] or ""),
        )
        for row in r.all()
    }


async def has_positive_clamped_mistake_type(
    session: AsyncSession, user_id: str
) -> bool:
    """True if any mistake_type has GREATEST(0, SUM(delta)) > 0 (same as top-1 non-empty)."""
    top = await top_mistake_types_by_clamped_score(session, user_id, 1)
    return bool(top)


async def count_examples_by_mistake_type(
    session: AsyncSession, user_id: str, mistake_type: str
) -> int:
    """Number of stored mistake_examples (example_id-bearing occurrences) the user
    has for this mistake_type. Drives the approach-selection example gate."""
    if not mistake_type:
        return 0
    stmt = (
        select(func.count())
        .select_from(MistakeOccurrence)
        .where(
            MistakeOccurrence.user_id == user_id,
            MistakeOccurrence.mistake_type == mistake_type,
            MistakeOccurrence.example_id.isnot(None),
        )
    )
    r = await session.execute(stmt)
    return int(r.scalar_one() or 0)


async def count_lessons_by_mistake_type(
    session: AsyncSession, user_id: str, mistake_type: str
) -> int:
    """Number of prior lesson_artifacts generated for this (user, mistake_type).
    Used as the per-type selection index (rotation / exploration cadence)."""
    if not mistake_type:
        return 0
    stmt = (
        select(func.count())
        .select_from(LessonArtifact)
        .where(
            LessonArtifact.user_id == user_id,
            LessonArtifact.mistake_type == mistake_type,
        )
    )
    r = await session.execute(stmt)
    return int(r.scalar_one() or 0)


# Minimum exercise-linked scoring rows per approach before phase-3 exploit activates.
MIN_APPROACH_SCORE_EVENTS = 2


async def approach_effectiveness_scores_by_mistake_type(
    session: AsyncSession,
    user_id: str,
    mistake_type: str,
    approaches: Sequence[str],
) -> Optional[dict[str, float]]:
    """Per-approach effectiveness for approach selection (phase 3).

    Derived from ``user_scoring_events`` joined through ``exercise_attempts`` to
    ``lesson_artifacts`` — i.e. outcomes that follow a lesson delivered with a
    given ``approach_type``.

    Higher returned value = better outcomes (fewer post-lesson mistakes /
    more exercise successes). Returns ``None`` when any registered approach lacks
    enough exercise-linked events to compare fairly.
    """
    if not mistake_type or not approaches:
        return None

    stmt = (
        select(
            LessonArtifact.approach_type,
            func.sum(UserScoringEvent.delta).label("raw_sum"),
            func.count(UserScoringEvent.id).label("n"),
        )
        .select_from(UserScoringEvent)
        .join(
            ExerciseAttempt,
            ExerciseAttempt.exercise_attempt_id
            == UserScoringEvent.session_or_exercise_id,
        )
        .join(
            LessonArtifact,
            LessonArtifact.artifact_id == ExerciseAttempt.lesson_artifact_id,
        )
        .where(
            UserScoringEvent.user_id == user_id,
            LessonArtifact.user_id == user_id,
            LessonArtifact.mistake_type == mistake_type,
            LessonArtifact.approach_type.in_(list(approaches)),
        )
        .group_by(LessonArtifact.approach_type)
    )
    r = await session.execute(stmt)
    rows = {str(row[0]): (float(row[1] or 0), int(row[2] or 0)) for row in r.all()}

    scores: dict[str, float] = {}
    for name in approaches:
        raw_sum, n = rows.get(name, (0.0, 0))
        if n < MIN_APPROACH_SCORE_EVENTS:
            return None
        # Lower mistake delta sum = better; invert so selector ranks higher = better.
        scores[name] = -raw_sum

    return scores
