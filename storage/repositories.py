"""Thin async repository functions for PostgreSQL persistence."""

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.activity_timeline import UserActivity, build_activity_timeline
from core.mistake_type_stats import (
    MistakeTypeStatsConfig,
    MistakeTypeStatsRow,
    ScoringEventRow,
    compute_mistake_type_stats,
)
from core.exercises import ExercisePayload, build_exercise_payload
from storage.models import (
    Exercise,
    ExerciseAttempt,
    LessonArtifact,
    MistakeOccurrence,
    UserMistakeTypeStats,
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
        approach_type=data.get("approach_type", ""),
        mistake_type=data.get("mistake_type", ""),
        selection_index=int(data.get("selection_index", 0) or 0),
        is_contrast_lesson=bool(data.get("is_contrast_lesson", False)),
        example_count_at_generation=int(
            data.get("example_count_at_generation", 0) or 0
        ),
        created_at=data.get("created_at", ""),
    )
    await session.merge(row)
    await session.flush()


async def insert_exercise_attempt(session: AsyncSession, data: dict) -> None:
    row = ExerciseAttempt(
        exercise_attempt_id=data["exercise_attempt_id"],
        exercise_id=data.get("exercise_id"),
        lesson_artifact_id=data["lesson_artifact_id"],
        user_id=data["user_id"],
        user_answer=data.get("user_answer"),
        is_correct=data.get("is_correct"),
        attempt_timestamp=data["attempt_timestamp"],
        origin_session_id=data.get("origin_session_id", ""),
    )
    session.add(row)
    await session.flush()


async def insert_exercises(session: AsyncSession, rows: list[dict]) -> None:
    for data in rows:
        session.add(
            Exercise(
                exercise_id=data["exercise_id"],
                artifact_id=data["artifact_id"],
                sort_order=int(data.get("sort_order", 0) or 0),
                type=data["type"],
                mistake_type=data.get("mistake_type", ""),
                source_sentence=data.get("source_sentence", ""),
                payload=data["payload"],
                answer_key=data["answer_key"],
                created_at=data["created_at"],
            )
        )
    await session.flush()


async def list_exercises_by_artifact_id(
    session: AsyncSession, artifact_id: str
) -> list[ExercisePayload]:
    stmt = (
        select(Exercise)
        .where(Exercise.artifact_id == artifact_id)
        .order_by(Exercise.sort_order, Exercise.created_at)
    )
    r = await session.execute(stmt)
    rows = r.scalars().all()
    return [
        build_exercise_payload(
            exercise_id=row.exercise_id,
            mistake_type=row.mistake_type or "",
            source_sentence=row.source_sentence or "",
            payload=row.payload,
        )
        for row in rows
    ]


async def get_exercise_by_id(
    session: AsyncSession, exercise_id: str
) -> Optional[Exercise]:
    stmt = select(Exercise).where(Exercise.exercise_id == exercise_id)
    r = await session.execute(stmt)
    return r.scalar_one_or_none()


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
    *,
    comparison_min_example_count: int,
) -> Optional[dict[str, float]]:
    """Per-approach effectiveness for approach selection (phase 3).

    Derived from ``user_scoring_events`` joined through ``exercise_attempts`` to
    ``lesson_artifacts`` — i.e. outcomes that follow a lesson delivered with a
    given ``approach_type``. Lessons from the coldest baseline-only window
    (``example_count_at_generation < comparison_min_example_count``, typically
    counts 0 and 1) are excluded; the last baseline-only lesson (count 2) stays
    in comparison because rotation at ``EXPLORE_MIN`` opens with ``rule_based``.

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
            LessonArtifact.example_count_at_generation >= comparison_min_example_count,
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


async def get_user_activity_timeline(
    session: AsyncSession, user_id: str
) -> list[UserActivity]:
    """Ordered submit + exercise activities for one user (activity_index 0 = earliest)."""
    submit_sessions = await _fetch_submit_session_starts(session, user_id)
    exercise_rows = await _fetch_exercise_activity_rows(session, user_id)
    return build_activity_timeline(
        submit_sessions=submit_sessions,
        exercise_attempts=exercise_rows,
    )


async def _fetch_submit_session_starts(
    session: AsyncSession, user_id: str
) -> list[tuple[str, str]]:
    """Earliest timestamp per submit session_id (occurrences + lesson artifacts)."""
    occ_stmt = (
        select(
            MistakeOccurrence.session_id,
            func.min(MistakeOccurrence.detected_at).label("occurred_at"),
        )
        .where(
            MistakeOccurrence.user_id == user_id,
            MistakeOccurrence.session_id != "",
            MistakeOccurrence.source != "exercise_attempt",
        )
        .group_by(MistakeOccurrence.session_id)
    )
    art_stmt = (
        select(
            LessonArtifact.session_id,
            func.min(LessonArtifact.created_at).label("occurred_at"),
        )
        .where(
            LessonArtifact.user_id == user_id,
            LessonArtifact.session_id != "",
        )
        .group_by(LessonArtifact.session_id)
    )
    occ_r = await session.execute(occ_stmt)
    art_r = await session.execute(art_stmt)

    earliest: dict[str, str] = {}
    for session_id, occurred_at in occ_r.all():
        sid = str(session_id or "")
        at = str(occurred_at or "")
        if sid and (sid not in earliest or at < earliest[sid]):
            earliest[sid] = at
    for session_id, occurred_at in art_r.all():
        sid = str(session_id or "")
        at = str(occurred_at or "")
        if sid and (sid not in earliest or at < earliest[sid]):
            earliest[sid] = at

    return list(earliest.items())


async def _fetch_exercise_activity_rows(
    session: AsyncSession, user_id: str
) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    stmt = (
        select(
            ExerciseAttempt.exercise_attempt_id,
            ExerciseAttempt.attempt_timestamp,
            ExerciseAttempt.lesson_artifact_id,
            ExerciseAttempt.origin_session_id,
        )
        .where(ExerciseAttempt.user_id == user_id)
        .order_by(ExerciseAttempt.attempt_timestamp)
    )
    r = await session.execute(stmt)
    return [
        (
            str(row[0]),
            str(row[1]),
            str(row[2]) if row[2] else None,
            str(row[3]) if row[3] else None,
        )
        for row in r.all()
    ]


async def recompute_user_mistake_type_stats(
    session: AsyncSession,
    user_id: str,
    *,
    config: MistakeTypeStatsConfig | None = None,
    computed_at: str | None = None,
) -> int:
    """Recompute and persist per-MT stats for one user. Returns row count."""
    timeline = await get_user_activity_timeline(session, user_id)
    scoring_events = await fetch_scoring_events_for_user(session, user_id)
    text_to_session = await fetch_text_to_session_map(session, user_id)
    rows = compute_mistake_type_stats(
        user_id=user_id,
        timeline=timeline,
        scoring_events=scoring_events,
        text_to_session=text_to_session,
        config=config,
    )
    ts = computed_at or datetime.now(timezone.utc).isoformat()
    await replace_user_mistake_type_stats(session, user_id, rows, computed_at=ts)
    return len(rows)


async def fetch_scoring_events_for_user(
    session: AsyncSession, user_id: str
) -> list[ScoringEventRow]:
    stmt = (
        select(
            UserScoringEvent.mistake_type,
            UserScoringEvent.delta,
            UserScoringEvent.session_or_exercise_id,
        )
        .where(UserScoringEvent.user_id == user_id)
        .order_by(UserScoringEvent.occurred_at)
    )
    r = await session.execute(stmt)
    return [
        ScoringEventRow(
            mistake_type=str(row[0] or ""),
            delta=float(row[1] or 0),
            session_or_exercise_id=str(row[2] or ""),
        )
        for row in r.all()
    ]


async def fetch_text_to_session_map(
    session: AsyncSession, user_id: str
) -> dict[str, str]:
    """Map submit user_text_id -> session_id from mistake occurrences."""
    stmt = (
        select(MistakeOccurrence.user_text_id, MistakeOccurrence.session_id)
        .where(
            MistakeOccurrence.user_id == user_id,
            MistakeOccurrence.user_text_id != "",
            MistakeOccurrence.session_id != "",
            MistakeOccurrence.source != "exercise_attempt",
        )
        .distinct()
    )
    r = await session.execute(stmt)
    return {
        str(row[0]): str(row[1])
        for row in r.all()
        if row[0] and row[1]
    }


async def replace_user_mistake_type_stats(
    session: AsyncSession,
    user_id: str,
    rows: list[MistakeTypeStatsRow],
    computed_at: str,
) -> None:
    await session.execute(
        delete(UserMistakeTypeStats).where(UserMistakeTypeStats.user_id == user_id)
    )
    for row in rows:
        session.add(
            UserMistakeTypeStats(
                user_id=row.user_id,
                mistake_type=row.mistake_type,
                first_activity_index=row.first_activity_index,
                lifetime_score=row.lifetime_score,
                recent_burden=row.recent_burden,
                historical_burden=row.historical_burden,
                is_new=row.is_new,
                is_improving=row.is_improving,
                is_relapsed=row.is_relapsed,
                priority_score=row.priority_score,
                total_activity_count=row.total_activity_count,
                computed_at=computed_at,
            )
        )
    await session.flush()


async def get_mistake_type_stats_for_user(
    session: AsyncSession, user_id: str
) -> list[MistakeTypeStatsRow]:
    stmt = (
        select(UserMistakeTypeStats)
        .where(UserMistakeTypeStats.user_id == user_id)
        .order_by(UserMistakeTypeStats.priority_score.desc())
    )
    r = await session.execute(stmt)
    return [
        MistakeTypeStatsRow(
            user_id=row.user_id,
            mistake_type=row.mistake_type,
            first_activity_index=row.first_activity_index,
            lifetime_score=row.lifetime_score,
            recent_burden=row.recent_burden,
            historical_burden=row.historical_burden,
            is_new=row.is_new,
            is_improving=row.is_improving,
            is_relapsed=row.is_relapsed,
            priority_score=row.priority_score,
            total_activity_count=row.total_activity_count,
        )
        for row in r.scalars().all()
    ]


async def top_priority_mistake_types(
    session: AsyncSession, user_id: str, k: int = 3
) -> list[str]:
    """Top-k MTs by batch priority_score (empty if batch has not run)."""
    stmt = (
        select(UserMistakeTypeStats.mistake_type)
        .where(UserMistakeTypeStats.user_id == user_id)
        .order_by(UserMistakeTypeStats.priority_score.desc())
        .limit(k)
    )
    r = await session.execute(stmt)
    return [str(row[0]) for row in r.all() if row[0]]


async def should_offer_supplemental_practice(
    session: AsyncSession,
    user_id: str,
    exclude_mistake_types: set[str],
    *,
    every_n_submits: int,
) -> bool:
    """True on every Nth submit when a priority MT is not in this submit."""
    timeline = await get_user_activity_timeline(session, user_id)
    submit_count = sum(1 for activity in timeline if activity.kind == "submit")
    if submit_count < 1 or submit_count % every_n_submits != 0:
        return False
    priority = await top_priority_mistake_types(session, user_id, k=10)
    excluded = exclude_mistake_types or set()
    return any(mt not in excluded for mt in priority)
