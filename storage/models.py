"""SQLAlchemy ORM models for PostgreSQL persistence."""

from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MistakeOccurrence(Base):
    __tablename__ = "mistake_occurrences"

    mistake_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str] = mapped_column(String, index=True, default="")
    user_text_id: Mapped[str] = mapped_column(String, index=True)
    detected_at: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)
    mistake_type: Mapped[str] = mapped_column(String, index=True)
    rule_id: Mapped[str] = mapped_column(String)
    example_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lesson_artifact_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class LessonArtifact(Base):
    __tablename__ = "lesson_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str] = mapped_column(String)
    topic: Mapped[str] = mapped_column(String)
    explanation: Mapped[str] = mapped_column(Text)
    exercises: Mapped[list] = mapped_column(JSONB, default=list)
    approach_type: Mapped[str] = mapped_column(String)
    mistake_type: Mapped[str] = mapped_column(String, index=True, default="")
    selection_index: Mapped[int] = mapped_column(Integer, default=0)
    is_contrast_lesson: Mapped[bool] = mapped_column(Boolean, default=False)
    example_count_at_generation: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String, index=True)


class ExerciseAttempt(Base):
    __tablename__ = "exercise_attempts"

    exercise_attempt_id: Mapped[str] = mapped_column(String, primary_key=True)
    lesson_artifact_id: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    attempt_timestamp: Mapped[str] = mapped_column(String, index=True)
    origin_session_id: Mapped[str] = mapped_column(String, default="")


class UserScoringEvent(Base):
    """
    Incremental scoring: +1 / 0 / -0.5 per event. Totals are SUM(delta) per mistake_type,
    clamped at read time; tie-break uses MAX(occurred_at) per type.
    """

    __tablename__ = "user_scoring_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    rule_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mistake_type: Mapped[str] = mapped_column(String, index=True)
    mistake_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    session_or_exercise_id: Mapped[str] = mapped_column(String, index=True)
    occurred_at: Mapped[str] = mapped_column(String, index=True)
    delta: Mapped[float] = mapped_column(Float, index=False)


class UserMistakeTypeStats(Base):
    """Batch-materialized per-user MT priority stats (activity-index windows)."""

    __tablename__ = "user_mistake_type_stats"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    mistake_type: Mapped[str] = mapped_column(String, primary_key=True)
    first_activity_index: Mapped[int] = mapped_column(Integer)
    lifetime_score: Mapped[float] = mapped_column(Float)
    recent_burden: Mapped[float] = mapped_column(Float)
    historical_burden: Mapped[float] = mapped_column(Float)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_improving: Mapped[bool] = mapped_column(Boolean, default=False)
    is_relapsed: Mapped[bool] = mapped_column(Boolean, default=False)
    priority_score: Mapped[float] = mapped_column(Float, index=True)
    total_activity_count: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[str] = mapped_column(String, index=True)
