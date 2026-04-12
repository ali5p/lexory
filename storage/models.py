"""SQLAlchemy ORM models for PostgreSQL persistence."""

from typing import Optional

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MistakeOccurrence(Base):
    __tablename__ = "mistake_occurrences"

    mistake_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    user_text_id: Mapped[str] = mapped_column(String, index=True)
    detected_at: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)
    mistake_type: Mapped[str] = mapped_column(String, index=True)
    rule_id: Mapped[str] = mapped_column(String)
    example_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    lesson_artifact_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class ExampleImprint(Base):
    """Chronological record of example points written to Qdrant (fallback retrieval)."""

    __tablename__ = "example_imprints"

    mistake_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[str] = mapped_column(String, index=True)
    user_text_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rule_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mistake_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class LessonArtifact(Base):
    __tablename__ = "lesson_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    lesson_type: Mapped[str] = mapped_column(String)
    approach_type: Mapped[str] = mapped_column(String)
    mistake_types_covered: Mapped[list] = mapped_column(JSONB, default=list)
    pedagogy_tags: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[str] = mapped_column(String, index=True)


class ExerciseAttempt(Base):
    __tablename__ = "exercise_attempts"

    exercise_attempt_id: Mapped[str] = mapped_column(String, primary_key=True)
    lesson_artifact_id: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    attempt_timestamp: Mapped[str] = mapped_column(String, index=True)
