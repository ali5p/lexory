"""Add exercises table; extend exercise_attempts; drop lesson_artifacts.exercises.

Revision ID: 0009_exercises_table
Revises: 0008_user_mistake_type_stats
Create Date: 2026-07-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009_exercises_table"
down_revision: Union[str, Sequence[str], None] = "0008_user_mistake_type_stats"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_names() -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return set(inspector.get_table_names())


def _column_names(table: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if "exercises" not in _table_names():
        op.create_table(
            "exercises",
            sa.Column("exercise_id", sa.String(), nullable=False),
            sa.Column("artifact_id", sa.String(), nullable=False),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("type", sa.String(), nullable=False),
            sa.Column("mistake_type", sa.String(), nullable=False, server_default=""),
            sa.Column("source_sentence", sa.Text(), nullable=False, server_default=""),
            sa.Column("payload", JSONB, nullable=False),
            sa.Column("answer_key", JSONB, nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("exercise_id"),
        )
        op.create_index("ix_exercises_artifact_id", "exercises", ["artifact_id"])

    attempt_cols = _column_names("exercise_attempts")
    if "exercise_id" not in attempt_cols:
        op.add_column(
            "exercise_attempts",
            sa.Column("exercise_id", sa.String(), nullable=True),
        )
        op.create_index(
            "ix_exercise_attempts_exercise_id",
            "exercise_attempts",
            ["exercise_id"],
        )
    if "user_answer" not in attempt_cols:
        op.add_column(
            "exercise_attempts",
            sa.Column("user_answer", JSONB, nullable=True),
        )
    if "is_correct" not in attempt_cols:
        op.add_column(
            "exercise_attempts",
            sa.Column("is_correct", sa.Boolean(), nullable=True),
        )

    artifact_cols = _column_names("lesson_artifacts")
    if "exercises" in artifact_cols:
        op.drop_column("lesson_artifacts", "exercises")


def downgrade() -> None:
    artifact_cols = _column_names("lesson_artifacts")
    if "exercises" not in artifact_cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("exercises", JSONB, nullable=False, server_default="[]"),
        )
        op.alter_column("lesson_artifacts", "exercises", server_default=None)

    attempt_cols = _column_names("exercise_attempts")
    if "is_correct" in attempt_cols:
        op.drop_column("exercise_attempts", "is_correct")
    if "user_answer" in attempt_cols:
        op.drop_column("exercise_attempts", "user_answer")
    if "exercise_id" in attempt_cols:
        op.drop_index("ix_exercise_attempts_exercise_id", table_name="exercise_attempts")
        op.drop_column("exercise_attempts", "exercise_id")

    if "exercises" in _table_names():
        op.drop_index("ix_exercises_artifact_id", table_name="exercises")
        op.drop_table("exercises")
