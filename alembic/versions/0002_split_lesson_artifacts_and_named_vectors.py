"""Split lesson_artifacts fields for lesson artifact points.

Revision ID: 0002_split_lesson_artifacts
Revises: 0001_user_scoring_events
Create Date: 2026-05-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "0002_split_lesson_artifacts"
down_revision: Union[str, Sequence[str], None] = "0001_user_scoring_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    cols = _column_names("lesson_artifacts")

    if not cols:
        op.create_table(
            "lesson_artifacts",
            sa.Column("artifact_id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("topic", sa.String(), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column(
                "exercises",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
            sa.Column("approach_type", sa.String(), nullable=False),
            sa.Column(
                "mistake_types_covered",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("artifact_id"),
        )
        op.create_index(
            "ix_lesson_artifacts_user_id",
            "lesson_artifacts",
            ["user_id"],
        )
        op.create_index(
            "ix_lesson_artifacts_created_at",
            "lesson_artifacts",
            ["created_at"],
        )
        return

    if "topic" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("topic", sa.String(), server_default="", nullable=False),
        )
    if "explanation" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("explanation", sa.Text(), server_default="", nullable=False),
        )
    if "exercises" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column(
                "exercises",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
        )

    if "lesson_type" in cols:
        op.execute(
            "UPDATE lesson_artifacts SET topic = lesson_type "
            "WHERE topic = '' OR topic IS NULL"
        )
    if "content" in cols:
        op.execute(
            "UPDATE lesson_artifacts SET explanation = content "
            "WHERE explanation = '' OR explanation IS NULL"
        )

    op.alter_column("lesson_artifacts", "topic", server_default=None)
    op.alter_column("lesson_artifacts", "explanation", server_default=None)
    op.alter_column("lesson_artifacts", "exercises", server_default=None)

    if "lesson_type" in cols:
        op.drop_column("lesson_artifacts", "lesson_type")
    if "content" in cols:
        op.drop_column("lesson_artifacts", "content")


def downgrade() -> None:
    cols = _column_names("lesson_artifacts")
    if not cols:
        return

    if "content" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("content", sa.Text(), server_default="", nullable=False),
        )
    if "lesson_type" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("lesson_type", sa.String(), server_default="", nullable=False),
        )

    if "explanation" in cols:
        op.execute(
            "UPDATE lesson_artifacts SET content = explanation "
            "WHERE content = '' OR content IS NULL"
        )
    if "topic" in cols:
        op.execute(
            "UPDATE lesson_artifacts SET lesson_type = topic "
            "WHERE lesson_type = '' OR lesson_type IS NULL"
        )

    op.alter_column("lesson_artifacts", "content", server_default=None)
    op.alter_column("lesson_artifacts", "lesson_type", server_default=None)

    if "exercises" in cols:
        op.drop_column("lesson_artifacts", "exercises")
    if "explanation" in cols:
        op.drop_column("lesson_artifacts", "explanation")
    if "topic" in cols:
        op.drop_column("lesson_artifacts", "topic")
