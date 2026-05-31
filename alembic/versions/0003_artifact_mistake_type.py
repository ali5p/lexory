"""Replace mistake_types_covered with mistake_type on lesson_artifacts.

Revision ID: 0003_artifact_mistake_type
Revises: 0002_split_lesson_artifacts
Create Date: 2026-05-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "0003_artifact_mistake_type"
down_revision: Union[str, Sequence[str], None] = "0002_split_lesson_artifacts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def upgrade() -> None:
    cols = _column_names("lesson_artifacts")
    if "mistake_type" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column("mistake_type", sa.String(), nullable=False, server_default=""),
        )

    if "mistake_types_covered" in cols:
        op.execute(
            """
            UPDATE lesson_artifacts
            SET mistake_type = COALESCE(mistake_types_covered->>0, '')
            WHERE mistake_type = '' OR mistake_type IS NULL
            """
        )
        op.drop_column("lesson_artifacts", "mistake_types_covered")

    op.alter_column("lesson_artifacts", "mistake_type", server_default=None)

    if "ix_lesson_artifacts_mistake_type" not in _index_names("lesson_artifacts"):
        op.create_index(
            "ix_lesson_artifacts_mistake_type",
            "lesson_artifacts",
            ["mistake_type"],
            unique=False,
        )


def downgrade() -> None:
    cols = _column_names("lesson_artifacts")
    if "mistake_types_covered" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column(
                "mistake_types_covered",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
        )
        op.execute(
            """
            UPDATE lesson_artifacts
            SET mistake_types_covered = CASE
                WHEN mistake_type IS NOT NULL AND mistake_type <> ''
                THEN jsonb_build_array(mistake_type)
                ELSE '[]'::jsonb
            END
            """
        )

    if "mistake_type" in cols:
        if "ix_lesson_artifacts_mistake_type" in _index_names("lesson_artifacts"):
            op.drop_index(
                "ix_lesson_artifacts_mistake_type", table_name="lesson_artifacts"
            )
        op.drop_column("lesson_artifacts", "mistake_type")
