"""Add selection_index and is_contrast_lesson to lesson_artifacts.

Revision ID: 0005_artifact_selection_metadata
Revises: 0004_occurrence_user_type_index
Create Date: 2026-06-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0005_artifact_selection_metadata"
down_revision: Union[str, Sequence[str], None] = "0004_occurrence_user_type_index"
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
    if "selection_index" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column(
                "selection_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    if "is_contrast_lesson" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column(
                "is_contrast_lesson",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )

    op.alter_column("lesson_artifacts", "selection_index", server_default=None)
    op.alter_column("lesson_artifacts", "is_contrast_lesson", server_default=None)


def downgrade() -> None:
    cols = _column_names("lesson_artifacts")
    if "is_contrast_lesson" in cols:
        op.drop_column("lesson_artifacts", "is_contrast_lesson")
    if "selection_index" in cols:
        op.drop_column("lesson_artifacts", "selection_index")
