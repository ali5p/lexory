"""Add example_count_at_generation to lesson_artifacts.

Revision ID: 0006_artifact_ex_count
Revises: 0005_artifact_selection_metadata
Create Date: 2026-07-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0006_artifact_ex_count"
down_revision: Union[str, Sequence[str], None] = "0005_artifact_selection_metadata"
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
    if "example_count_at_generation" not in cols:
        op.add_column(
            "lesson_artifacts",
            sa.Column(
                "example_count_at_generation",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    op.alter_column(
        "lesson_artifacts", "example_count_at_generation", server_default=None
    )


def downgrade() -> None:
    cols = _column_names("lesson_artifacts")
    if "example_count_at_generation" in cols:
        op.drop_column("lesson_artifacts", "example_count_at_generation")
