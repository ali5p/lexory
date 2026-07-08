"""Add origin_session_id to exercise_attempts.

Revision ID: 0007_exercise_origin_session
Revises: 0006_artifact_ex_count
Create Date: 2026-07-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0007_exercise_origin_session"
down_revision: Union[str, Sequence[str], None] = "0006_artifact_ex_count"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    cols = _column_names("exercise_attempts")
    if "origin_session_id" not in cols:
        op.add_column(
            "exercise_attempts",
            sa.Column(
                "origin_session_id",
                sa.String(),
                nullable=False,
                server_default="",
            ),
        )
    op.alter_column("exercise_attempts", "origin_session_id", server_default=None)


def downgrade() -> None:
    cols = _column_names("exercise_attempts")
    if "origin_session_id" in cols:
        op.drop_column("exercise_attempts", "origin_session_id")
