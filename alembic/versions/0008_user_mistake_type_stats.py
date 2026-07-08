"""Add user_mistake_type_stats table.

Revision ID: 0008_user_mistake_type_stats
Revises: 0007_exercise_origin_session
Create Date: 2026-07-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0008_user_mistake_type_stats"
down_revision: Union[str, Sequence[str], None] = "0007_exercise_origin_session"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_names() -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return set(inspector.get_table_names())


def upgrade() -> None:
    if "user_mistake_type_stats" in _table_names():
        return
    op.create_table(
        "user_mistake_type_stats",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("mistake_type", sa.String(), nullable=False),
        sa.Column("first_activity_index", sa.Integer(), nullable=False),
        sa.Column("lifetime_score", sa.Float(), nullable=False),
        sa.Column("recent_burden", sa.Float(), nullable=False),
        sa.Column("historical_burden", sa.Float(), nullable=False),
        sa.Column("is_new", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_improving", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_relapsed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("total_activity_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("computed_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "mistake_type"),
    )
    op.create_index(
        "ix_user_mistake_type_stats_user_priority",
        "user_mistake_type_stats",
        ["user_id", "priority_score"],
    )


def downgrade() -> None:
    if "user_mistake_type_stats" not in _table_names():
        return
    op.drop_index(
        "ix_user_mistake_type_stats_user_priority",
        table_name="user_mistake_type_stats",
    )
    op.drop_table("user_mistake_type_stats")
