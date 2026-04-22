"""Create user_scoring_events table

Revision ID: 0001_user_scoring_events
Revises:
Create Date: 2026-04-13
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_user_scoring_events"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_scoring_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("rule_id", sa.String(), nullable=True),
        sa.Column("mistake_type", sa.String(), nullable=False),
        sa.Column("mistake_id", sa.String(), nullable=True),
        sa.Column("session_or_exercise_id", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.String(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_user_scoring_events_user_id",
        "user_scoring_events",
        ["user_id"],
    )
    op.create_index(
        "ix_user_scoring_events_mistake_type",
        "user_scoring_events",
        ["mistake_type"],
    )
    op.create_index(
        "ix_user_scoring_events_session",
        "user_scoring_events",
        ["session_or_exercise_id"],
    )
    op.create_index(
        "ix_user_scoring_events_occurred_at",
        "user_scoring_events",
        ["occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_user_scoring_events_occurred_at", table_name="user_scoring_events")
    op.drop_index("ix_user_scoring_events_session", table_name="user_scoring_events")
    op.drop_index("ix_user_scoring_events_mistake_type", table_name="user_scoring_events")
    op.drop_index("ix_user_scoring_events_user_id", table_name="user_scoring_events")
    op.drop_table("user_scoring_events")
