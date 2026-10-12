"""Create accounts for authenticated requests.

Revision ID: 0010_auth_users
Revises: 0009_exercises_table
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0010_auth_users"
down_revision: Union[str, Sequence[str], None] = "0009_exercises_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
