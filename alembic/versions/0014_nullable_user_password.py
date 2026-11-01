"""Allow accounts authenticated only by an external identity.

Revision ID: 0014_nullable_user_password
Revises: 0013_add_external_identities
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0014_nullable_user_password"
down_revision: Union[str, Sequence[str], None] = "0013_add_external_identities"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.String(),
        nullable=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    has_passwordless_accounts = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM users WHERE hashed_password IS NULL)")
    ).scalar_one()
    if has_passwordless_accounts:
        raise RuntimeError(
            "Cannot require passwords while passwordless OAuth accounts exist"
        )

    op.alter_column(
        "users",
        "hashed_password",
        existing_type=sa.String(),
        nullable=False,
    )
