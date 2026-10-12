"""Reserved revision for the manual legacy-data cleanup script.

Revision ID: 0011_purge_anonymous_data
Revises: 0010_auth_users
"""

from __future__ import annotations

from typing import Sequence, Union

revision: str = "0011_purge_anonymous_data"
down_revision: Union[str, Sequence[str], None] = "0010_auth_users"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
