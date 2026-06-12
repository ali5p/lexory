"""Composite index on mistake_occurrences (user_id, mistake_type).

Supports the approach-selection example count
(COUNT WHERE user_id = ? AND mistake_type = ? AND example_id IS NOT NULL).

Revision ID: 0004_occurrence_user_type_index
Revises: 0003_artifact_mistake_type
Create Date: 2026-06-12
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

revision: str = "0004_occurrence_user_type_index"
down_revision: Union[str, Sequence[str], None] = "0003_artifact_mistake_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX_NAME = "ix_mistake_occurrences_user_type"
_TABLE = "mistake_occurrences"


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def upgrade() -> None:
    if _INDEX_NAME not in _index_names(_TABLE):
        op.create_index(
            _INDEX_NAME,
            _TABLE,
            ["user_id", "mistake_type"],
            unique=False,
        )


def downgrade() -> None:
    if _INDEX_NAME in _index_names(_TABLE):
        op.drop_index(_INDEX_NAME, table_name=_TABLE)
