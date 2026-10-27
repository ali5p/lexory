"""add external identities with RLS isolation.

Revision ID: 0013_add_external_identities
Revises: 0012_postgres_row_level_security
"""

from __future__ import annotations

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "0013_add_external_identities"
down_revision: Union[str, Sequence[str], None] = "0012_postgres_row_level_security"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "external_identities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_subject", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "provider_subject", name="uq_external_identity_provider_subject")
    )
    op.create_index(op.f("ix_external_identities_user_id"), "external_identities", ["user_id"], unique=False)

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE external_identities TO lexory_app")
    op.execute("GRANT USAGE, SELECT ON SEQUENCE external_identities_id_seq TO lexory_app")

    op.execute("ALTER TABLE external_identities ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE external_identities FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY external_identities_user_isolation ON external_identities
        FOR ALL
        USING (user_id = current_setting('app.current_user_id', true))
        WITH CHECK (user_id = current_setting('app.current_user_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY external_identities_user_isolation ON external_identities")
    op.execute("ALTER TABLE external_identities DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE external_identities NO FORCE ROW LEVEL SECURITY")

    op.execute("REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE external_identities FROM lexory_app")
    op.execute("REVOKE USAGE, SELECT ON SEQUENCE external_identities_id_seq FROM lexory_app")

    op.drop_index(op.f("ix_external_identities_user_id"), table_name="external_identities")
    op.drop_table("external_identities")
