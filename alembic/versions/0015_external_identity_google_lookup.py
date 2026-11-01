"""Permit RLS-scoped Google identity lookup during OAuth sign-in.

Revision ID: 0015_google_identity_rls
Revises: 0014_nullable_user_password
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0015_google_identity_rls"
down_revision: Union[str, Sequence[str], None] = "0014_nullable_user_password"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP POLICY external_identities_user_isolation ON external_identities")
    op.execute(
        """
        CREATE POLICY external_identities_user_isolation
        ON external_identities
        FOR ALL
        USING (user_id = current_setting('app.current_user_id', true))
        WITH CHECK (user_id = current_setting('app.current_user_id', true))
        """
    )
    op.execute(
        """
        CREATE POLICY external_identities_google_lookup
        ON external_identities
        FOR SELECT
        USING (
            provider = 'google'
            AND provider_subject =
                current_setting('app.google_subject', true)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY external_identities_google_lookup ON external_identities")
    op.execute("DROP POLICY external_identities_user_isolation ON external_identities")
    op.execute(
        """
        CREATE POLICY external_identities_user_isolation
        ON external_identities
        FOR ALL
        USING (user_id = current_setting('app.current_user_id', true))
        WITH CHECK (user_id = current_setting('app.current_user_id', true))
        """
    )
