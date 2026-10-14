"""Enforce user isolation in PostgreSQL.

Revision ID: 0012_postgres_row_level_security
Revises: 0011_purge_anonymous_data
"""

from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0012_postgres_row_level_security"
down_revision: Union[str, Sequence[str], None] = "0011_purge_anonymous_data"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

USER_ID_TABLES = (
    "mistake_occurrences",
    "lesson_artifacts",
    "exercise_attempts",
    "user_scoring_events",
    "user_mistake_type_stats",
)

def upgrade() -> None:
    password = os.environ.get("LEXORY_APP_DB_PASSWORD", "").strip()
    if len(password) < 32:
        raise RuntimeError(
            "LEXORY_APP_DB_PASSWORD must contain at least 32 characters "
            "before applying the PostgreSQL RLS migration"
        )

    connection = op.get_bind()
    connection.execute(
        sa.text(
            "SELECT set_config('lexory.app_role_password', :password, true)"
        ),
        {"password": password},
    )
    connection.exec_driver_sql(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lexory_app') THEN
                EXECUTE format(
                    'ALTER ROLE lexory_app LOGIN NOSUPERUSER NOCREATEDB '
                    'NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS '
                    'PASSWORD %L',
                    current_setting('lexory.app_role_password')
                );
            ELSE
                EXECUTE format(
                    'CREATE ROLE lexory_app LOGIN NOSUPERUSER NOCREATEDB '
                    'NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS '
                    'PASSWORD %L',
                    current_setting('lexory.app_role_password')
                );
            END IF;
        END
        $$
        """
    )

    op.execute("GRANT USAGE ON SCHEMA public TO lexory_app")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "TO lexory_app"
    )
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO lexory_app")
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO lexory_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO lexory_app"
    )

    for table in USER_ID_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_user_isolation ON {table}
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true))
            WITH CHECK (user_id = current_setting('app.current_user_id', true))
            """
        )

    op.execute("ALTER TABLE exercises ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE exercises FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY exercises_user_isolation ON exercises
        FOR ALL
        USING (
            EXISTS (
                SELECT 1
                FROM lesson_artifacts
                WHERE lesson_artifacts.artifact_id = exercises.artifact_id
                  AND lesson_artifacts.user_id =
                      current_setting('app.current_user_id', true)
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM lesson_artifacts
                WHERE lesson_artifacts.artifact_id = exercises.artifact_id
                  AND lesson_artifacts.user_id =
                      current_setting('app.current_user_id', true)
            )
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY exercises_user_isolation ON exercises")
    op.execute("ALTER TABLE exercises DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE exercises NO FORCE ROW LEVEL SECURITY")

    for table in reversed(USER_ID_TABLES):
        op.execute(f"DROP POLICY {table}_user_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")

    op.execute(
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "FROM lexory_app"
    )
    op.execute(
        "REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM lexory_app"
    )
    op.execute("REVOKE USAGE ON SCHEMA public FROM lexory_app")
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM lexory_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE USAGE, SELECT ON SEQUENCES FROM lexory_app"
    )
    op.execute("DROP ROLE lexory_app")
