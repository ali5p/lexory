import pytest
from sqlalchemy import create_engine, event, text

from core.context import current_user_id_ctx
from storage import database


def test_session_transaction_sets_rls_user_context():
    values = []
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def register_set_config(connection, _record):
        connection.create_function(
            "set_config",
            3,
            lambda name, value, is_local: values.append(
                (name, value, is_local)
            )
            or value,
        )

    context_token = current_user_id_ctx.set("account-uuid")
    try:
        with database.Session(engine) as session:
            session.execute(text("SELECT 1"))
    finally:
        current_user_id_ctx.reset(context_token)
        engine.dispose()

    assert values == [("app.current_user_id", "account-uuid", 1)]


def test_rls_context_is_empty_without_authenticated_user():
    values = []
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def register_set_config(connection, _record):
        connection.create_function(
            "set_config",
            3,
            lambda name, value, is_local: values.append(
                (name, value, is_local)
            )
            or value,
        )

    try:
        with database.Session(engine) as session:
            session.execute(text("SELECT 1"))
    finally:
        engine.dispose()

    assert values == [("app.current_user_id", "", 1)]


def test_runtime_database_url_does_not_fall_back_to_admin_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://admin@localhost/db")
    monkeypatch.delenv("APP_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="APP_DATABASE_URL"):
        database.get_app_database_url()
