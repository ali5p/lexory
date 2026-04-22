"""Async PostgreSQL engine and session factory."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from urllib.parse import quote, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from storage.models import Base

_log = logging.getLogger(__name__)


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _normalize_database_url_for_docker(url: str) -> str:
    """
    Host-mounted .env often uses localhost:5432. Inside a container that points at the
    container itself, not the postgres service. Rewrite to POSTGRES_HOST (default postgres).
    """
    if not url or not _running_in_docker():
        return url
    parsed = urlparse(url)
    host = parsed.hostname
    if host not in ("localhost", "127.0.0.1", "::1"):
        return url
    target = os.environ.get("POSTGRES_HOST", "postgres")
    port = parsed.port or 5432
    user = parsed.username or ""
    password = parsed.password or ""
    if user:
        user_enc = quote(user, safe="")
        pass_enc = quote(password, safe="") if password else ""
        auth = f"{user_enc}:{pass_enc}@" if password else f"{user_enc}@"
    else:
        auth = ""
    netloc = f"{auth}{target}:{port}"
    fixed = urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )
    _log.info(
        "DATABASE_URL pointed at %s inside Docker; using host %s instead",
        host,
        target,
    )
    return fixed


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return _normalize_database_url_for_docker(url)


def get_sync_migrations_url() -> str:
    """
    Sync driver URL for Alembic (and command.upgrade from app). Async sessions keep asyncpg.
    """
    url = get_database_url()
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql+asyncpg://")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    return url


def run_alembic_upgrade_sync() -> None:
    """Run `alembic upgrade head` using DATABASE_URL; intended for app startup in a thread."""
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    ini = root / "alembic.ini"
    if not ini.is_file():
        _log.warning("alembic.ini not found; skipping migrations")
        return
    cfg = Config(str(ini))
    cfg.set_main_option("sqlalchemy.url", get_sync_migrations_url())
    command.upgrade(cfg, "head")


def build_engine(database_url: str | None = None) -> AsyncEngine:
    url = database_url or get_database_url()
    # Docker / local Postgres often has TLS off; asyncpg may negotiate SSL otherwise.
    connect_args: dict = {}
    if os.environ.get("DATABASE_SSL", "").lower() not in ("1", "true", "yes"):
        connect_args["ssl"] = False
    return create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_tables_with_retry(
    engine: AsyncEngine,
    *,
    attempts: int = 30,
    delay_seconds: float = 2.0,
) -> None:
    """Run DDL after PostgreSQL accepts connections (Docker startup race)."""
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            await create_tables(engine)
            if attempt > 1:
                _log.info("PostgreSQL ready after %s attempt(s)", attempt)
            return
        except Exception as e:
            last_exc = e
            _log.warning(
                "Database not ready (attempt %s/%s): %s",
                attempt,
                attempts,
                e,
            )
            if attempt < attempts:
                await asyncio.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not connect to PostgreSQL after {attempts} attempts"
    ) from last_exc


async def dispose_engine(engine: AsyncEngine) -> None:
    await engine.dispose()


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield one session per call; commit is the caller's responsibility."""
    async with session_factory() as session:
        yield session
