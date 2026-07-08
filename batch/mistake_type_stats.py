"""CLI to recompute user_mistake_type_stats for one user (debug / backfill)."""

from __future__ import annotations

import asyncio
import logging
import sys

from storage.database import build_engine, build_session_factory
from storage import repositories as repo

_log = logging.getLogger(__name__)


async def _main_async(user_id: str) -> None:
    logging.basicConfig(level=logging.INFO)
    engine = build_engine()
    session_factory = build_session_factory(engine)
    try:
        async with session_factory() as session:
            count = await repo.recompute_user_mistake_type_stats(session, user_id)
            await session.commit()
        _log.info("user=%s rows=%s", user_id, count)
    finally:
        await engine.dispose()


def main(user_id: str) -> None:
    asyncio.run(_main_async(user_id))


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: python -m batch.mistake_type_stats USER_ID")
    main(user_id=sys.argv[1].strip())
