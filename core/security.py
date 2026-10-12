"""Authentication support clients."""

import os

from redis.asyncio import Redis


def create_redis_client() -> Redis:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        host = os.getenv("REDIS_HOST", "localhost").strip()
        port = os.getenv("REDIS_PORT", "6379").strip()
        redis_url = f"redis://{host}:{port}/0"
    return Redis.from_url(redis_url, decode_responses=True)


def get_allowed_origins() -> list[str]:
    raw = os.getenv(
        "LEXORY_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
