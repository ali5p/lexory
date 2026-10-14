import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.auth import get_jwt_secret
from core.auth import router as auth_router
from core.security import create_redis_client, get_allowed_origins
from rag.embedder import Embedder
from rag.service import RAGService
from storage.database import (
    build_engine,
    build_session_factory,
    check_database_with_retry,
    dispose_engine,
    run_alembic_upgrade_sync,
)
from vectorstore.qdrant_client import QdrantStore

# Environment-backed settings are read during app configuration and lifespan startup.
load_dotenv(override=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_jwt_secret()
    await asyncio.to_thread(run_alembic_upgrade_sync)
    engine = build_engine()
    redis_client = create_redis_client()
    try:
        await redis_client.ping()
        await check_database_with_retry(engine)
        session_factory = build_session_factory(engine)
        app.state.session_factory = session_factory
        app.state.redis_client = redis_client
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_store = QdrantStore(url=qdrant_url) if qdrant_url else QdrantStore()
        qdrant_store = QdrantStore(url=qdrant_url) if qdrant_url else QdrantStore()
        embedder = Embedder()
        app.state.rag_service = RAGService(qdrant_store, embedder, session_factory)
        yield
    finally:
        await redis_client.aclose()
        await dispose_engine(engine)


app = FastAPI(title="Lexory", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(auth_router)
