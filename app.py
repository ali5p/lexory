import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Do not override variables already set by Docker Compose (e.g. DATABASE_URL=@postgres).
load_dotenv(override=False)

from api.routes import router
from rag.embedder import Embedder
from rag.service import RAGService
from storage.database import (
    build_engine,
    build_session_factory,
    create_tables_with_retry,
    dispose_engine,
    run_alembic_upgrade_sync,
)
from vectorstore.qdrant_client import QdrantStore


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "LEXORY_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(run_alembic_upgrade_sync)
    engine = build_engine()
    await create_tables_with_retry(engine)
    session_factory = build_session_factory(engine)

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_store = QdrantStore(url=qdrant_url) if qdrant_url else QdrantStore()
    embedder = Embedder()

    app.state.rag_service = RAGService(qdrant_store, embedder, session_factory)
    try:
        yield
    finally:
        await dispose_engine(engine)


app = FastAPI(title="Lexory", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

