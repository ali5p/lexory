import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

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
)
from vectorstore.qdrant_client import QdrantStore


@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.include_router(router)

