import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from rag.embedder import Embedder
from rag.service import RAGService
from vectorstore.qdrant_client import QdrantStore


def create_rag_service() -> RAGService:
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_store = QdrantStore(url=qdrant_url) if qdrant_url else QdrantStore()
    embedder = Embedder()
    return RAGService(qdrant_store, embedder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = create_rag_service()
    try:
        yield
    finally:
        pass


app = FastAPI(title="Lexory", version="0.1.0", lifespan=lifespan)
app.include_router(router)

