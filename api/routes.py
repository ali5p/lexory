from fastapi import APIRouter, Depends
from core.models import DocumentRequest, DocumentResponse, QueryRequest, QueryResponse
from rag.service import RAGService
from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


def get_rag_service() -> RAGService:
    qdrant_store = QdrantStore()
    embedder = Embedder()
    return RAGService(qdrant_store, embedder)


router = APIRouter()


@router.post("/documents", response_model=DocumentResponse)
async def ingest_document(
    request: DocumentRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    from core.models import UserText

    user_text = UserText(text=request.text, user_id=request.user_id)
    document_id, user_text_id = rag_service.ingest_user_text(
        user_text, session_id=request.session_id
    )

    return DocumentResponse(
        document_id=document_id, user_text_id=user_text_id, status="ingested"
    )


@router.post("/query", response_model=QueryResponse)
async def query_lesson(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    response = rag_service.generate_lesson(request)
    return response

