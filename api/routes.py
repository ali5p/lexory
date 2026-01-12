from fastapi import APIRouter, Depends, Request

from core.models import DocumentRequest, DocumentResponse, QueryRequest, QueryResponse
from rag.service import RAGService


router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


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

