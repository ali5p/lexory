from fastapi import APIRouter, Depends, Request

from core.models import SubmitRequest, SubmitResponse, ExerciseAttempt
from rag.service import RAGService


router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post("/submit", response_model=SubmitResponse)
async def submit_and_lesson(
    request: SubmitRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Combined ingest + lesson. Single flow: text (optional) + user_id → lesson + context."""
    user_text_id, session_id, lesson_artifact_id, lesson, context = (
        await rag_service.submit_and_lesson(text=request.text, user_id=request.user_id)
    )

    return SubmitResponse(
        user_text_id=user_text_id,
        session_id=session_id,
        lesson_artifact_id=lesson_artifact_id,
        lesson=lesson,
        context=context,
        status="ok",
    )


@router.post("/exercise-feedback")
async def submit_exercise(
    request: ExerciseAttempt,
    rag_service: RAGService = Depends(get_rag_service),
):
    return await rag_service.process_exercise_attempt(request)
