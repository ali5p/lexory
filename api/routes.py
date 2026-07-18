from fastapi import APIRouter, Depends, HTTPException, Request

from core.exercises import ExerciseAnswerRequest, ExerciseAnswerResponse
from core.models import SubmitRequest, SubmitResponse
from rag.service import RAGService


router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post("/submit", response_model=SubmitResponse)
async def submit_and_lesson(
    request: SubmitRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Combined ingest + lesson. Single flow: text (optional) + user_id → lesson items."""
    return await rag_service.submit_and_lesson(
        text=request.text,
        user_id=request.user_id,
    )


@router.post("/exercises/{exercise_id}/answer", response_model=ExerciseAnswerResponse)
async def submit_exercise_answer(
    exercise_id: str,
    request: ExerciseAnswerRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        return await rag_service.process_exercise_answer(exercise_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
