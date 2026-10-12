from fastapi import APIRouter, Depends, HTTPException, Request

from core.auth import get_current_user
from core.exercises import ExerciseAnswerRequest, ExerciseAnswerResponse
from core.models import SubmitRequest, SubmitResponse
from rag.service import RAGService
from storage.models import DBUser

router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post("/submit", response_model=SubmitResponse)
async def submit_and_lesson(
    request: SubmitRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: DBUser = Depends(get_current_user),
):

    return await rag_service.submit_and_lesson(
        text=request.text,
        user_id=current_user.id,
    )


@router.post("/exercises/{exercise_id}/answer", response_model=ExerciseAnswerResponse)
async def submit_exercise_answer(
    exercise_id: str,
    request: ExerciseAnswerRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: DBUser = Depends(get_current_user),
):
    try:
        return await rag_service.process_exercise_answer(
            exercise_id,
            request,
            user_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
