from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes import submit_and_lesson, submit_exercise_answer
from core.auth import (
    create_access_token,
    get_current_user,
    get_jwt_secret,
    hash_password,
    verify_password,
)
from core.exercises import ExerciseAnswerRequest, ExerciseAnswerResponse
from core.models import SubmitRequest
from storage.models import DBUser


def test_jwt_secret_requires_at_least_32_characters(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        get_jwt_secret()


def test_password_hash_is_verified_without_storing_plaintext():
    password = "correct horse battery staple"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("incorrect password", hashed)


@pytest.mark.asyncio
async def test_get_current_user_uses_signed_subject(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 40)
    user = DBUser(id="account-id", email="person@example.com", hashed_password="hash")
    db = AsyncMock(spec=AsyncSession)
    db.get.return_value = user
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=create_access_token(user.id),
    )

    result = await get_current_user(credentials, db)

    assert result is user
    db.get.assert_awaited_once_with(DBUser, user.id)


@pytest.mark.asyncio
async def test_get_current_user_rejects_missing_credentials(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 40)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(None, AsyncMock(spec=AsyncSession))

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_submit_uses_authenticated_user_id():
    rag_service = AsyncMock()
    account = DBUser(
        id="account-id",
        email="person@example.com",
        hashed_password="hash",
    )
    request = SubmitRequest.model_validate(
        {"text": "I has a book.", "user_id": "untrusted-client-id"}
    )

    await submit_and_lesson(request, rag_service, account)

    rag_service.submit_and_lesson.assert_awaited_once_with(
        text="I has a book.",
        user_id="account-id",
    )


@pytest.mark.asyncio
async def test_exercise_answer_uses_authenticated_user_id():
    rag_service = AsyncMock()
    account = DBUser(
        id="account-id",
        email="person@example.com",
        hashed_password="hash",
    )
    request = ExerciseAnswerRequest.model_validate(
        {"selected_option": "has", "user_id": "untrusted-client-id"}
    )
    expected = ExerciseAnswerResponse(
        correct=True,
        explanation="Correct.",
        exercise_attempt_id="attempt-id",
    )
    rag_service.process_exercise_answer.return_value = expected

    result = await submit_exercise_answer("exercise-id", request, rag_service, account)

    assert result is expected
    rag_service.process_exercise_answer.assert_awaited_once_with(
        "exercise-id",
        request,
        user_id="account-id",
    )
