"""Tests for deterministic exercise answer processing."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.context import current_user_id_ctx
from core.exercises import ExerciseAnswerRequest
from rag.embedder import Embedder
from rag.service import RAGService
from storage.models import Exercise, LessonArtifact
from vectorstore.qdrant_client import QdrantStore


@pytest.fixture
def mock_qdrant():
    return Mock(spec=QdrantStore)


@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=Embedder)
    embedder.embed_single = Mock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def mock_session_factory():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory = Mock()
    factory.return_value = ctx
    return factory, session


@pytest.fixture
def rag_service(mock_qdrant, mock_embedder, mock_session_factory):
    factory, _ = mock_session_factory
    return RAGService(mock_qdrant, mock_embedder, factory)


@pytest.fixture
def authenticated_user_context():
    token = current_user_id_ctx.set("user-1")
    try:
        yield
    finally:
        current_user_id_ctx.reset(token)


@pytest.mark.asyncio
async def test_process_exercise_answer_correct_scores_negative(
    rag_service: RAGService, mock_session_factory, authenticated_user_context
):
    _, session = mock_session_factory
    exercise = Exercise(
        exercise_id="ex-1",
        artifact_id="art-1",
        sort_order=0,
        type="multiple_choice",
        mistake_type="subject_verb_agreement",
        source_sentence="She walk.",
        payload={
            "type": "multiple_choice",
            "sentence": "She ___ to school.",
            "options": ["walks", "walk"],
        },
        answer_key={
            "type": "multiple_choice",
            "correct_option": "walks",
            "explanation_on_success": "Good!",
            "explanation_on_error": "No.",
        },
        created_at="2026-01-01T00:00:00+00:00",
    )
    artifact = LessonArtifact(
        artifact_id="art-1",
        user_id="user-1",
        session_id="sess-1",
        topic="SVA",
        explanation="Use -s.",
        approach_type="rule_based",
        mistake_type="subject_verb_agreement",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with patch("rag.service.repo.get_exercise_by_id", AsyncMock(return_value=exercise)), patch(
        "rag.service.repo.get_lesson_artifact_by_id", AsyncMock(return_value=artifact)
    ), patch(
        "rag.service.repo.insert_user_scoring_event", AsyncMock()
    ) as scoring, patch(
        "rag.service.repo.insert_exercise_attempt", AsyncMock()
    ), patch.object(
        rag_service, "_refresh_user_mistake_type_stats", AsyncMock()
    ):
        result = await rag_service.process_exercise_answer(
            "ex-1",
            ExerciseAnswerRequest(selected_option="walks"),
        )

    assert result.correct is True
    assert result.explanation == "Good!"
    scoring.assert_awaited_once()
    assert scoring.await_args.args[1]["delta"] == -0.5


@pytest.mark.asyncio
async def test_process_exercise_answer_wrong_scores_plus_one(
    rag_service: RAGService, mock_session_factory, authenticated_user_context
):
    exercise = Exercise(
        exercise_id="ex-1",
        artifact_id="art-1",
        sort_order=0,
        type="multiple_choice",
        mistake_type="articles",
        source_sentence="I ate apple.",
        payload={
            "type": "multiple_choice",
            "sentence": "I ate ___ apple.",
            "options": ["an", "a"],
        },
        answer_key={
            "type": "multiple_choice",
            "correct_option": "an",
            "explanation_on_success": "Good!",
            "explanation_on_error": "No.",
        },
        created_at="2026-01-01T00:00:00+00:00",
    )
    artifact = LessonArtifact(
        artifact_id="art-1",
        user_id="user-1",
        session_id="sess-1",
        topic="Articles",
        explanation="Use an before vowels.",
        approach_type="rule_based",
        mistake_type="articles",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with patch("rag.service.repo.get_exercise_by_id", AsyncMock(return_value=exercise)), patch(
        "rag.service.repo.get_lesson_artifact_by_id", AsyncMock(return_value=artifact)
    ), patch(
        "rag.service.repo.insert_user_scoring_event", AsyncMock()
    ) as scoring, patch(
        "rag.service.repo.insert_exercise_attempt", AsyncMock()
    ), patch.object(
        rag_service, "_refresh_user_mistake_type_stats", AsyncMock()
    ):
        result = await rag_service.process_exercise_answer(
            "ex-1",
            ExerciseAnswerRequest(selected_option="a"),
        )

    assert result.correct is False
    assert scoring.await_args.args[1]["delta"] == 1.0


@pytest.mark.asyncio
async def test_service_rejects_calls_without_authenticated_context(rag_service):
    with pytest.raises(RuntimeError, match="authenticated user context is required"):
        await rag_service.submit_and_lesson("")

    with pytest.raises(RuntimeError, match="authenticated user context is required"):
        await rag_service.process_exercise_answer(
            "exercise-id",
            ExerciseAnswerRequest(selected_option="answer"),
        )


@pytest.mark.asyncio
async def test_exercise_answer_uses_authenticated_tenant_for_ownership(
    rag_service: RAGService,
    authenticated_user_context,
):
    context_token = current_user_id_ctx.set("another-user")
    exercise = Exercise(
        exercise_id="ex-1",
        artifact_id="art-1",
        sort_order=0,
        type="multiple_choice",
        mistake_type="articles",
        source_sentence="I ate apple.",
        payload={"type": "multiple_choice", "sentence": "I ate ___ apple."},
        answer_key={"type": "multiple_choice", "correct_option": "an"},
        created_at="2026-01-01T00:00:00+00:00",
    )
    artifact = LessonArtifact(
        artifact_id="art-1",
        user_id="user-1",
        session_id="sess-1",
        topic="Articles",
        explanation="Use an before vowels.",
        approach_type="rule_based",
        mistake_type="articles",
        created_at="2026-01-01T00:00:00+00:00",
    )

    try:
        with patch(
            "rag.service.repo.get_exercise_by_id",
            AsyncMock(return_value=exercise),
        ), patch(
            "rag.service.repo.get_lesson_artifact_by_id",
            AsyncMock(return_value=artifact),
        ):
            with pytest.raises(ValueError, match="Exercise not found"):
                await rag_service.process_exercise_answer(
                    "ex-1",
                    ExerciseAnswerRequest(selected_option="an"),
                )
    finally:
        current_user_id_ctx.reset(context_token)
