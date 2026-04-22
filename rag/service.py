import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models import (
    ContextAssembly,
    DetectedMistakeExample,
    ExerciseAttempt,
    LessonResponse,
    UserText,
)
from rag.approaches.base import BaseApproach, StubApproachHandler
from rag.approaches.default import DefaultApproach
from rag.approaches.example_based import ExampleBasedApproach
from rag.approaches.rule_based import RuleBasedApproach
from rag.embedder import Embedder
from rag.mistake_exclusion import (
    delta_for_exercise_missed_target,
    delta_for_ingest_mistake_event,
    skip_example_for_qdrant,
)
from rag.pipelines.languagetool_pipeline import create_language_tool, process_text
from storage import repositories as repo
from vectorstore.qdrant_client import QdrantStore

_missing_artifacts = logging.getLogger("lexory.missing_artifacts")


class InMemoryTextStore:
    """Session-scoped text buffer (kept until LanguageTool finishes processing)."""

    def __init__(self):
        self.texts: Dict[str, dict] = {}

    def store(
        self,
        user_text_id: str,
        text: str,
        user_id: str,
        session_id: Optional[str],
        detected_at: datetime,
    ):
        self.texts[user_text_id] = {
            "user_text_id": user_text_id,
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "detected_at": detected_at.isoformat(),
        }


class RAGService:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        session_factory: async_sessionmaker[AsyncSession],
    ):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.session_factory = session_factory
        self.text_store = InMemoryTextStore()
        self.max_context_items = 10
        self.min_similarity_score = 0.5
        self.semantic_dedup_threshold = 0.9
        _llm = None
        if os.environ.get("GENERATOR_MODE", "llm").lower() == "llm":
            try:
                from llm.ollama_adapter import OllamaAdapter
                _llm = OllamaAdapter()
            except ImportError:
                pass
        self._approach_registry: Dict[str, BaseApproach] = {
            "rule_based": RuleBasedApproach(llm=_llm),
            "example_based": ExampleBasedApproach(),
            "default": DefaultApproach(),
        }
        self.lt_tool = create_language_tool()

    async def ingest_user_text(
        self, user_text: UserText
    ) -> tuple[str, str, List[dict]]:
        """
        Ingest user text, process through LanguageTool, deduplicate, store.
        Returns: (user_text_id, session_id, session_candidate_points).
        session_candidate_points: point-like dicts intercepted after category check
        and before semantic dedup, for use in query embedding.
        """
        session_id = str(uuid.uuid4())
        user_text_id = str(uuid.uuid4())

        self.text_store.store(
            user_text_id=user_text_id,
            text=user_text.text,
            user_id=user_text.user_id,
            session_id=session_id,
            detected_at=user_text.detected_at,
        )

        session_candidate_points: List[dict] = []

        try:
            events = process_text(
                text=user_text.text,
                user_id=user_text.user_id,
                user_text_id=user_text_id,
                session_id=session_id,
                detected_at=user_text.detected_at,
                embedder=self.embedder,
                source="raw_text",
                lt_tool=self.lt_tool,
            )

            example_points: List[dict] = []
            occurrence_points: List[dict] = []

            async with self.session_factory() as session:
                for event in events:
                    example_point, occurrence_point = await self._ingest_mistake_event(
                        session=session,
                        event=event,
                        user_text_id=user_text_id,
                        out_session_candidates=session_candidate_points,
                    )
                    if example_point:
                        example_points.append(example_point)
                    if occurrence_point:
                        occurrence_points.append(occurrence_point)

                if example_points:
                    self.qdrant.upsert("mistake_examples", example_points)
                if occurrence_points:
                    self.qdrant.upsert("mistake_occurrences", occurrence_points)

                await session.commit()
        except ImportError:
            raise RuntimeError(
                "LanguageTool is unavailable. Check your internet connection or try again later."
            ) from None

        return user_text_id, session_id, session_candidate_points

    @staticmethod
    def _user_filter(user_id: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Mandatory user_id filter for multi-tenant isolation.
        Merges with extra conditions. Use for all Qdrant searches.
        """
        base = {"user_id": user_id}
        if extra:
            return {**base, **extra}
        return base

    @staticmethod
    def _build_occurrence_payload(
        event: dict, lesson_artifact_id: Optional[str] = None
    ) -> dict:
        """Explicit whitelist for mistake_occurrences payload. No vectors."""
        payload = {
            "mistake_id": event["mistake_id"],
            "user_id": event["user_id"],
            "user_text_id": event["user_text_id"],
            "session_id": event["session_id"],
            "rule_id": event["rule_id"],
            "mistake_type": event["mistake_type"],
            "text": event["text"],
            "source": event["source"],
            "weight": event["weight"],
            "detected_at": event["detected_at"],
        }
        if lesson_artifact_id:
            payload["lesson_artifact_id"] = lesson_artifact_id
        return payload

    @staticmethod
    def _build_example_payload(event: dict) -> dict:
        """Explicit whitelist for mistake_examples payload. No vectors.
        example_id is set when persisting an example (SQL row id / FK target); omitted for
        session-candidate payloads before semantic dedup decides whether a row is written.
        """
        payload = {
            "mistake_id": event["mistake_id"],
            "user_id": event["user_id"],
            "user_text_id": event["user_text_id"],
            "session_id": event["session_id"],
            "rule_id": event["rule_id"],
            "mistake_type": event["mistake_type"],
            "rule_message": event.get("rule_message", ""),
            "text": event["text"],
            "source": event["source"],
            "weight": event["weight"],
            "detected_at": event["detected_at"],
        }
        eid = event.get("example_id")
        if eid is not None:
            payload["example_id"] = eid
        return payload

    async def _ingest_mistake_event(
        self,
        session: AsyncSession,
        event: dict,
        user_text_id: str,
        lesson_artifact_id: Optional[str] = None,
        out_session_candidates: Optional[List[dict]] = None,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """
        Deduplication workflow for mistake events.

        Qdrant point id for both mistake_occurrences and mistake_examples (when written)
        is always event["mistake_id"] (one id per detection). The mistake_examples payload
        also includes example_id (new UUID per stored example row) for SQL / FK separation
        from the occurrence row.

        Skip example_point (mistake_examples) when `skip_example_for_qdrant(event)` is true
        (exercise sources and Qdrant-excluded mistake types; see `rag.mistake_exclusion`).

        When out_session_candidates is provided and event passes the category check,
        append one session candidate (vectors + payload) for this request's lesson
        query embedding.

        Returns:
            (example_point, occurrence_point) - either can be None
        """
        event.setdefault("user_text_id", user_text_id)
        session_id_val = event.get("session_id", "")
        mistake_type = event["mistake_type"]
        context_vector = event["context_vector"]
        mistake_logic_vector = event["mistake_logic_vector"]
        skip_example = skip_example_for_qdrant(event)

        if skip_example:
            # Only create occurrence_point (mistake_occurrences)
            occurrence_point = {
                "id": event["mistake_id"],
                "vectors": {"mistake_logic": mistake_logic_vector},
                "payload": self._build_occurrence_payload(event, lesson_artifact_id),
            }
            occurrence_data = {
                "user_id": event["user_id"],
                "mistake_id": event["mistake_id"],
                "session_id": session_id_val,
                "user_text_id": user_text_id,
                "detected_at": event["detected_at"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            }
            if lesson_artifact_id:
                occurrence_data["lesson_artifact_id"] = lesson_artifact_id
            await repo.insert_occurrence(session, occurrence_data)
            await self._record_user_scoring_for_occurrence(
                session, event, user_text_id
            )
            return None, occurrence_point

        # Stage 1: Category check - does this mistake_type exist for this user?
        dummy_vector = [0.0] * 64
        existing_examples = self.qdrant.search(
            collection_name="mistake_examples",
            vector=None,
            limit=1,
            filters=self._user_filter(event["user_id"], {"mistake_type": mistake_type}),
            named_query={"vector_name": "mistake_logic", "vector": dummy_vector},
        )
        
        if not existing_examples:
            # No examples for this mistake_type - create both example and occurrence
            example_id = str(uuid.uuid4())
            event["example_id"] = example_id
            example_point = {
                "id": event["mistake_id"],
                "vectors": {
                    "mistake_logic": mistake_logic_vector,
                    "context": context_vector,
                },
                "payload": self._build_example_payload(event),
            }
            occurrence_point = {
                "id": event["mistake_id"],
                "vectors": {"mistake_logic": mistake_logic_vector},
                "payload": self._build_occurrence_payload(event, lesson_artifact_id),
            }

            occurrence_data = {
                "user_id": event["user_id"],
                "mistake_id": event["mistake_id"],
                "session_id": session_id_val,
                "user_text_id": user_text_id,
                "detected_at": event["detected_at"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
                "example_id": example_id,
            }
            await repo.insert_occurrence(session, occurrence_data)
            await self._record_user_scoring_for_occurrence(
                session, event, user_text_id
            )

            # Intercept: first example for this mistake_type is also a session candidate
            if out_session_candidates is not None:
                out_session_candidates.append({
                    "vectors": {
                        "mistake_logic": mistake_logic_vector,
                        "context": context_vector,
                    },
                    "payload": self._build_example_payload(event),
                })

            return example_point, occurrence_point

        # Intercept before Stage 2: candidate for session query embedding
        if out_session_candidates is not None:
            out_session_candidates.append({
                "vectors": {
                    "mistake_logic": mistake_logic_vector,
                    "context": context_vector,
                },
                "payload": self._build_example_payload(event),
            })

        # Stage 2: Semantic deduplication
        # Search by context vector with user_id + mistake_type filter
        similar_examples = self.qdrant.search(
            collection_name="mistake_examples",
            vector=None,
            limit=5,
            filters=self._user_filter(event["user_id"], {"mistake_type": mistake_type}),
            named_query={
                "vector_name": "context",
                "vector": context_vector,
            },
        )
        
        if similar_examples and similar_examples[0]["score"] > self.semantic_dedup_threshold:
            # High similarity - only create occurrence, not new example
            occurrence_point = {
                "id": event["mistake_id"],
                "vectors": {"mistake_logic": mistake_logic_vector},
                "payload": self._build_occurrence_payload(event, lesson_artifact_id),
            }

            await repo.insert_occurrence(session, {
                "user_id": event["user_id"],
                "mistake_id": event["mistake_id"],
                "session_id": session_id_val,
                "user_text_id": user_text_id,
                "detected_at": event["detected_at"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            })
            await self._record_user_scoring_for_occurrence(
                session, event, user_text_id
            )

            return None, occurrence_point

        # Low similarity or no results - create new example + occurrence
        example_id = str(uuid.uuid4())
        event["example_id"] = example_id
        example_point = {
            "id": event["mistake_id"],
            "vectors": {
                "mistake_logic": mistake_logic_vector,
                "context": context_vector,
            },
            "payload": self._build_example_payload(event),
        }
        occurrence_point = {
            "id": event["mistake_id"],
            "vectors": {"mistake_logic": mistake_logic_vector},
            "payload": self._build_occurrence_payload(event, lesson_artifact_id),
        }

        await repo.insert_occurrence(session, {
            "user_id": event["user_id"],
            "mistake_id": event["mistake_id"],
            "session_id": session_id_val,
            "user_text_id": user_text_id,
            "detected_at": event["detected_at"],
            "source": event["source"],
            "mistake_type": mistake_type,
            "rule_id": event["rule_id"],
            "example_id": example_id,
        })
        await self._record_user_scoring_for_occurrence(session, event, user_text_id)

        return example_point, occurrence_point

    async def _record_user_scoring_for_occurrence(
        self,
        session: AsyncSession,
        event: dict,
        session_or_exercise_id: str,
    ) -> None:
        d = delta_for_ingest_mistake_event(event)
        await repo.insert_user_scoring_event(
            session,
            {
                "user_id": event["user_id"],
                "rule_id": event.get("rule_id"),
                "mistake_type": event["mistake_type"],
                "mistake_id": event.get("mistake_id"),
                "session_or_exercise_id": session_or_exercise_id,
                "occurred_at": event["detected_at"],
                "delta": d,
            },
        )

    async def submit_and_lesson(
        self, text: str, user_id: str
    ) -> tuple[Optional[str], str, str, "LessonResponse", "ContextAssembly"]:
        """
        Combined ingest + lesson generation. Single flow trigger.
        Returns: (user_text_id, session_id, lesson_artifact_id, lesson, context)
        """
        session_id: str
        user_text_id: Optional[str] = None

        session_candidate_points: List[dict] = []
        if text and text.strip():
            user_text_id, session_id, session_candidate_points = await self.ingest_user_text(
                UserText(text=text.strip(), user_id=user_id)
            )
        else:
            session_id = str(uuid.uuid4())

        query_embedding, primary_example = await self._get_query_embedding_and_primary_example(
            session_candidate_points=session_candidate_points,
            user_id=user_id,
        )
        user_filter = {"user_id": user_id}

        if query_embedding is None:
            async with self.session_factory() as s:
                has_rows = await repo.has_any_user_scoring_event(s, user_id)
                has_graded = await repo.has_nonzero_user_scoring_event(s, user_id)
                has_positive = await repo.has_positive_clamped_mistake_type(s, user_id)

            if not has_rows:
                expl = (
                    "No usable session context. Please submit more text to generate a lesson."
                )
                title = "Cold start"
            elif has_graded and not has_positive:
                expl = (
                    "You have worked through your past struggles. "
                    "All tracked mistake types are now at zero. "
                    "Submit new text to focus on a different area if you like."
                )
                title = "Mastered your previous struggles"
            else:
                expl = (
                    "No usable session context. Please submit more text to generate a lesson."
                )
                title = "No usable session context"
            no_context_lesson = LessonResponse(
                topic=title,
                explanation=expl,
                exercises=[],
                approach_type="rule_based",
            )
            empty_context = ContextAssembly(
                detected_mistake_examples=[],
                long_term_dynamics=[],
                recently_used_explanations=[],
            )
            return (
                user_text_id,
                session_id,
                str(uuid.uuid4()),
                no_context_lesson,
                empty_context,
            )

        context = self._retrieve_staged_context(
            query_embedding, user_filter, primary_example
        )
        lesson = self._construct_lesson(context)
        artifact_id = await self._persist_lesson_artifact(
            lesson=lesson,
            user_id=user_id,
            session_id=session_id,
            context=context,
        )

        return user_text_id, session_id, artifact_id, lesson, context


    async def process_exercise_attempt(self, attempt: ExerciseAttempt) -> dict:
        exercise_attempt_id = str(uuid.uuid4())
        attempt_timestamp = datetime.now(timezone.utc)

        try:
            events = process_text(
                text=attempt.text,
                user_id=attempt.user_id,
                user_text_id=exercise_attempt_id,
                session_id=None,
                detected_at=attempt_timestamp,
                embedder=self.embedder,
                source="exercise_attempt",
                lt_tool=self.lt_tool,
            )

            tasks = []
            async with self.session_factory() as session:
                art = await repo.get_lesson_artifact_by_id(
                    session, attempt.lesson_artifact_id
                )
                if art is None:
                    _missing_artifacts.error("Lesson artifact with targeted mistake_type is missing")
                targets: Set[str] = set(art.mistake_types_covered or []) if art else set()
                detected: Set[str] = (
                    {e.get("mistake_type", "") for e in events} if events else set()
                )

                if events:
                    for event in events:
                        tasks.append({
                            "mistake_id": event.get("mistake_id"),
                            "is_correct": False,
                            "mistake_type": event.get("mistake_type"),
                            "rule_message": event.get("rule_message", ""),
                        })
                        _, occurrence_point = await self._ingest_mistake_event(
                            session=session,
                            event=event,
                            user_text_id=exercise_attempt_id,
                            lesson_artifact_id=attempt.lesson_artifact_id,
                        )
                        if occurrence_point:
                            self.qdrant.upsert("mistake_occurrences", [occurrence_point])
                else:
                    tasks.append({"mistake_id": None, "is_correct": True})

                for mt in targets - detected:
                    await repo.insert_user_scoring_event(
                        session,
                        {
                            "user_id": attempt.user_id,
                            "rule_id": None,
                            "mistake_type": mt,
                            "mistake_id": None,
                            "session_or_exercise_id": exercise_attempt_id,
                            "occurred_at": attempt_timestamp.isoformat(),
                            "delta": delta_for_exercise_missed_target(),
                        },
                    )

                await repo.insert_exercise_attempt(session, {
                    "exercise_attempt_id": exercise_attempt_id,
                    "lesson_artifact_id": attempt.lesson_artifact_id,
                    "user_id": attempt.user_id,
                    "attempt_timestamp": attempt_timestamp.isoformat(),
                })

                await session.commit()

            return {
                "exercise_attempt_id": exercise_attempt_id,
                "lesson_artifact_id": attempt.lesson_artifact_id,
                "attempt_timestamp": attempt_timestamp.isoformat(),
                "tasks": tasks,
            }
        except ImportError:
            raise RuntimeError(
                "LanguageTool is unavailable. Check your internet connection or try again later."
            ) from None

    @staticmethod
    def _exercise_feedback_from_event(event: dict) -> str:
        """Generate feedback from LanguageTool event."""
        mistake_type = event.get("mistake_type", "error")
        return f"Detected {mistake_type} mistake. Review the sentence structure."

    async def _get_fallback_point(self, user_id: str) -> tuple[Optional[List[float]], Optional[dict]]:
        """
        When there are no session_candidate_points: top-k mistake_type by clamped aggregate
        score (tie-break: latest occurred_at), then first mistake_examples point per type.
        """
        async with self.session_factory() as session:
            top_types = await repo.top_mistake_types_by_clamped_score(
                session, user_id, k=2
            )
        for mt in top_types:
            points = self.qdrant.scroll_by_mistake_type(
                collection_name="mistake_examples",
                user_id=user_id,
                mistake_type=mt,
                limit=1,
            )
            if points:
                p = points[0]
                vec = p.get("vectors", {}).get("context")
                payload = p.get("payload", {})
                if vec and len(vec) == 384:
                    return list(vec), payload

        return None, None

    async def _get_query_embedding_and_primary_example(
        self,
        session_candidate_points: List[dict],
        user_id: str,
    ) -> tuple[Optional[List[float]], Optional[dict]]:
        """
        Returns (query_embedding, primary_example) for lesson context.
        a) Session candidates: use first candidate's context_vector and payload.
        b) Fallback: _get_fallback_point (PostgreSQL occurrences with example_id → Qdrant).
        c) Neither: (None, None) → no usable session context.
        """
        points = session_candidate_points or []
        if points:
            first = points[0]
            vectors = first.get("vectors", {})
            context_vec = vectors.get("context")
            payload = first.get("payload", {})
            if context_vec and len(context_vec) == 384:
                return list(context_vec), payload

        return await self._get_fallback_point(user_id)

    def _payload_to_detected_example(self, payload: dict) -> DetectedMistakeExample:
        """Format point payload for ContextAssembly detected_mistake_examples."""
        mistake_type = payload.get("mistake_type", "")
        # Sentence span from the detector (payload "text"); lesson code reads examples[0].
        example_text = payload.get("text", "") or ""
        return DetectedMistakeExample(
            mistake_id=payload.get("mistake_id"),
            rule_id=str(payload.get("rule_id", "") or ""),
            mistake_type=mistake_type,
            description=self._mistake_type_to_description(mistake_type),
            examples=[example_text] if example_text else [],
            rule_message=str(payload.get("rule_message", "") or ""),
        )

    def _retrieve_staged_context(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
        primary_example: Optional[dict],
    ) -> ContextAssembly:
        detected_mistake_examples = (
            [self._payload_to_detected_example(primary_example)] if primary_example else []
        )
        recently_used_explanations = self._retrieve_lesson_artifacts(
            query_embedding, user_filter
        )
        long_term_dynamics = self._retrieve_learning_summaries(
            query_embedding, user_filter
        )

        return ContextAssembly(
            detected_mistake_examples=detected_mistake_examples,
            recently_used_explanations=recently_used_explanations,
            long_term_dynamics=long_term_dynamics,
        )

    @staticmethod
    def _mistake_type_to_description(mistake_type: str) -> str:
        """
        Convert mistake_type to human-readable description.
        For V1, uses simple formatting. V2 can add taxonomy labels.
        """
    
        return mistake_type.replace("_", " ").replace(".", " ").title()


    # Lesson Artifacts

    def _retrieve_lesson_artifacts(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
    ) -> List[dict]:
        user_id = user_filter.get("user_id")
        if not user_id:
            return []
        results = self.qdrant.search(
            collection_name="lesson_artifact_embeddings",
            vector=query_embedding,
            limit=10,
            filters=self._user_filter(user_id),
        )

        artifacts = []
        seen_artifact_ids: Set[str] = set()

        for result in results:
            if result["score"] < self.min_similarity_score:
                continue

            payload = result.get("payload", {})
            artifact_id = payload.get("artifact_id", result["id"])

            if artifact_id in seen_artifact_ids:
                continue

            seen_artifact_ids.add(artifact_id)
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "content": payload.get("content", ""),
                    "lesson_type": payload.get("lesson_type", ""),
                    "approach_type": payload.get("approach_type", ""),
                    "created_at": payload.get("created_at", ""),
                    "similarity_score": result["score"],
                }
            )

            if len(artifacts) >= 5:
                break

        return artifacts

    def _retrieve_learning_summaries(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
        limit: int = 5,
    ) -> List[dict]:
        """
        Retrieve relevant LearningSummaries for ContextAssembly long_term_dynamics.
        Uses learning_summary_embeddings (batch-generated summaries).
        """
        user_id = user_filter.get("user_id")
        if not user_id:
            return []

        results = self.qdrant.search(
            collection_name="learning_summary_embeddings",
            vector=query_embedding,
            limit=limit,
            filters=self._user_filter(user_id),
        )

        summaries = []
        for result in results:
            if result["score"] < self.min_similarity_score:
                continue
            payload = result.get("payload", {})
            summaries.append(
                {
                    "id": result["id"],
                    "content": payload.get("content", ""),
                    "window_type": payload.get("window_type", ""),
                    "scope": payload.get("scope", ""),
                    "similarity_score": result["score"],
                }
            )

        return summaries

    async def _persist_lesson_artifact(
        self,
        lesson: LessonResponse,
        user_id: str,
        session_id: Optional[str],
        context: ContextAssembly,
    ) -> str:
        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        mistake_types_covered = [
            p.mistake_type
            for p in context.detected_mistake_examples
            if p.mistake_type
        ]

        artifact_payload = {
            "artifact_id": artifact_id,
            "user_id": user_id,
            "session_id": session_id or "",
            "content": lesson.explanation,
            "exercises": lesson.exercises,
            "lesson_type": lesson.topic,
            "approach_type": lesson.approach_type,
            "mistake_types_covered": mistake_types_covered,
            "created_at": created_at.isoformat(),
        }

        async with self.session_factory() as session:
            await repo.upsert_artifact(session, artifact_payload)
            await session.commit()

        content_for_embedding = self._artifact_embedding_text(lesson)

        artifact_vector = self.embedder.embed_single(content_for_embedding)
        self.qdrant.upsert(
            collection_name="lesson_artifact_embeddings",
            points=[
                {
                    "id": artifact_id,
                    "vector": artifact_vector,
                    "payload": artifact_payload,
                }
            ],
        )
        return artifact_id


    @staticmethod
    def _artifact_embedding_text(lesson: LessonResponse) -> str:
        """Text embedded for lesson_artifact_embeddings — explanation only (LLM or rule-based)."""
        return (lesson.explanation or "").strip()


    def _construct_lesson(
        self, context: ContextAssembly) -> LessonResponse:
        used_approach_types = {
            artifact.get("approach_type", "")
            for artifact in context.recently_used_explanations
            if artifact.get("approach_type")
        }

        primary_mistake_context: Optional[DetectedMistakeExample] = (
            context.detected_mistake_examples[0] if context.detected_mistake_examples else None
        )

        # TODO: add long term dynamics
        primary_summary = (
            context.long_term_dynamics[0] if context.long_term_dynamics else None
        )

        topic = self._extract_topic(primary_mistake_context, primary_summary)
        approach_type = self._select_approach_type(used_approach_types)
        handler = self._get_approach_handler(approach_type)
        explanation = handler.build_explanation(context, topic)
        exercises = handler.generate_exercises(primary_mistake_context)

        # LLM override: use topic and approach_type from handler when available
        if getattr(handler, "_last_llm_result", None):
            result = handler._last_llm_result
            if result.get("topic"):
                topic = result["topic"]
            if result.get("approach_type"):
                approach_type = result["approach_type"]

        return LessonResponse(
            topic=topic,
            explanation=explanation,
            exercises=exercises,
            approach_type=approach_type,
        )

    def _extract_topic(
        self,
        primary_mistake_context: Optional[DetectedMistakeExample],
        primary_summary: Optional[dict],
    ) -> str:
        if primary_mistake_context:
            desc = primary_mistake_context.description
            if desc:
                return desc.split(".")[0].strip()[:100]

        if primary_summary:
            content = primary_summary.get("content", "")
            if content:
                return content.split(".")[0].strip()[:100]

        return "General Language Learning"

    def _select_approach_type(self, used_approach_types: Set[str]) -> str:
        available_types = [
            "rule_based",
            "example_based",
            "interactive",
            "visual",
            "contextual",
        ]

        for approach in available_types:
            if approach not in used_approach_types:
                return approach

        return available_types[0]

    def _get_approach_handler(self, approach_type: str) -> BaseApproach:
        mode = os.environ.get("GENERATOR_MODE", "llm").lower()

        if mode == "stub":
            return StubApproachHandler()
            
        return self._approach_registry.get(
            approach_type, self._approach_registry["default"]
        )