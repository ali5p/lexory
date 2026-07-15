import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.lesson_artifact import LessonArtifactRecord
from core.models import (
    ContextAssembly,
    DetectedMistakeExample,
    DetectedMistakeItem,
    ExerciseAttempt,
    LessonContent,
    LessonItemResponse,
    LessonResponse,
    LessonTarget,
    SubmitResponse,
    UserText,
)
from rag.approach_selection import ApproachSelector
from rag.approaches.base import BaseApproach
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
        self.max_primary_mistakes = 3
        self.supplemental_every_n_submits = 2
        self.min_similarity_score = 0.5
        self.semantic_dedup_threshold = 0.9
        from llm.factory import build_llm

        llm = build_llm()
        self._approach_registry: Dict[str, BaseApproach] = {
            "rule_based": RuleBasedApproach(llm=llm),
            "example_based": ExampleBasedApproach(llm=llm),
            "default": DefaultApproach(llm=llm),
        }
        self.selector = ApproachSelector(
            approaches=["rule_based", "example_based"],
            baseline="rule_based",
        )
        self.lt_tool = create_language_tool()

    async def ingest_user_text(
        self, user_text: UserText
    ) -> tuple[str, str, List[dict], List[dict]]:
        """
        Ingest user text, process through LanguageTool, deduplicate, store.
        Returns: (user_text_id, session_id, session_candidate_points, detected_mistakes).
        session_candidate_points: point-like dicts intercepted after category check
        and before semantic dedup, for use in query embedding.
        detected_mistakes: every LanguageTool event from this submission (for API).
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
        detected_mistakes: List[dict] = []

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
                    detected_mistakes.append(self._event_to_detected_mistake(event))
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

        return user_text_id, session_id, session_candidate_points, detected_mistakes

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

    async def _refresh_user_mistake_type_stats(self, user_id: str) -> None:
        async with self.session_factory() as session:
            await repo.recompute_user_mistake_type_stats(session, user_id)
            await session.commit()

    async def submit_and_lesson(self, text: str, user_id: str) -> SubmitResponse:
        """Combined ingest + lesson generation. Single flow trigger."""
        session_id: str
        user_text_id: Optional[str] = None
        detected_mistakes_raw: List[dict] = []

        session_candidate_points: List[dict] = []
        if text and text.strip():
            (
                user_text_id,
                session_id,
                session_candidate_points,
                detected_mistakes_raw,
            ) = await self.ingest_user_text(
                UserText(text=text.strip(), user_id=user_id)
            )
        else:
            session_id = str(uuid.uuid4())

        query_points = (
            await self._get_query_points_and_primary_examples(
                session_candidate_points=session_candidate_points,
                user_id=user_id,
            )
        )
        query_points = self._unique_query_points_by_lesson_target(query_points)
        user_filter = {"user_id": user_id}
        selected_mistake_ids = self._selected_mistake_ids(query_points)
        detected_mistakes = self._build_detected_mistake_items(
            detected_mistakes_raw,
            selected_mistake_ids,
        )

        if not query_points:
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
            no_context_lesson = LessonContent(
                topic=title,
                explanation=expl,
                exercises=[],
            )
            response = SubmitResponse(
                user_text_id=user_text_id,
                session_id=session_id,
                detected_mistakes=detected_mistakes,
                lesson_items=[
                    LessonItemResponse(
                        lesson_artifact_id=None,
                        target=None,
                        lesson=no_context_lesson,
                    )
                ],
            )
        else:
            lesson_items = await self._generate_atomic_lesson_items(
                query_points=query_points,
                user_id=user_id,
                session_id=session_id,
                user_filter=user_filter,
            )
            if session_candidate_points:
                lesson_items = await self._maybe_add_supplemental_lesson_item(
                    items=lesson_items,
                    query_points=query_points,
                    user_id=user_id,
                    session_id=session_id,
                    user_filter=user_filter,
                )
            response = SubmitResponse(
                user_text_id=user_text_id,
                session_id=session_id,
                detected_mistakes=detected_mistakes,
                lesson_items=lesson_items,
            )

        await self._refresh_user_mistake_type_stats(user_id)
        return response


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
                targets: Set[str] = (
                    {art.mistake_type} if art and art.mistake_type else set()
                )
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
                    "origin_session_id": art.session_id if art else "",
                })

                await session.commit()

            await self._refresh_user_mistake_type_stats(attempt.user_id)

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

    async def _get_fallback_points(self, user_id: str) -> List[dict]:
        """
        When there are no session_candidate_points: top-k mistake_type by trend-aware
        priority stats (user_mistake_type_stats), then first mistake_examples point per type.
        """
        out: List[dict] = []
        async with self.session_factory() as session:
            top_types = await repo.top_priority_mistake_types(
                session, user_id, k=self.max_primary_mistakes
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
                if vec and len(vec) == 384:
                    out.append(p)

        return out

    async def _get_query_points_and_primary_examples(
        self,
        session_candidate_points: List[dict],
        user_id: str,
    ) -> List[dict]:
        """
        Returns selected point-like dicts for lesson generation.
        a) Session candidates: select top-k current-session mistake types by aggregate
           user score.
        b) Fallback: _get_fallback_points (priority stats → Qdrant mistake_examples).
        c) Neither: [] → no usable session context.
        """
        points = session_candidate_points or []
        if points:
            return await self._select_scored_session_candidates(
                points=points,
                user_id=user_id,
                limit=self.max_primary_mistakes,
            )

        return await self._get_fallback_points(user_id)

    @staticmethod
    def _lesson_target_key(payload: dict) -> str:
        mistake_type = str(payload.get("mistake_type", "") or "")
        rule_id = str(payload.get("rule_id", "") or "")
        if mistake_type == "unmapped" and rule_id:
            return f"{mistake_type}:{rule_id}"
        return mistake_type or rule_id or str(payload.get("mistake_id", ""))

    @classmethod
    def _unique_query_points_by_lesson_target(cls, points: List[dict]) -> List[dict]:
        unique: List[dict] = []
        seen: Set[str] = set()
        for point in points:
            payload = point.get("payload", {})
            key = cls._lesson_target_key(payload) or str(id(point))
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        return unique

    async def _select_scored_session_candidates(
        self,
        points: List[dict],
        user_id: str,
        limit: int,
    ) -> List[dict]:
        """
        Pick current-session candidates ranked by the user's long-term scores.
        Only candidates from this submission are eligible; scoring decides priority.
        """
        grouped: dict[str, dict] = {}
        for idx, point in enumerate(points):
            payload = point.get("payload", {})
            mistake_type = str(payload.get("mistake_type", "") or "")
            rule_id = str(payload.get("rule_id", "") or "")
            lesson_key = self._lesson_target_key(payload)
            context_vec = point.get("vectors", {}).get("context")
            if not lesson_key or not context_vec or len(context_vec) != 384:
                continue
            group = grouped.setdefault(
                lesson_key,
                {
                    "point": point,
                    "count": 0,
                    "last_idx": idx,
                    "mistake_type": mistake_type,
                    "rule_id": rule_id,
                },
            )
            group["count"] += 1
            # Within one lesson target, use the latest occurrence in the submitted text.
            group["point"] = point
            group["last_idx"] = idx

        if not grouped:
            return []

        mistake_types = [
            data["mistake_type"] for data in grouped.values() if data["mistake_type"]
        ]
        async with self.session_factory() as session:
            scores = await repo.clamped_scores_by_mistake_type(
                session,
                user_id,
                mistake_types,
            )
            rule_scores = await repo.clamped_scores_by_mistake_type_and_rule(
                session,
                user_id,
                mistake_types,
            )

        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                self._score_for_lesson_target(item[1], scores, rule_scores)[0],
                item[1]["count"],
                self._score_for_lesson_target(item[1], scores, rule_scores)[1],
                item[1]["last_idx"],
            ),
            reverse=True,
        )
        return [data["point"] for _, data in ranked[:limit]]

    @staticmethod
    def _score_for_lesson_target(
        target: dict,
        type_scores: dict[str, tuple[float, str]],
        rule_scores: dict[tuple[str, str], tuple[float, str]],
    ) -> tuple[float, str]:
        mistake_type = str(target.get("mistake_type", "") or "")
        rule_id = str(target.get("rule_id", "") or "")
        if mistake_type == "unmapped" and rule_id:
            return rule_scores.get((mistake_type, rule_id), (0.0, ""))
        return type_scores.get(mistake_type, (0.0, ""))

    @staticmethod
    def _event_to_detected_mistake(event: dict) -> dict:
        return {
            "mistake_id": event["mistake_id"],
            "rule_id": str(event.get("rule_id", "") or ""),
            "mistake_type": event["mistake_type"],
            "text": str(event.get("text", "") or ""),
            "rule_message": str(event.get("rule_message", "") or ""),
        }

    @staticmethod
    def _selected_mistake_ids(query_points: List[dict]) -> Set[str]:
        selected: Set[str] = set()
        for point in query_points:
            mistake_id = str(point.get("payload", {}).get("mistake_id", "") or "")
            if mistake_id:
                selected.add(mistake_id)
        return selected

    @staticmethod
    def _build_detected_mistake_items(
        detected_mistakes: List[dict],
        selected_mistake_ids: Set[str],
    ) -> List[DetectedMistakeItem]:
        return [
            DetectedMistakeItem(
                mistake_id=item["mistake_id"],
                rule_id=item["rule_id"],
                mistake_type=item["mistake_type"],
                text=item["text"],
                rule_message=item.get("rule_message", ""),
                selected_for_lesson=item["mistake_id"] in selected_mistake_ids,
            )
            for item in detected_mistakes
        ]

    @staticmethod
    def _payload_to_lesson_target(payload: dict) -> LessonTarget:
        return LessonTarget(
            mistake_id=str(payload.get("mistake_id", "") or ""),
            rule_id=str(payload.get("rule_id", "") or ""),
            mistake_type=str(payload.get("mistake_type", "") or ""),
            text=str(payload.get("text", "") or ""),
            rule_message=str(payload.get("rule_message", "") or ""),
        )

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

    def _retrieve_similar_examples(
        self,
        query_embedding: List[float],
        user_id: str,
        mistake_type: str,
        exclude_mistake_id: Optional[str] = None,
        limit: int = 3,
    ) -> List[dict]:
        """The user's own prior sentences with the same mistake_type, ranked by
        sentence (context) similarity. Personalized fuel for the inductive
        (example_based) approach. Small over-fetch to survive the score threshold,
        the current-mistake exclusion, and text dedup.
        """
        if not user_id or not mistake_type:
            return []
        filters = self._user_filter(user_id, {"mistake_type": mistake_type})
        results = self.qdrant.search(
            collection_name="mistake_examples",
            vector=None,
            limit=limit * 2,
            filters=filters,
            named_query={"vector_name": "context", "vector": query_embedding},
        )

        examples: List[dict] = []
        seen_texts: Set[str] = set()
        for result in results:
            if result["score"] < self.min_similarity_score:
                continue
            payload = result.get("payload", {})
            if exclude_mistake_id and payload.get("mistake_id") == exclude_mistake_id:
                continue
            text = (payload.get("text", "") or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            examples.append(
                {
                    "text": text,
                    "rule_message": str(payload.get("rule_message", "") or ""),
                }
            )
            if len(examples) >= limit:
                break
        return examples

    def _retrieve_staged_context(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
        primary_examples: List[dict],
    ) -> ContextAssembly:
        detected_mistake_examples = [
            self._payload_to_detected_example(payload)
            for payload in primary_examples
            if payload
        ]

        return ContextAssembly(
            detected_mistake_examples=detected_mistake_examples,
        )

    async def _generate_atomic_lesson_items(
        self,
        query_points: List[dict],
        user_id: str,
        session_id: str,
        user_filter: Dict[str, str],
    ) -> List[LessonItemResponse]:
        items: List[LessonItemResponse] = []
        seen_lesson_targets: Set[str] = set()
        for point in query_points:
            lesson_key = self._lesson_target_key(point.get("payload", {}))
            if lesson_key in seen_lesson_targets:
                continue
            item = await self._generate_lesson_item_from_point(
                point=point,
                user_id=user_id,
                session_id=session_id,
                user_filter=user_filter,
            )
            if item is None:
                continue
            if lesson_key:
                seen_lesson_targets.add(lesson_key)
            items.append(item)
        return items

    async def _maybe_add_supplemental_lesson_item(
        self,
        *,
        items: List[LessonItemResponse],
        query_points: List[dict],
        user_id: str,
        session_id: str,
        user_filter: Dict[str, str],
    ) -> List[LessonItemResponse]:
        """Add one practice item from priority MTs not covered by this submit (throttled)."""
        current_mts = {
            str(p.get("payload", {}).get("mistake_type", "") or "") for p in query_points
        }
        current_mts.discard("")

        async with self.session_factory() as session:
            if not await repo.should_offer_supplemental_practice(
                session,
                user_id,
                exclude_mistake_types=current_mts,
                every_n_submits=self.supplemental_every_n_submits,
            ):
                return items
            priority_mts = await repo.top_priority_mistake_types(
                session, user_id, k=5
            )

        for mt in priority_mts:
            if mt in current_mts:
                continue
            points = self.qdrant.scroll_by_mistake_type(
                collection_name="mistake_examples",
                user_id=user_id,
                mistake_type=mt,
                limit=1,
            )
            if not points:
                continue
            point = points[0]
            vec = point.get("vectors", {}).get("context")
            if not vec or len(vec) != 384:
                continue
            item = await self._generate_lesson_item_from_point(
                point=point,
                user_id=user_id,
                session_id=session_id,
                user_filter=user_filter,
            )
            if item is not None:
                return items + [item]
        return items

    async def _generate_lesson_item_from_point(
        self,
        *,
        point: dict,
        user_id: str,
        session_id: str,
        user_filter: Dict[str, str],
    ) -> Optional[LessonItemResponse]:
        context_vec = point.get("vectors", {}).get("context")
        payload = point.get("payload", {})
        if not context_vec or len(context_vec) != 384 or not payload:
            return None

        mistake_type = str(payload.get("mistake_type", "") or "")
        async with self.session_factory() as session:
            example_count = await repo.count_examples_by_mistake_type(
                session, user_id, mistake_type
            )
            selection_index = await repo.count_lessons_by_mistake_type(
                session, user_id, mistake_type
            )
            approach_scores = await repo.approach_effectiveness_scores_by_mistake_type(
                session,
                user_id,
                mistake_type,
                self.selector.approaches,
                comparison_min_example_count=self.selector.COMPARISON_MIN_EXAMPLE_COUNT,
            )
        approach_type = self.selector.select(
            example_count=example_count,
            selection_index=selection_index,
            scores=approach_scores,
        )
        is_contrast_lesson = self.selector.is_contrast_lesson(
            example_count=example_count,
            selection_index=selection_index,
            scores=approach_scores,
        )

        item_context = self._retrieve_staged_context(
            list(context_vec),
            user_filter,
            [payload],
        )
        if approach_type == "example_based":
            item_context.similar_past_examples = self._retrieve_similar_examples(
                query_embedding=list(context_vec),
                user_id=user_id,
                mistake_type=mistake_type,
                exclude_mistake_id=str(payload.get("mistake_id", "") or "") or None,
            )

        lesson = self._construct_lesson(item_context, approach_type)
        artifact_id = await self._persist_lesson_artifact(
            lesson=lesson,
            user_id=user_id,
            session_id=session_id,
            context=item_context,
            query_embedding=list(context_vec),
            selection_index=selection_index,
            is_contrast_lesson=is_contrast_lesson,
            example_count_at_generation=example_count,
        )
        return LessonItemResponse(
            lesson_artifact_id=artifact_id,
            target=self._payload_to_lesson_target(payload),
            lesson=LessonContent(
                topic=lesson.topic,
                explanation=lesson.explanation,
                exercises=lesson.exercises,
            ),
        )

    @staticmethod
    def _mistake_type_to_description(mistake_type: str) -> str:
        """
        Convert mistake_type to human-readable description.
        For V1, uses simple formatting. V2 can add taxonomy labels.
        """
        return mistake_type.replace("_", " ").replace(".", " ").title()

    async def _persist_lesson_artifact(
        self,
        lesson: LessonResponse,
        user_id: str,
        session_id: Optional[str],
        context: ContextAssembly,
        query_embedding: List[float],
        selection_index: int = 0,
        is_contrast_lesson: bool = False,
        example_count_at_generation: int = 0,
    ) -> str:
        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        primary_mistake = (
            context.detected_mistake_examples[0]
            if context.detected_mistake_examples
            else None
        )
        record = LessonArtifactRecord.for_lesson(
            artifact_id=artifact_id,
            lesson=lesson,
            user_id=user_id,
            session_id=session_id,
            primary_mistake=primary_mistake,
            created_at=created_at,
            selection_index=selection_index,
            is_contrast_lesson=is_contrast_lesson,
            example_count_at_generation=example_count_at_generation,
        )

        async with self.session_factory() as session:
            await repo.upsert_artifact(session, record.sql_row())
            await session.commit()

        explanation_text = self._artifact_embedding_text(lesson)
        explanation_vector = self.embedder.embed_single(explanation_text)
        self.qdrant.upsert(
            collection_name="lesson_artifact_points",
            points=[
                {
                    "id": artifact_id,
                    "vectors": {
                        "mistake_context": query_embedding,
                        "explanation": explanation_vector,
                    },
                    "payload": record.qdrant_payload(),
                }
            ],
        )
        return artifact_id


    @staticmethod
    def _artifact_embedding_text(lesson: LessonResponse) -> str:
        """Text embedded as the lesson_artifact_points explanation vector."""
        return (lesson.explanation or "").strip()


    def _construct_lesson(
        self, context: ContextAssembly, approach_type: str
    ) -> LessonResponse:
        primary_mistake_context: Optional[DetectedMistakeExample] = (
            context.detected_mistake_examples[0] if context.detected_mistake_examples else None
        )

        topic = self._extract_topic(primary_mistake_context)
        handler = self._get_approach_handler(approach_type)
        explanation = handler.build_explanation(context, topic)
        exercises = handler.generate_exercises(primary_mistake_context)

        # Use the LLM's chosen topic when available. The stored approach_type is the
        # selected teaching approach (rule_based/example_based) so downstream scoring
        # can attribute outcomes to the approach, not to generation status.
        if getattr(handler, "_last_llm_result", None):
            result = getattr(handler, "_last_llm_result", None)
            if result and result.get("topic"):
                topic = result["topic"]

        return LessonResponse(
            topic=topic,
            explanation=explanation,
            exercises=exercises,
            approach_type=approach_type,
        )

    def _extract_topic(
        self,
        primary_mistake_context: Optional[DetectedMistakeExample],
    ) -> str:
        if primary_mistake_context:
            desc = primary_mistake_context.description
            if desc:
                return desc.split(".")[0].strip()[:100]

        return "General Language Learning"

    def _get_approach_handler(self, approach_type: str) -> BaseApproach:
        return self._approach_registry.get(
            approach_type, self._approach_registry["default"]
        )