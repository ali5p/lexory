import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, TYPE_CHECKING

try:
    from language_tool_python import LanguageTool
except ImportError:
    LanguageTool = None

if TYPE_CHECKING:
    from batch.learning_summaries import InMemorySQLStore

from core.models import (
    ContextAssembly,
    ExerciseAttempt,
    LessonResponse,
    QueryRequest,
    QueryResponse,
    UserText,
)
from rag.approaches.base import BaseApproach
from rag.approaches.default import DefaultApproach
from rag.approaches.example_based import ExampleBasedApproach
from rag.approaches.rule_based import RuleBasedApproach
from rag.embedder import Embedder
from rag.pipelines.languagetool_pipeline import process_text
from vectorstore.qdrant_client import QdrantStore


class InMemoryTextStore:  # TODO replace with SQL storage
    """Simulates SQL storage for raw user text (source of truth)."""

    def __init__(self):
        self.texts: Dict[str, dict] = {}

    def store(
        self,
        user_text_id: str,
        text: str,
        user_id: str,
        session_id: Optional[str],
        timestamp: datetime,
    ):
        self.texts[user_text_id] = {
            "user_text_id": user_text_id,
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
        }

    def get_by_user(self, user_id: str) -> List[dict]:
        return [t for t in self.texts.values() if t["user_id"] == user_id]

    def get_by_user_session(self, user_id: str, session_id: Optional[str]) -> List[dict]:
        return [
            t
            for t in self.texts.values()
            if t["user_id"] == user_id and (session_id is None or t["session_id"] == session_id)
        ]


class InMemoryArtifactStore:  # TODO replace with SQL storage
    """Stores LessonArtifacts for non-repetition logic."""

    def __init__(self):
        self.artifacts: Dict[str, dict] = {}

    def upsert(self, artifact_id: str, data: dict):
        self.artifacts[artifact_id] = data

    def get_by_user(self, user_id: str) -> List[dict]:
        return [a for a in self.artifacts.values() if a["user_id"] == user_id]


class InMemoryOccurrenceStore:  # TODO replace with SQL storage
    """Stores mistake occurrences (authoritative)."""

    def __init__(self):
        self.occurrences: List[dict] = []

    def insert(self, occurrence: dict):
        self.occurrences.append(occurrence)


class RAGService:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        sql_store: Optional["InMemorySQLStore"] = None,
    ):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.sql_store = sql_store  # Optional: syncs to batch processor's SQL store
        self.text_store = InMemoryTextStore()
        self.artifact_store = InMemoryArtifactStore()
        self.occurrence_store = InMemoryOccurrenceStore()
        self.exercise_attempts: List[dict] = []  # Event register: metadata only (no full text)
        self.mistake_occurrences: List[dict] = []
        self.max_context_items = 10
        self.min_similarity_score = 0.5
        self.pattern_similarity_threshold = 0.65
        self.semantic_dedup_threshold = 0.98
        self.max_examples_per_pattern = 3
        self._approach_registry: Dict[str, BaseApproach] = {
            "rule_based": RuleBasedApproach(),
            "example_based": ExampleBasedApproach(),
            "default": DefaultApproach(),
        }
        # Initialize LanguageTool singleton (process-lifetime)
        if LanguageTool is not None:
            try:
                self.lt_tool = LanguageTool("en-US")
            except Exception:
                # Fallback if LanguageTool initialization fails
                self.lt_tool = None
        else:
            self.lt_tool = None

    def ingest_user_text(
        self, user_text: UserText, session_id: Optional[str] = None
    ) -> tuple[str, str]:
        user_text_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        self.text_store.store(
            user_text_id=user_text_id,
            text=user_text.text,
            user_id=user_text.user_id,
            session_id=session_id,
            timestamp=user_text.timestamp,
        )
        
        # Sync to SQL store for batch processing
        if self.sql_store is not None:
            self.sql_store.user_texts.append({
                "id": user_text_id,  # Batch processor joins on "id" field
                "user_text_id": user_text_id,  # Keep for clarity
                "text": user_text.text,
                "user_id": user_text.user_id,
                "session_id": session_id or "",
                "timestamp": user_text.timestamp.isoformat(),
            })

        # Process through LanguageTool pipeline
        try:
            events = process_text(
                text=user_text.text,
                user_id=user_text.user_id,
                user_text_id=user_text_id,
                session_id=session_id,
                timestamp=user_text.timestamp,
                embedder=self.embedder,
                source="raw_text",
                lt_tool=self.lt_tool,
            )
            
            # Batch process all events
            example_points = []
            occurrence_points = []
            
            for event in events:
                example_point, occurrence_point = self._ingest_mistake_event(
                    event=event,
                    user_text_id=user_text_id,
                )
                if example_point:
                    example_points.append(example_point)
                if occurrence_point:
                    occurrence_points.append(occurrence_point)
            
            # Batch upsert to Qdrant
            if example_points:
                self.qdrant.upsert("mistake_examples", example_points)
            if occurrence_points:
                self.qdrant.upsert("mistake_occurrences", occurrence_points)
        except ImportError:
            raise RuntimeError(
                "LanguageTool is unavailable. Check your internet connection or try again later."
            ) from None

        return document_id, user_text_id

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
            "timestamp": event["timestamp"],
        }
        if lesson_artifact_id:
            payload["lesson_artifact_id"] = lesson_artifact_id
        return payload

    @staticmethod
    def _build_example_payload(event: dict) -> dict:
        """Explicit whitelist for mistake_examples payload. No vectors."""
        return {
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
            "timestamp": event["timestamp"],
            "canonical_example": event["text"],
        }

    def _ingest_mistake_event(
        self,
        event: dict,
        user_text_id: str,
        lesson_artifact_id: Optional[str] = None,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """
        Deduplication workflow for mistake events.
        
        Skip example_point (mistake_examples) for:
        - source == "exercise_attempt" (AI-generated text, avoid semantic pollution)
        - mistake_type in ("other", "style") (excluded from lessons; only track in occurrences)
        
        Returns:
            (example_point, occurrence_point) - either can be None
        """
        mistake_type = event["mistake_type"]
        context_vector = event["context_vector"]
        mistake_logic_vector = event["mistake_logic_vector"]
        skip_example = (
            event.get("source") == "exercise_attempt"
            or mistake_type in ("other", "style")
        )

        if skip_example:
            # Only create occurrence_point (mistake_occurrences)
            occurrence_point = {
                "id": event["mistake_id"],
                "vectors": {"mistake_logic": mistake_logic_vector},
                "payload": self._build_occurrence_payload(event, lesson_artifact_id),
            }
            occurrence_data = {
                "mistake_id": event["mistake_id"],
                "user_text_id": user_text_id,
                "detected_at": event["timestamp"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            }
            if lesson_artifact_id:
                occurrence_data["lesson_artifact_id"] = lesson_artifact_id
            self.occurrence_store.insert(occurrence_data)
            if self.sql_store is not None:
                self.sql_store.mistake_occurrences.append(occurrence_data)
            return None, occurrence_point
        
        # Stage 1: Category check - does this mistake_type exist?
        dummy_vector = [0.0] * 64
        existing_examples = self.qdrant.search(
            collection_name="mistake_examples",
            vector=None,
            limit=1,
            filters={"mistake_type": mistake_type},
            named_query={"vector_name": "mistake_logic", "vector": dummy_vector},
        )
        
        if not existing_examples:
            # No examples for this mistake_type - create both example and occurrence
            example_id = str(uuid.uuid4())
            example_point = {
                "id": example_id,
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
            
            # Persist occurrence to SQL placeholder
            occurrence_data = {
                "mistake_id": event["mistake_id"],
                "user_text_id": user_text_id,
                "detected_at": event["timestamp"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            }
            self.occurrence_store.insert(occurrence_data)
            
            # Sync to SQL store for batch processing
            if self.sql_store is not None:
                self.sql_store.mistake_occurrences.append(occurrence_data)
            
            return example_point, occurrence_point
        
        # Stage 2: Semantic deduplication
        # Search by context vector with mistake_type filter
        similar_examples = self.qdrant.search(
            collection_name="mistake_examples",
            vector=None,
            limit=5,
            filters={"mistake_type": mistake_type},
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
            
            self.occurrence_store.insert({
                "mistake_id": event["mistake_id"],
                "user_text_id": user_text_id,
                "detected_at": event["timestamp"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            })
            
            return None, occurrence_point
        
        # Low similarity or no results - create new example + occurrence
        example_id = str(uuid.uuid4())
        example_point = {
            "id": example_id,
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
        
        self.occurrence_store.insert({
            "mistake_id": event["mistake_id"],
            "user_text_id": user_text_id,
            "detected_at": event["timestamp"],
            "source": event["source"],
            "mistake_type": mistake_type,
            "rule_id": event["rule_id"],
        })
        
        return example_point, occurrence_point

    def generate_lesson(self, query: QueryRequest) -> QueryResponse:
        query_embedding = self._build_query_embedding(
            user_id=query.user_id, session_id=query.session_id, fallback_query=query.query
        )
        user_filter = {"user_id": query.user_id}

        context = self._retrieve_staged_context(query_embedding, user_filter, query.session_id)
        lesson = self._construct_lesson(context, query.user_id)

        self._persist_lesson_artifact(
            lesson=lesson,
            user_id=query.user_id,
            session_id=query.session_id,
            context=context,
        )

        return QueryResponse(lesson=lesson, context=context)

    def process_exercise_attempt(self, attempt: ExerciseAttempt) -> dict:
        exercise_attempt_id = str(uuid.uuid4())
        
        # Process through LanguageTool pipeline
        try:
            events = process_text(
                text=attempt.text,
                user_id=attempt.user_id,
                user_text_id=exercise_attempt_id,
                session_id=None,
                timestamp=attempt.timestamp,
                embedder=self.embedder,
                source="exercise_attempt",
                lt_tool=self.lt_tool,
            )
            
            # Per-task results: one ✅/❌ per mistake unit (LanguageTool match)
            tasks = []
            if events:
                for event in events:
                    tasks.append({
                        "mistake_id": event.get("mistake_id"),
                        "is_correct": False,
                        "mistake_type": event.get("mistake_type"),
                        "rule_message": event.get("rule_message", ""),
                    })
                    # Ingest: only occurrence_point (no example_point for exercise_attempt)
                    _, occurrence_point = self._ingest_mistake_event(
                        event=event,
                        user_text_id=exercise_attempt_id,
                        lesson_artifact_id=attempt.lesson_artifact_id,
                    )
                    if occurrence_point:
                        self.qdrant.upsert("mistake_occurrences", [occurrence_point])
            else:
                tasks.append({"mistake_id": None, "is_correct": True})
            
            # Event register: metadata only (no full text storage)
            self._register_exercise_attempt({
                "exercise_attempt_id": exercise_attempt_id,
                "lesson_artifact_id": attempt.lesson_artifact_id,
                "user_id": attempt.user_id,
                "timestamp": attempt.timestamp.isoformat(),
            })
            
            return {
                "exercise_attempt_id": exercise_attempt_id,
                "lesson_artifact_id": attempt.lesson_artifact_id,
                "tasks": tasks,
            }
        except ImportError:
            raise RuntimeError(
                "LanguageTool is unavailable. Check your internet connection or try again later."
            ) from None

    def _register_exercise_attempt(self, metadata: dict) -> None:
        """Store exercise attempt metadata (no full text). V1: in-memory; V2: SQL."""
        self.exercise_attempts.append(metadata)

    @staticmethod
    def _exercise_feedback_from_event(event: dict) -> str:
        """Generate feedback from LanguageTool event."""
        mistake_type = event.get("mistake_type", "error")
        return f"Detected {mistake_type} mistake. Review the sentence structure."

    def _build_query_embedding(
        self, user_id: str, session_id: Optional[str], fallback_query: Optional[str]
    ) -> List[float]:
        """
        Build query embedding from recent mistake_types and session texts.
        
        Replaced legacy pattern descriptions with mistake_type labels from occurrences.
        """
        embedding_texts: List[str] = []

        # Get recent mistake_types for this user/session
        recent_types = self._get_recent_mistake_types(user_id, session_id, limit=self.max_context_items)
        mistake_type_labels = [
            self._mistake_type_to_description(mt) for mt in recent_types
        ]

        session_texts = [
            t["text"]
            for t in self.text_store.get_by_user_session(user_id, session_id)
            if t.get("source") != "exercise_attempt"
        ]

        if mistake_type_labels:
            embedding_texts.extend(mistake_type_labels[: self.max_context_items])

        if session_texts and (not embedding_texts or len(embedding_texts) < self.max_context_items):
            remaining = self.max_context_items - len(embedding_texts)
            embedding_texts.extend(session_texts[:remaining])

        if not embedding_texts and fallback_query:
            embedding_texts.append(fallback_query)

        if not embedding_texts:
            embedding_texts.append("")  # deterministic, no placeholders

        dedup_seen: Set[str] = set()
        ordered_unique: List[str] = []
        for text in embedding_texts:
            if text not in dedup_seen:
                dedup_seen.add(text)
                ordered_unique.append(text)

        combined = " ".join(ordered_unique[: self.max_context_items])
        return self.embedder.embed_single(combined)

    def _retrieve_staged_context(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
        session_id: Optional[str],
    ) -> ContextAssembly:
        detected_patterns = self._retrieve_mistake_patterns(query_embedding, user_filter)
        recently_used_explanations = self._retrieve_lesson_artifacts(
            query_embedding, user_filter, session_id
        )
        long_term_dynamics = self._retrieve_learning_summaries(
            query_embedding, user_filter
        )

        return ContextAssembly(
            detected_patterns=detected_patterns,
            recently_used_explanations=recently_used_explanations,
            long_term_dynamics=long_term_dynamics,
        )

    def _get_recent_mistake_types(
        self, user_id: str, session_id: Optional[str], limit: int = 10
    ) -> List[str]:
        """
        Get recent mistake_types from occurrences for the user/session.
        Used for lesson context and query embedding.
        
        Note: occurrences store user_text_id, not user_id directly. We need to
        join through user_texts to filter by user_id. For V1, we use a simpler
        approach: query mistake_occurrences Qdrant collection by user_id filter.
        """
        # Query Qdrant mistake_occurrences collection for this user
        # Use a dummy vector since we're filtering by user_id
        dummy_vector = [0.0] * 64
        results = self.qdrant.search(
            collection_name="mistake_occurrences",
            vector=None,
            limit=limit * 2,  # Get more to account for deduplication
            filters={"user_id": user_id},
            named_query={"vector_name": "mistake_logic", "vector": dummy_vector},
        )
        
        # Extract unique mistake_types, preserving order (most recent first via Qdrant ordering)
        seen_types: Set[str] = set()
        recent_types: List[str] = []
        for result in results:
            payload = result.get("payload", {})
            mistake_type = payload.get("mistake_type")
            if mistake_type and mistake_type not in seen_types:
                # Check session_id filter if provided
                if session_id is None or payload.get("session_id") == session_id:
                    seen_types.add(mistake_type)
                    recent_types.append(mistake_type)
                    if len(recent_types) >= limit:
                        break
        
        return recent_types

    @staticmethod
    def _mistake_type_to_description(mistake_type: str) -> str:
        """
        Convert mistake_type to human-readable description.
        For V1, uses simple formatting. V2 can add taxonomy labels.
        """
        # Replace underscores/dots with spaces and capitalize
        return mistake_type.replace("_", " ").replace(".", " ").title()

    def _retrieve_mistake_patterns(
        self, query_embedding: List[float], user_filter: Dict[str, str]
    ) -> List[dict]:
        """
        DEPRECATED: Legacy function name. Now retrieves mistake examples by mistake_type.
        
        Replaced pattern-based retrieval with mistake_type-based retrieval from mistake_examples.
        Uses recent mistake_types from occurrences to filter, then semantic search on context_vector.
        
        Returns structure compatible with existing lesson construction:
        - pattern_id → mistake_type (for backward compatibility)
        - description → mistake_type formatted as description
        - examples → canonical_example from mistake_examples
        """
        user_id = user_filter.get("user_id")
        if not user_id:
            return []

        # Get recent mistake_types for this user
        recent_types = self._get_recent_mistake_types(user_id, None, limit=5)
        if not recent_types:
            return []

        # Query mistake_examples for each mistake_type using semantic search
        patterns = []
        seen_types: Set[str] = set()

        for mistake_type in recent_types:
            if mistake_type in seen_types:
                continue
            seen_types.add(mistake_type)

            # Search mistake_examples by mistake_type with semantic similarity
            results = self.qdrant.search(
                collection_name="mistake_examples",
                vector=None,
                limit=1,
                filters={"mistake_type": mistake_type, "user_id": user_id},
                named_query={
                    "vector_name": "context",
                    "vector": query_embedding,
                },
            )

            if not results or results[0]["score"] < self.min_similarity_score:
                continue

            payload = results[0].get("payload", {})
            canonical_example = payload.get("canonical_example", payload.get("text", ""))
            rule_message = payload.get("rule_message", "")  # LanguageTool message for lesson context

            patterns.append(
                {
                    "pattern_id": mistake_type,  # Backward compatibility: use mistake_type as pattern_id
                    "description": self._mistake_type_to_description(mistake_type),
                    "examples": [canonical_example] if canonical_example else [],
                    "rule_message": rule_message,
                    "similarity_score": results[0]["score"],
                }
            )

            if len(patterns) >= 3:
                break

        return patterns


    # Lesson Artifacts

    def _retrieve_lesson_artifacts(
        self,
        query_embedding: List[float],
        user_filter: Dict[str, str],
        session_id: Optional[str] = None,
    ) -> List[dict]:
        results = self.qdrant.search(
            collection_name="lesson_artifact_embeddings",
            vector=query_embedding,
            limit=10,
            filters=user_filter,
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
            filters={"user_id": user_id},
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

    def _persist_lesson_artifact(
        self,
        lesson: LessonResponse,
        user_id: str,
        session_id: Optional[str],
        context: ContextAssembly,
    ) -> None:
        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        # Extract mistake_types from detected_patterns (pattern_id is now mistake_type)
        mistake_types_covered = [
            p.get("pattern_id")  # pattern_id now contains mistake_type for backward compatibility
            for p in context.detected_patterns
            if p.get("pattern_id")
        ]

        pedagogy_tags = self._pedagogy_tags_for_lesson()

        artifact_payload = {
            "artifact_id": artifact_id,
            "user_id": user_id,
            "session_id": session_id or "",
            "content": lesson.explanation,
            "lesson_type": lesson.topic,
            "approach_type": lesson.approach_type,
            "mistake_types_covered": mistake_types_covered,  # Replaced patterns_covered
            "patterns_covered": mistake_types_covered,  # Backward compatibility for batch processor
            "pedagogy_tags": pedagogy_tags,
            "created_at": created_at.isoformat(),
        }

        self.artifact_store.upsert(artifact_id, artifact_payload)
        
        # Sync to SQL store for batch processing
        if self.sql_store is not None:
            self.sql_store.lesson_artifacts.append(artifact_payload)

        content_for_embedding = self._artifact_embedding_text(
            lesson=lesson,
            patterns_covered=mistake_types_covered,  # Still uses old param name for internal consistency
            pedagogy_tags=pedagogy_tags,
        )

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

    @staticmethod
    def _pedagogy_tags_for_lesson() -> List[str]:
        return ["error_explanation", "guided_practice"]

    @staticmethod
    def _artifact_embedding_text(
        lesson: LessonResponse,
        patterns_covered: List[str],
        pedagogy_tags: List[str],
    ) -> str:
        patterns_text = ", ".join(patterns_covered) if patterns_covered else "no-specific-patterns"
        tags_text = ", ".join(pedagogy_tags) if pedagogy_tags else "general"

        return (
            f"Lesson topic: {lesson.topic}. "
            f"Explanation: {lesson.explanation}. "
            f"Taught patterns: {patterns_text}. "
            f"Pedagogy: {tags_text}."
        )


    def _construct_lesson(
        self, context: ContextAssembly, user_id: str  # reserved for personalization
    ) -> LessonResponse:
        used_approach_types = {
            artifact.get("approach_type", "")
            for artifact in context.recently_used_explanations
            if artifact.get("approach_type")
        }

        primary_pattern = (
            context.detected_patterns[0] if context.detected_patterns else None
        )

        # TODO: add long term dynamics
        primary_summary = (
            context.long_term_dynamics[0] if context.long_term_dynamics else None
        )

        topic = self._extract_topic(primary_pattern, primary_summary)
        approach_type = self._select_approach_type(used_approach_types)
        handler = self._get_approach_handler(approach_type)
        explanation = handler.build_explanation(context, topic)
        exercises = handler.generate_exercises(primary_pattern)

        return LessonResponse(
            topic=topic,
            explanation=explanation,
            exercises=exercises,
            approach_type=approach_type,
        )

    def _extract_topic(
        self, primary_pattern: Optional[dict], primary_summary: Optional[dict]
    ) -> str:
        if primary_pattern:
            desc = primary_pattern.get("description", "")
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
        return self._approach_registry.get(approach_type, self._approach_registry["default"])