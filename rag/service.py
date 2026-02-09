import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

try:
    from language_tool_python import LanguageTool
except ImportError:
    LanguageTool = None

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


class InMemoryPatternStore:  # TODO replace with SQL + batch consolidation
    """
    DEPRECATED: Pattern subsystem replaced by mistake_type + named vectors.
    
    This store is no longer actively populated in the new architecture.
    Pattern creation (_create_or_update_mistake_pattern) still runs but is disconnected
    from the active event-driven flow. Will be removed in V2.
    
    Legacy: Stores MistakePatterns for the user.
    """

    def __init__(self):
        self.patterns: Dict[str, dict] = {}

    def upsert(self, pattern_id: str, data: dict):
        self.patterns[pattern_id] = data

    def get_by_user_session(
        self, user_id: str, session_id: Optional[str], limit: int = 5
    ) -> List[dict]:
        if session_id is not None:
            return [
                p
                for p in self.patterns.values()
                if p["user_id"] == user_id and p.get("last_session_id") == session_id
            ]

        user_patterns = [p for p in self.patterns.values() if p["user_id"] == user_id]
        user_patterns.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        return user_patterns[:limit]


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
    def __init__(self, qdrant_store: QdrantStore, embedder: Embedder):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.text_store = InMemoryTextStore()
        self.pattern_store = InMemoryPatternStore()
        self.artifact_store = InMemoryArtifactStore()
        self.occurrence_store = InMemoryOccurrenceStore()
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

        embedding = self.embedder.embed_single(user_text.text)

        self.qdrant.upsert(
            collection_name="user_text_embeddings",
            points=[
                {
                    "id": document_id,
                    "vector": embedding,
                    "payload": {
                        "user_text_id": user_text_id,
                        "text": user_text.text,
                        "user_id": user_text.user_id,
                        "session_id": session_id or "",
                        "timestamp": user_text.timestamp.isoformat(),
                        "source": "raw_text",
                    },
                }
            ],
        )

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
            # LanguageTool not available, fall back to old pattern creation
            pass

        self._create_or_update_mistake_pattern_from_text(
            user_text=user_text,
            user_text_id=user_text_id,
            embedding=embedding,
            session_id=session_id,
        )

        return document_id, user_text_id

    def _ingest_mistake_event(
        self, event: dict, user_text_id: str
    ) -> tuple[Optional[dict], Optional[dict]]:
        """
        Deduplication workflow for mistake events.
        
        Returns:
            (example_point, occurrence_point) - either can be None
        """
        mistake_type = event["mistake_type"]
        context_vector = event["context_vector"]
        mistake_logic_vector = event["mistake_logic_vector"]
        
        # Stage 1: Category check - does this mistake_type exist?
        # Use a dummy vector since Qdrant requires a vector for search
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
                "payload": {
                    **{k: v for k, v in event.items() if k not in ["mistake_logic_vector", "context_vector"]},
                    "canonical_example": event["text"],
                },
            }
            
            occurrence_point = {
                "id": event["mistake_id"],
                "vectors": {
                    "mistake_logic": mistake_logic_vector,
                },
                "payload": {
                    **{k: v for k, v in event.items() if k not in ["mistake_logic_vector", "context_vector"]},
                },
            }
            
            # Persist occurrence to SQL placeholder
            self.occurrence_store.insert({
                "mistake_id": event["mistake_id"],
                "user_text_id": user_text_id,
                "detected_at": event["timestamp"],
                "source": event["source"],
                "mistake_type": mistake_type,
                "rule_id": event["rule_id"],
            })
            
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
                "vectors": {
                    "mistake_logic": mistake_logic_vector,
                },
                "payload": {
                    **{k: v for k, v in event.items() if k not in ["mistake_logic_vector", "context_vector"]},
                },
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
            "payload": {
                **{k: v for k, v in event.items() if k not in ["mistake_logic_vector", "context_vector"]},
                "canonical_example": event["text"],
            },
        }
        
        occurrence_point = {
            "id": event["mistake_id"],
            "vectors": {
                "mistake_logic": mistake_logic_vector,
            },
            "payload": {
                **{k: v for k, v in event.items() if k not in ["mistake_logic_vector", "context_vector"]},
            },
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
        self._ensure_patterns_for_session(query.user_id, query.session_id)

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
            
            is_correct = len(events) == 0
            detected_mistake_id = None
            
            if events:
                # Use first detected mistake for feedback
                first_event = events[0]
                detected_mistake_id = first_event.get("mistake_id")
                feedback = self._exercise_feedback_from_event(first_event)
                
                # Process all events through deduplication
                example_points = []
                occurrence_points = []
                
                for event in events:
                    example_point, occurrence_point = self._ingest_mistake_event(
                        event=event,
                        user_text_id=exercise_attempt_id,
                    )
                    if example_point:
                        example_points.append(example_point)
                    if occurrence_point:
                        occurrence_points.append(occurrence_point)
                
                if example_points:
                    self.qdrant.upsert("mistake_examples", example_points)
                if occurrence_points:
                    self.qdrant.upsert("mistake_occurrences", occurrence_points)
            else:
                feedback = "Looks correct. Keep going."
            
            # Store exercise attempt in user_text_embeddings (weak signal)
            attempt_vector = self.embedder.embed_single(attempt.text)
            self.qdrant.upsert(
                collection_name="user_text_embeddings",
                points=[
                    {
                        "id": exercise_attempt_id,
                        "vector": attempt_vector,
                        "payload": {
                            "user_id": attempt.user_id,
                            "text": attempt.text,
                            "source": "exercise_attempt",
                            "weight": 0.5,
                        },
                    }
                ],
            )
            
            return {
                "exercise_attempt_id": exercise_attempt_id,
                "detected_mistake_id": detected_mistake_id,
                "is_correct": is_correct,
                "feedback": feedback,
            }
        except ImportError:
            # Fallback if LanguageTool not available
            attempt_vector = self.embedder.embed_single(attempt.text)
            candidate_pattern_ids = self._candidate_patterns_for_exercise(attempt)
            detected_pattern = self._detect_exercise_pattern(
                attempt_vector=attempt_vector,
                user_id=attempt.user_id,
                candidate_pattern_ids=candidate_pattern_ids,
            )
            
            is_correct = detected_pattern is None
            feedback = self._exercise_feedback(detected_pattern)
            
            self.qdrant.upsert(
                collection_name="user_text_embeddings",
                points=[
                    {
                        "id": exercise_attempt_id,
                        "vector": attempt_vector,
                        "payload": {
                            "user_id": attempt.user_id,
                            "text": attempt.text,
                            "source": "exercise_attempt",
                            "weight": 0.5,
                        },
                    }
                ],
            )
            
            return {
                "exercise_attempt_id": exercise_attempt_id,
                "detected_mistake_id": detected_pattern.get("pattern_id") if detected_pattern else None,
                "is_correct": is_correct,
                "feedback": feedback,
            }

    def _candidate_patterns_for_exercise(self, attempt: ExerciseAttempt) -> List[str]:
        if attempt.target_pattern_id:
            return [attempt.target_pattern_id]

        recent_patterns = self.pattern_store.get_by_user_session(attempt.user_id, None)
        return [p["pattern_id"] for p in recent_patterns if p.get("pattern_id")]

    def _detect_exercise_pattern(
        self,
        attempt_vector: List[float],
        user_id: str,
        candidate_pattern_ids: List[str],
    ) -> Optional[dict]:
        if not candidate_pattern_ids:
            return None

        filters = {"user_id": user_id}
        if len(candidate_pattern_ids) == 1:
            filters["pattern_id"] = candidate_pattern_ids[0]

        results = self.qdrant.search(
            collection_name="mistake_pattern_embeddings",
            vector=attempt_vector,
            limit=5,
            filters=filters,
        )

        for result in results:
            if result["score"] < self.pattern_similarity_threshold:
                continue
            payload = result.get("payload", {})
            pattern_id = payload.get("pattern_id")
            if candidate_pattern_ids and pattern_id not in candidate_pattern_ids:
                continue
            return {
                "pattern_id": pattern_id,
                "description": payload.get("description", ""),
            }

        return None

    @staticmethod
    def _exercise_feedback(detected_pattern: Optional[dict]) -> str:
        if not detected_pattern:
            return "Looks correct. Keep going."
        description = detected_pattern.get("description", "this pattern")
        return f"Possible issue with {description}. Try correcting the form."
    
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
            if t.get("source") != "exercise"
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

    def _ensure_patterns_for_session(self, user_id: str, session_id: Optional[str]) -> None:
        """
        DEPRECATED: Pattern creation is no longer part of active flow.
        
        Still called for backward compatibility but patterns are not used in lesson generation.
        Lesson context now comes from mistake_examples collection.
        """
        session_texts = self.text_store.get_by_user_session(user_id, session_id)
        candidate_texts: List[str] = []

        if session_texts:
            candidate_texts = [t["text"] for t in session_texts]
        else:
            # Fallback to recent user texts to keep semantics global, not session-gated.
            candidate_texts = [t["text"] for t in self.text_store.get_by_user(user_id)]

        if not candidate_texts:
            return

        cluster_texts = candidate_texts[: self.max_context_items]
        # Semantic creation/reuse happens inside; pattern_id reuse is driven by similarity in Qdrant.
        self._create_or_update_mistake_pattern(cluster_texts, user_id, session_id)

    def _create_or_update_mistake_pattern_from_text(
        self,
        user_text: UserText,
        user_text_id: str,
        embedding: List[float],
        session_id: Optional[str],
    ) -> None:
        """
        DEPRECATED: Pattern creation replaced by LanguageTool event-driven mistake_type system.
        
        This function still runs but patterns are not used in active lesson generation.
        Will be removed in V2.
        """
        similar_results = self.qdrant.search(
            collection_name="user_text_embeddings",
            vector=embedding,
            limit=5,
            filters={"user_id": user_text.user_id},
        )

        cluster_texts = [user_text.text]
        for result in similar_results:
            if result["score"] < self.pattern_similarity_threshold:
                continue
            payload = result.get("payload", {})
            if payload.get("source") == "exercise":
                continue
            if payload.get("user_text_id") == user_text_id:
                continue
            candidate_text = payload.get("text")
            if candidate_text:
                cluster_texts.append(candidate_text)

        self._create_or_update_mistake_pattern(
            cluster_texts=cluster_texts, user_id=user_text.user_id, session_id=session_id
        )

    def _create_or_update_mistake_pattern(
        self, cluster_texts: List[str], user_id: str, session_id: Optional[str]
    ) -> None:
        """
        DEPRECATED: Pattern creation replaced by mistake_type + mistake_examples collection.
        
        Patterns are no longer used in lesson generation. Lesson context comes from
        mistake_examples queried by mistake_type. Will be removed in V2.
        """
        if not cluster_texts:
            return

        dedup_seen: Set[str] = set()
        unique_texts: List[str] = []
        for text in cluster_texts:
            if text and text not in dedup_seen:
                dedup_seen.add(text)
                unique_texts.append(text)

        if not unique_texts:
            return

        combined_text = " ".join(unique_texts)
        pattern_vector = self.embedder.embed_single(combined_text)

        existing = self.qdrant.search(
            collection_name="mistake_pattern_embeddings",
            vector=pattern_vector,
            limit=1,
            filters={"user_id": user_id},
        )

        canonical_description = unique_texts[0][:120]
        pattern_id = None

        if existing and existing[0]["score"] >= self.pattern_similarity_threshold:
            pattern_id = existing[0]["payload"].get("pattern_id", existing[0]["id"])

        if pattern_id is None:
            for pattern in self.pattern_store.patterns.values():
                if (
                    pattern.get("user_id") == user_id
                    and pattern.get("canonical_description") == canonical_description
                ):
                    pattern_id = pattern.get("pattern_id")
                    break

        if pattern_id is None:
            pattern_id = str(uuid.uuid4())

        description = f"Recurring mistake: {canonical_description}"
        examples = unique_texts[: self.max_examples_per_pattern]
        updated_at = datetime.now(timezone.utc).isoformat()

        pattern_payload = {
            "pattern_id": pattern_id,
            "canonical_description": canonical_description,
            "description": description,
            "examples": examples,
            "user_id": user_id,
            "last_session_id": session_id or "",
            "updated_at": updated_at,
        }

        self.pattern_store.upsert(pattern_id, pattern_payload)

        self.qdrant.upsert(
            collection_name="mistake_pattern_embeddings",
            points=[
                {
                    "id": pattern_id,
                    "vector": pattern_vector,
                    "payload": pattern_payload,
                }
            ],
        )

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

        return ContextAssembly(
            detected_patterns=detected_patterns,
            recently_used_explanations=recently_used_explanations,
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

            patterns.append(
                {
                    "pattern_id": mistake_type,  # Backward compatibility: use mistake_type as pattern_id
                    "description": self._mistake_type_to_description(mistake_type),
                    "examples": [canonical_example] if canonical_example else [],
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