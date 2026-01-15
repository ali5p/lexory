import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from core.models import (
    ContextAssembly,
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
    """Stores MistakePatterns for the user."""

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


class RAGService:
    def __init__(self, qdrant_store: QdrantStore, embedder: Embedder):
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.text_store = InMemoryTextStore()
        self.pattern_store = InMemoryPatternStore()
        self.artifact_store = InMemoryArtifactStore()
        self.max_context_items = 10
        self.min_similarity_score = 0.5
        self.pattern_similarity_threshold = 0.65
        self.max_examples_per_pattern = 3
        self._approach_registry: Dict[str, BaseApproach] = {
            "rule_based": RuleBasedApproach(),
            "example_based": ExampleBasedApproach(),
            "default": DefaultApproach(),
        }

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
                    },
                }
            ],
        )

        self._create_or_update_mistake_pattern_from_text(
            user_text=user_text,
            user_text_id=user_text_id,
            embedding=embedding,
            session_id=session_id,
        )

        return document_id, user_text_id

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

    def _build_query_embedding(
        self, user_id: str, session_id: Optional[str], fallback_query: Optional[str]
    ) -> List[float]:
        embedding_texts: List[str] = []

        session_patterns = self.pattern_store.get_by_user_session(user_id, session_id)
        pattern_descriptions = [
            p["description"] for p in session_patterns if p.get("description")
        ]

        session_texts = [
            t["text"] for t in self.text_store.get_by_user_session(user_id, session_id)
        ]

        if pattern_descriptions:
            embedding_texts.extend(pattern_descriptions[: self.max_context_items])

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
        similar_results = self.qdrant.search(
            collection_name="user_text_embeddings",
            query_vector=embedding,
            limit=5,
            filter_dict={"user_id": user_text.user_id},
        )

        cluster_texts = [user_text.text]
        for result in similar_results:
            if result["score"] < self.pattern_similarity_threshold:
                continue
            payload = result.get("payload", {})
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
            query_vector=pattern_vector,
            limit=1,
            filter_dict={"user_id": user_id},
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

    def _retrieve_mistake_patterns(
        self, query_embedding: List[float], user_filter: Dict[str, str]
    ) -> List[dict]:
        results = self.qdrant.search(
            collection_name="mistake_pattern_embeddings",
            query_vector=query_embedding,
            limit=5,
            filter_dict=user_filter,
        )

        patterns = []
        seen_pattern_ids: Set[str] = set()

        for result in results:
            if result["score"] < self.min_similarity_score:
                continue

            payload = result.get("payload", {})
            pattern_id = payload.get("pattern_id", result["id"])

            if pattern_id in seen_pattern_ids:
                continue

            seen_pattern_ids.add(pattern_id)
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "description": payload.get("description", ""),
                    "examples": payload.get("examples", []),
                    "similarity_score": result["score"],
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
            query_vector=query_embedding,
            limit=10,
            filter_dict=user_filter,
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

        patterns_covered = [
            p.get("pattern_id")
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
            "patterns_covered": patterns_covered,
            "pedagogy_tags": pedagogy_tags,
            "created_at": created_at.isoformat(),
        }

        self.artifact_store.upsert(artifact_id, artifact_payload)

        content_for_embedding = self._artifact_embedding_text(
            lesson=lesson,
            patterns_covered=patterns_covered,
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