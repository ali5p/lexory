import uuid
from datetime import datetime
from typing import List, Optional
from core.models import (
    UserText,
    MistakePattern,
    LearningSummary,
    LessonArtifact,
    QueryRequest,
    QueryResponse,
)
from rag.embedder import Embedder
from vectorstore.qdrant_client import QdrantStore


class RAGService:
    def __init__(self, qdrant_store: QdrantStore, embedder: Embedder):
        self.qdrant = qdrant_store
        self.embedder = embedder

    def ingest_user_text(self, user_text: UserText) -> str:
        document_id = str(uuid.uuid4())
        embedding = self.embedder.embed_single(user_text.text)

        self.qdrant.upsert(
            collection_name="user_text_embeddings",
            points=[
                {
                    "id": document_id,
                    "vector": embedding,
                    "payload": {
                        "text": user_text.text,
                        "user_id": user_text.user_id,
                        "timestamp": user_text.timestamp.isoformat(),
                    },
                }
            ],
        )

        return document_id

    def generate_lesson(self, query: QueryRequest) -> QueryResponse:
        query_text = query.query or "Generate a learning lesson"
        query_embedding = self.embedder.embed_single(query_text)

        user_filter = {"user_id": query.user_id}

        patterns = self.qdrant.search(
            collection_name="mistake_pattern_embeddings",
            query_vector=query_embedding,
            limit=3,
            filter_dict=user_filter,
        )

        summaries = self.qdrant.search(
            collection_name="learning_summary_embeddings",
            query_vector=query_embedding,
            limit=3,
            filter_dict=user_filter,
        )

        artifacts = self.qdrant.search(
            collection_name="lesson_artifact_embeddings",
            query_vector=query_embedding,
            limit=3,
            filter_dict=user_filter,
        )

        relevant_patterns = [
            p["payload"].get("description", "") for p in patterns if p["payload"]
        ]
        relevant_summaries = [
            s["payload"].get("content", "") for s in summaries if s["payload"]
        ]
        relevant_artifacts = [
            a["payload"].get("content", "") for a in artifacts if a["payload"]
        ]

        lesson_content = self._construct_lesson(
            relevant_patterns, relevant_summaries, relevant_artifacts
        )

        return QueryResponse(
            lesson_content=lesson_content,
            relevant_patterns=relevant_patterns,
            relevant_summaries=relevant_summaries,
            relevant_artifacts=relevant_artifacts,
        )

    def _construct_lesson(
        self,
        patterns: List[str],
        summaries: List[str],
        artifacts: List[str],
    ) -> str:
        parts = []
        if patterns:
            parts.append("Relevant mistake patterns:\n" + "\n".join(patterns))
        if summaries:
            parts.append("Learning summaries:\n" + "\n".join(summaries))
        if artifacts:
            parts.append("Previous lessons:\n" + "\n".join(artifacts))

        if not parts:
            return "No relevant learning materials found. Start by submitting some text."

        return "\n\n".join(parts)

