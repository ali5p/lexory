from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct


class QdrantStore:
    def __init__(self, path: str = "./qdrant_storage", url: str | None = None):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantLocal(str(self.path))
        self._ensure_collections()

    def _ensure_collections(self):
        collections = [
            "user_text_embeddings",
            "mistake_pattern_embeddings",
            "learning_summary_embeddings",
            "lesson_artifact_embeddings",
        ]
        for collection_name in collections:
            try:
                self.client.get_collection(collection_name)
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

    def upsert(self, collection_name: str, points: list[dict]):
        points_struct = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in points
        ]
        self.client.upsert(collection_name=collection_name, points=points_struct)

    def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        # Qdrant expects a Filter object; we map simple dicts to keep callers minimal.
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

