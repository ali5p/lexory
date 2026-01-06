from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import Optional


class QdrantStore:
    def __init__(self, path: str = "./qdrant_storage", port: int = 6333):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
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

    def upsert(
        self,
        collection_name: str,
        points: list[dict],
    ):
        from qdrant_client.models import PointStruct

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
        query_vector: list[float],
        limit: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, Match

        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(FieldCondition(key=key, match=Match(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
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

