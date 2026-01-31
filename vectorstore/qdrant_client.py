from pathlib import Path
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
)

# qdrant-client 1.9+ uses dict-based named vectors directly
# No need for NamedVector/NamedVectors classes


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
        # Standard single-vector collections (384-dim)
        standard_collections = [
            "user_text_embeddings",
            "mistake_pattern_embeddings",
            "learning_summary_embeddings",
            "lesson_artifact_embeddings",
        ]
        for collection_name in standard_collections:
            try:
                self.client.get_collection(collection_name)
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        
        # Named-vector collections
        self._ensure_named_collection(
            "mistake_examples",
            {
                "mistake_logic": (64, Distance.COSINE),
                "context": (384, Distance.COSINE),
            },
        )
        self._ensure_named_collection(
            "mistake_occurrences",
            {
                "mistake_logic": (64, Distance.COSINE),
            },
        )
    
    def _ensure_named_collection(
        self, collection_name: str, named_vectors: Dict[str, tuple[int, Distance]]
    ):
        """Create collection with named vectors if missing."""
        try:
            self.client.get_collection(collection_name)
        except Exception:
            # qdrant-client 1.9+ accepts dict directly for named vectors
            vectors_config = {
                name: VectorParams(size=size, distance=distance)
                for name, (size, distance) in named_vectors.items()
            }
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

    def upsert(self, collection_name: str, points: list[dict]):
        """Upsert points, supporting both single vector and named vectors."""
        points_struct = []
        for point in points:
            point_id = point["id"]
            payload = point.get("payload", {})
            
            # Check for named vectors (new format)
            if "vectors" in point or "named_vectors" in point:
                vectors_dict = point.get("vectors") or point.get("named_vectors", {})
                # qdrant-client 1.9+ accepts dict directly for named vectors
                points_struct.append(
                    PointStruct(id=point_id, vector=vectors_dict, payload=payload)
                )
            # Legacy single vector format
            elif "vector" in point:
                points_struct.append(
                    PointStruct(
                        id=point_id,
                        vector=point["vector"],
                        payload=payload,
                    )
                )
            else:
                raise ValueError(f"Point {point_id} must have 'vector' or 'vectors' field")
        
        self.client.upsert(collection_name=collection_name, points=points_struct)

    def search(
        self,
        collection_name: str,
        vector: Optional[list[float]] = None,
        limit: int = 5,
        filters: Optional[dict] = None,
        named_query: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        """
        Search with support for single vector (legacy) or named vector queries.
        
        Args:
            collection_name: Collection to search
            vector: Legacy single vector (for backward compatibility)
            limit: Max results
            filters: Payload filters dict
            named_query: Dict with 'vector_name' and 'vector' for named vector search
        """
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Named vector query (new format)
        if named_query:
            vector_name = named_query["vector_name"]
            query_vector = named_query["vector"]
            # qdrant-client 1.9+ accepts dict for named vector queries
            query_vec = {vector_name: query_vector}
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vec,
                limit=limit,
                query_filter=query_filter,
            )
        # Legacy single vector query
        elif vector is not None:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
            )
        else:
            raise ValueError("Must provide either 'vector' or 'named_query'")

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

