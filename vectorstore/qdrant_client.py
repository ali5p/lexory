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
    OrderBy,
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
        self._ensure_timestamp_index("mistake_examples")
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

    def _ensure_timestamp_index(self, collection_name: str) -> None:
        """Ensure timestamp payload index for order_by (scroll_most_recent)."""
        if isinstance(self.client, QdrantLocal):
            return  # Payload indexes have no effect in local Qdrant
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="timestamp",
                field_schema="keyword",
            )
        except Exception:
            pass  # Index may already exist

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
            # qdrant-client expects (vector_name, vector) tuple for named vector search
            query_vec = (vector_name, query_vector)
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


    # TODO: scroll_most_recent could be returned to replace scroll_by_mistake_id
    """
    def scroll_most_recent(
        self,
        collection_name: str,
        user_id: str,
        limit: int = 1,
        order_by_key: str = "timestamp",
    ) -> list[dict]:
        
        # Scroll most recent points by payload timestamp (desc).
        # Used for fallback query embedding when session has no candidates.
        # Returns points with vectors (context) for mistake_examples.
        # QdrantLocal: order_by not supported, so we scroll without it (any matching point).
        
        query_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        kwargs: dict = {
            "collection_name": collection_name,
            "scroll_filter": query_filter,
            "limit": limit,
            "with_payload": True,
            "with_vectors": ["context"],
        }
        if not isinstance(self.client, QdrantLocal):
            kwargs["order_by"] = OrderBy(key=order_by_key, direction="desc")
        try:
            records, _ = self.client.scroll(**kwargs)
        except Exception:
            return []
        out = []
        for rec in records:
            vec = None
            if isinstance(getattr(rec, "vector", None), dict):
                vec = rec.vector.get("context")
            out.append({
                "id": rec.id,
                "payload": rec.payload or {},
                "vectors": {"context": vec} if vec else {},
            })
        return out
    """

    def scroll_by_mistake_id(
        self,
        collection_name: str,
        user_id: str,
        mistake_id: str,
    ) -> list[dict]:
        """
        Scroll points filtered by user_id + mistake_id (no order_by).
        Returns points with context vector for mistake_examples.
        """
        query_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="mistake_id", match=MatchValue(value=mistake_id)),
            ]
        )
        try:
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=["context"],
            )
        except Exception:
            return []
        out = []
        for rec in records:
            vec = None
            if isinstance(getattr(rec, "vector", None), dict):
                vec = rec.vector.get("context")
            out.append({
                "id": rec.id,
                "payload": rec.payload or {},
                "vectors": {"context": vec} if vec else {},
            })
        return out

