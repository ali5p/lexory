import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
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

_log = logging.getLogger(__name__)

# Remote Qdrant may not accept connections until a few seconds after container start.
_REMOTE_INIT_RETRIES = 60
_REMOTE_INIT_DELAY_SEC = 1.0


def _collection_missing(exc: BaseException) -> bool:
    """True if get_collection failed because the collection does not exist."""
    if isinstance(exc, UnexpectedResponse):
        code = getattr(exc, "status_code", None)
        if code == 404:
            return True
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg or "does not exist" in msg:
            return True
    msg = str(exc).lower()
    if "not found" in msg or "does not exist" in msg or "doesn't exist" in msg:
        return True
    return False


class QdrantStore:
    def __init__(self, path: str = "./qdrant_storage", url: str | None = None):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        if url:
            self.client = QdrantClient(url=url)
            self._ensure_collections_remote_with_retries()
        else:
            self.client = QdrantLocal(str(self.path))
            self._ensure_collections()

    def _ensure_collections_remote_with_retries(self) -> None:
        last: BaseException | None = None
        for attempt in range(1, _REMOTE_INIT_RETRIES + 1):
            try:
                self._ensure_collections()
                if attempt > 1:
                    _log.info("Qdrant became reachable after %s attempt(s)", attempt)
                return
            except ResponseHandlingException as e:
                last = e
                _log.warning(
                    "Qdrant not reachable yet (%s/%s): %s",
                    attempt,
                    _REMOTE_INIT_RETRIES,
                    e,
                )
                time.sleep(_REMOTE_INIT_DELAY_SEC)
        assert last is not None
        raise last

    def _ensure_collections(self):
        # Standard single-vector collections (384-dim)
        standard_collections = [
            "learning_summary_embeddings",
            "lesson_artifact_embeddings",
        ]
        for collection_name in standard_collections:
            try:
                self.client.get_collection(collection_name)
            except UnexpectedResponse as e:
                if _collection_missing(e):
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )
                else:
                    raise
            except (ValueError, RuntimeError) as e:
                if _collection_missing(e):
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )
                else:
                    raise
        
        # Named-vector collections
        self._ensure_named_collection(
            "mistake_examples",
            {
                "mistake_logic": (64, Distance.COSINE),
                "context": (384, Distance.COSINE),
            },
        )
        self._ensure_detected_at_index("mistake_examples")
        self._ensure_user_id_index("mistake_examples")
        self._ensure_named_collection(
            "mistake_occurrences",
            {
                "mistake_logic": (64, Distance.COSINE),
            },
        )
        self._ensure_user_id_index("mistake_occurrences")
        self._ensure_user_id_index("learning_summary_embeddings")
        self._ensure_user_id_index("lesson_artifact_embeddings")
    
    def _ensure_named_collection(
        self, collection_name: str, named_vectors: Dict[str, tuple[int, Distance]]
    ):
        """Create collection with named vectors if missing."""
        try:
            self.client.get_collection(collection_name)
        except UnexpectedResponse as e:
            if not _collection_missing(e):
                raise
            # qdrant-client 1.9+ accepts dict directly for named vectors
            vectors_config = {
                name: VectorParams(size=size, distance=distance)
                for name, (size, distance) in named_vectors.items()
            }
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
        except (ValueError, RuntimeError) as e:
            if not _collection_missing(e):
                raise
            vectors_config = {
                name: VectorParams(size=size, distance=distance)
                for name, (size, distance) in named_vectors.items()
            }
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

    def _ensure_detected_at_index(self, collection_name: str) -> None:
        # Payload index for payload field detected_at (ordering / filters).
        if isinstance(self.client, QdrantLocal):
            return  # Payload indexes have no effect in local Qdrant
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="detected_at",
                field_schema="keyword",
            )
        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                pass
            else:
                raise

    def _ensure_user_id_index(self, collection_name: str) -> None:
        """Ensure user_id payload index for multi-tenant isolation."""
        if isinstance(self.client, QdrantLocal):
            return  # Payload indexes have no effect in local Qdrant
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="user_id",
                field_schema="keyword",
            )
        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                pass
            else:
                raise
    
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
        Uses query_points (qdrant-client 1.10+) or search (QdrantLocal).
        """
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        # Prefer query_points (remote QdrantClient); fallback to search (QdrantLocal)
        if named_query:
            vector_name = named_query["vector_name"]
            query_vector = named_query["vector"]
            if hasattr(self.client, "query_points"):
                resp = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    limit=limit,
                    query_filter=query_filter,
                )
                results = resp.points if hasattr(resp, "points") else []
            else:
                query_vec = (vector_name, query_vector)
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vec,
                    limit=limit,
                    query_filter=query_filter,
                )
        elif vector is not None:
            if hasattr(self.client, "query_points"):
                resp = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                )
                results = resp.points if hasattr(resp, "points") else []
            else:
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
                "id": getattr(r, "id", r.get("id") if isinstance(r, dict) else None),
                "score": getattr(r, "score", r.get("score", 0.0) if isinstance(r, dict) else 0.0),
                "payload": getattr(r, "payload", r.get("payload", {}) if isinstance(r, dict) else {}),
            }
            for r in results
        ]


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
        except UnexpectedResponse as e:
            _log.warning(
                "scroll_by_mistake_id: Qdrant UnexpectedResponse (%s)",
                e,
                exc_info=True,
            )
            return []
        except (ConnectionError, TimeoutError, OSError) as e:
            _log.warning(
                "scroll_by_mistake_id: connection error (%s)",
                e,
                exc_info=True,
            )
            return []
        except Exception:
            _log.exception("scroll_by_mistake_id: unexpected error")
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

