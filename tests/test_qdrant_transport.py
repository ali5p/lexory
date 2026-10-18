from unittest.mock import Mock

from vectorstore import qdrant_client


def test_remote_qdrant_client_prefers_grpc(monkeypatch, tmp_path):
    client = Mock()
    qdrant_client_constructor = Mock(return_value=client)
    monkeypatch.setattr(qdrant_client, "QdrantClient", qdrant_client_constructor)
    monkeypatch.setattr(
        qdrant_client.QdrantStore,
        "_ensure_collections_remote_with_retries",
        Mock(),
    )

    qdrant_client.QdrantStore(path=str(tmp_path), url="http://qdrant:6333")

    qdrant_client_constructor.assert_called_once_with(
        url="http://qdrant:6333",
        prefer_grpc=True,
    )
