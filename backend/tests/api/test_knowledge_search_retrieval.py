"""知识库：索引配置与 search 的 retrieval 语义。"""

import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_create_vector_without_embedding_rejected(client: TestClient) -> None:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r = client.post(
        "/api/knowledge",
        json={
            "name": f"pytest-kb-{uuid.uuid4().hex[:8]}",
            "retrieval_type": "vector",
        },
        headers=headers,
    )
    assert r.status_code == 400
    assert "embedding" in r.json()["detail"].lower()


@patch(
    "app.repositories.chunk_repo.ChunkRepository.search_chunks",
    new_callable=AsyncMock,
    return_value=[],
)
def test_search_returns_retrieval_fields_keyword_kb(
    _mock_chunks: AsyncMock, client: TestClient
) -> None:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    cr = client.post(
        "/api/knowledge",
        json={"name": f"pytest-kb-{uuid.uuid4().hex[:8]}", "retrieval_type": "keyword"},
        headers=headers,
    )
    assert cr.status_code == 200
    kb_id = cr.json()["data"]["id"]

    sr = client.post(
        f"/api/knowledge/{kb_id}/search",
        json={"q": "anything", "limit": 5},
        headers=headers,
    )
    assert sr.status_code == 200
    data = sr.json()["data"]
    assert data["configured_retrieval"] == "keyword"
    assert data["applied_retrieval"] == "keyword"
    assert data.get("note") in (None, "")

    dr = client.delete(f"/api/knowledge/{kb_id}", headers={"Accept": "application/json"})
    assert dr.status_code == 204
