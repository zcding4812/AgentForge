"""知识库文档：列表与上传（MinIO 由 lifespan 单例注入；测试中 mock put/remove）。"""

import io
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(autouse=True)
def minio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """先于 TestClient / lifespan，保证 ``MinioConfig.from_settings`` 读到三项连接配置。"""
    monkeypatch.setenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@patch("app.services.knowledge_svc.enqueue_ingest_after_upload", new_callable=AsyncMock)
@patch(
    "app.infrastructure.minio.MinioObjectStore.remove_object",
    new_callable=AsyncMock,
)
@patch(
    "app.infrastructure.minio.MinioObjectStore.put_object",
    new_callable=AsyncMock,
)
def test_knowledge_documents_list_and_upload(
    mock_put: object,
    mock_remove: object,
    mock_enqueue: AsyncMock,
    client: TestClient,
) -> None:
    mock_enqueue.return_value = "pytest-task-id"
    name = f"pytest-kb-doc-{uuid.uuid4().hex[:10]}"
    cr = client.post(
        "/api/knowledge",
        json={"name": name, "description": "d0"},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert cr.status_code == 200
    kb_id = cr.json()["data"]["id"]

    lr = client.get(
        f"/api/knowledge/{kb_id}/documents",
        headers={"Accept": "application/json"},
    )
    assert lr.status_code == 200
    body = lr.json()["data"]
    assert body["total"] == 0
    assert body["items"] == []

    files = {"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")}
    ur = client.post(
        f"/api/knowledge/{kb_id}/documents/upload",
        files=files,
        headers={"Accept": "application/json"},
    )
    assert ur.status_code == 200
    payload = ur.json()["data"]
    doc = payload["document"]
    assert payload.get("task_id") == "pytest-task-id"
    assert doc["kb_id"] == kb_id
    assert doc["filename"] == "hello.txt"
    assert doc["sha256"] is not None
    assert doc["status"] == "pending"
    mock_put.assert_called_once()
    mock_remove.assert_not_called()
    mock_enqueue.assert_called_once()

    lr2 = client.get(
        f"/api/knowledge/{kb_id}/documents",
        headers={"Accept": "application/json"},
    )
    assert lr2.status_code == 200
    body2 = lr2.json()["data"]
    assert body2["total"] == 1
    assert body2["items"][0]["id"] == doc["id"]

    dr = client.delete(f"/api/knowledge/{kb_id}", headers={"Accept": "application/json"})
    assert dr.status_code == 204


@patch("app.services.knowledge_svc.enqueue_ingest_after_upload", new_callable=AsyncMock)
@patch(
    "app.infrastructure.minio.MinioObjectStore.remove_object",
    new_callable=AsyncMock,
)
@patch(
    "app.infrastructure.minio.MinioObjectStore.put_object",
    new_callable=AsyncMock,
)
def test_knowledge_upload_async_returns_202(
    mock_put: object,
    mock_remove: object,
    mock_enqueue: AsyncMock,
    client: TestClient,
) -> None:
    mock_enqueue.return_value = "async-task-id"
    name = f"pytest-kb-async-{uuid.uuid4().hex[:10]}"
    cr = client.post(
        "/api/knowledge",
        json={"name": name, "description": "async"},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert cr.status_code == 200
    kb_id = cr.json()["data"]["id"]
    ur = client.post(
        f"/api/knowledge/{kb_id}/documents/upload?async=true",
        files={"file": ("async.txt", io.BytesIO(b"x"), "text/plain")},
        headers={"Accept": "application/json"},
    )
    assert ur.status_code == 202
    body = ur.json()
    assert body["message"] == "accepted"
    assert body["data"]["document"]["filename"] == "async.txt"
    assert body["data"].get("task_id") == "async-task-id"
    mock_enqueue.assert_awaited_once()
    dr = client.delete(f"/api/knowledge/{kb_id}", headers={"Accept": "application/json"})
    assert dr.status_code == 204


@patch(
    "app.repositories.chunk_repo.ChunkRepository.delete_all_chunks_for_document",
    new_callable=AsyncMock,
)
@patch(
    "app.infrastructure.minio.MinioObjectStore.remove_object",
    new_callable=AsyncMock,
)
@patch(
    "app.infrastructure.minio.MinioObjectStore.put_object",
    new_callable=AsyncMock,
)
@patch("app.services.knowledge_svc.enqueue_ingest_after_upload", new_callable=AsyncMock)
def test_knowledge_document_delete(
    _mock_enqueue: AsyncMock,
    _mock_put: object,
    mock_remove: object,
    mock_chunk_delete: AsyncMock,
    client: TestClient,
) -> None:
    _mock_enqueue.return_value = "pytest-task-id"
    mock_chunk_delete.return_value = 0
    name = f"pytest-kb-deldoc-{uuid.uuid4().hex[:10]}"
    cr = client.post(
        "/api/knowledge",
        json={"name": name, "description": "d1"},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert cr.status_code == 200
    kb_id = cr.json()["data"]["id"]

    files = {"file": ("bye.txt", io.BytesIO(b"x"), "text/plain")}
    ur = client.post(
        f"/api/knowledge/{kb_id}/documents/upload",
        files=files,
        headers={"Accept": "application/json"},
    )
    assert ur.status_code == 200
    doc_id = ur.json()["data"]["document"]["id"]

    ddr = client.delete(
        f"/api/knowledge/{kb_id}/documents/{doc_id}",
        headers={"Accept": "application/json"},
    )
    assert ddr.status_code == 204
    mock_chunk_delete.assert_called_once()
    mock_remove.assert_called()

    lr = client.get(f"/api/knowledge/{kb_id}/documents", headers={"Accept": "application/json"})
    assert lr.status_code == 200
    assert lr.json()["data"]["total"] == 0


def test_knowledge_documents_404_unknown_kb(client: TestClient) -> None:
    r = client.get("/api/knowledge/999999999/documents", headers={"Accept": "application/json"})
    assert r.status_code == 404


@patch("app.services.knowledge_svc.enqueue_ingest_after_upload", new_callable=AsyncMock)
@patch(
    "app.infrastructure.minio.MinioObjectStore.remove_object",
    new_callable=AsyncMock,
)
@patch(
    "app.infrastructure.minio.MinioObjectStore.put_object",
    new_callable=AsyncMock,
)
def test_knowledge_document_patch_chunk_override(
    _mock_put: object,
    _mock_remove: object,
    _mock_enqueue: AsyncMock,
    client: TestClient,
) -> None:
    _mock_enqueue.return_value = "pytest-task-id"
    name = f"pytest-kb-patch-{uuid.uuid4().hex[:10]}"
    cr = client.post(
        "/api/knowledge",
        json={"name": name, "description": "d"},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert cr.status_code == 200
    kb_id = cr.json()["data"]["id"]

    files = {"file": ("p.txt", io.BytesIO(b"hello"), "text/plain")}
    ur = client.post(
        f"/api/knowledge/{kb_id}/documents/upload",
        files=files,
        headers={"Accept": "application/json"},
    )
    assert ur.status_code == 200
    doc = ur.json()["data"]["document"]
    doc_id = doc["id"]
    assert doc.get("chunk_method") is None

    pr = client.patch(
        f"/api/knowledge/{kb_id}/documents/{doc_id}",
        json={
            "chunk_method": "length",
            "chunk_size": 1024,
            "chunk_overlap": 64,
            "chunk_separator": None,
        },
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert pr.status_code == 200
    patched = pr.json()["data"]
    assert patched["chunk_method"] == "length"
    assert patched["chunk_size"] == 1024
    assert patched["chunk_overlap"] == 64

    lr = client.get(f"/api/knowledge/{kb_id}/documents", headers={"Accept": "application/json"})
    assert lr.status_code == 200
    items = lr.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["chunk_method"] == "length"

    clr = client.patch(
        f"/api/knowledge/{kb_id}/documents/{doc_id}",
        json={"chunk_method": None},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    assert clr.status_code == 200
    cleared = clr.json()["data"]
    assert cleared["chunk_method"] is None
    assert cleared["chunk_size"] is None

    dr = client.delete(f"/api/knowledge/{kb_id}", headers={"Accept": "application/json"})
    assert dr.status_code == 204
