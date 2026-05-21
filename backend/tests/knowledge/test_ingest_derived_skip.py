"""``should_skip_ingest_derived_writes``：Mongo 与权威一致；向量路径下再校验 Milvus 行数。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.constants.knowledge import DOCUMENT_STATUS_INDEXED
from app.knowledge.application.indexing import should_skip_ingest_derived_writes


@pytest.mark.asyncio
async def test_skip_when_keyword_kb_mongo_matches_mysql(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_repo = MagicMock()
    chunk_repo.count_chunks_for_document_version = AsyncMock(return_value=3)

    kb = MagicMock()
    kb.id = 1
    kb.retrieval_type = "keyword"
    kb.embedding_model_config_id = None

    doc = MagicMock()
    doc.id = 9
    doc.status = DOCUMENT_STATUS_INDEXED
    doc.content_version = "cv1"
    doc.chunk_count = 3

    monkeypatch.setattr(
        "app.knowledge.application.indexing.get_settings",
        lambda: SimpleNamespace(milvus_configured=True),
    )

    ok = await should_skip_ingest_derived_writes(
        chunk_repo,
        kb=kb,
        doc=doc,
        content_version="cv1",
    )
    assert ok is True
    chunk_repo.count_chunks_for_document_version.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_skip_when_vector_milvus_required(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_repo = MagicMock()
    chunk_repo.count_chunks_for_document_version = AsyncMock(return_value=2)

    kb = MagicMock()
    kb.id = 1
    kb.retrieval_type = "vector"
    kb.embedding_model_config_id = 42
    kb.milvus_collection = None

    doc = MagicMock()
    doc.id = 9
    doc.status = DOCUMENT_STATUS_INDEXED
    doc.content_version = "cv1"
    doc.chunk_count = 2

    monkeypatch.setattr(
        "app.knowledge.application.indexing.get_settings",
        lambda: SimpleNamespace(milvus_configured=True),
    )

    vix = MagicMock()
    vix.count_vectors_for_doc_version = AsyncMock(return_value=1)

    ok = await should_skip_ingest_derived_writes(
        chunk_repo,
        kb=kb,
        doc=doc,
        content_version="cv1",
        vector_index=vix,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_skip_when_vector_kb_mongo_and_milvus_match(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_repo = MagicMock()
    chunk_repo.count_chunks_for_document_version = AsyncMock(return_value=3)

    kb = MagicMock()
    kb.id = 1
    kb.retrieval_type = "vector"
    kb.embedding_model_config_id = 7
    kb.milvus_collection = None

    doc = MagicMock()
    doc.id = 9
    doc.status = DOCUMENT_STATUS_INDEXED
    doc.content_version = "cv1"
    doc.chunk_count = 3

    monkeypatch.setattr(
        "app.knowledge.application.indexing.get_settings",
        lambda: SimpleNamespace(milvus_configured=True),
    )

    vix = MagicMock()
    vix.count_vectors_for_doc_version = AsyncMock(return_value=3)

    ok = await should_skip_ingest_derived_writes(
        chunk_repo,
        kb=kb,
        doc=doc,
        content_version="cv1",
        vector_index=vix,
    )
    assert ok is True


@pytest.mark.asyncio
async def test_no_skip_when_milvus_count_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_repo = MagicMock()
    chunk_repo.count_chunks_for_document_version = AsyncMock(return_value=2)

    kb = MagicMock()
    kb.id = 1
    kb.retrieval_type = "hybrid"
    kb.embedding_model_config_id = 7
    kb.milvus_collection = None

    doc = MagicMock()
    doc.id = 9
    doc.status = DOCUMENT_STATUS_INDEXED
    doc.content_version = "cv1"
    doc.chunk_count = 2

    monkeypatch.setattr(
        "app.knowledge.application.indexing.get_settings",
        lambda: SimpleNamespace(milvus_configured=True),
    )

    vix = MagicMock()
    vix.count_vectors_for_doc_version = AsyncMock(return_value=-1)

    ok = await should_skip_ingest_derived_writes(
        chunk_repo,
        kb=kb,
        doc=doc,
        content_version="cv1",
        vector_index=vix,
    )
    assert ok is False
