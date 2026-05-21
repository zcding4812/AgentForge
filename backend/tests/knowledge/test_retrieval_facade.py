"""``KnowledgeRetrievalFacade.retrieve``：关键词 / 向量 / 混合与降级行为稳定。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.knowledge.facades import KnowledgeRetrievalFacade
from app.schemas.knowledge import KnowledgeSearchHit


def _kb(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "id": 1,
        "retrieval_type": "keyword",
        "embedding_model_config_id": None,
        "milvus_collection": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_keyword_applied() -> None:
    db = MagicMock()
    chunks = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks)
    with patch.object(
        KnowledgeRetrievalFacade,
        "_require_kb",
        new_callable=AsyncMock,
        return_value=_kb(retrieval_type="vector"),
    ):
        out = await facade.retrieve(1, q="   ", limit=5)
    assert out.items == []
    assert out.applied_retrieval == "keyword"
    assert out.total == 0


@pytest.mark.asyncio
async def test_retrieve_keyword_only() -> None:
    db = MagicMock()
    chunks = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks)
    hits = [
        KnowledgeSearchHit(
            doc_id=1,
            chunk_index=0,
            text_snippet="hi",
            filename="a.txt",
            match_type="keyword",
        )
    ]
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(retrieval_type="keyword"),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fetch_keyword_hits",
            new_callable=AsyncMock,
            return_value=hits,
        ),
    ):
        out = await facade.retrieve(1, q="hello", limit=5)
    assert out.configured_retrieval == "keyword"
    assert out.applied_retrieval == "keyword"
    assert out.total == 1
    assert out.note is None


@pytest.mark.asyncio
async def test_retrieve_vector_downgrade_when_milvus_off() -> None:
    db = MagicMock()
    chunks = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks)
    kw = [
        KnowledgeSearchHit(
            doc_id=1,
            chunk_index=0,
            text_snippet="x",
            filename=None,
            match_type="keyword",
        )
    ]
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(
                retrieval_type="vector",
                embedding_model_config_id=99,
            ),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fetch_keyword_hits",
            new_callable=AsyncMock,
            return_value=kw,
        ),
        patch(
            "app.knowledge.facades._settings",
            SimpleNamespace(milvus_configured=False),
        ),
    ):
        out = await facade.retrieve(1, q="q", limit=5)
    assert out.configured_retrieval == "vector"
    assert out.applied_retrieval == "keyword"
    assert out.note is not None
    assert "MILVUS" in (out.note or "").upper()


@pytest.mark.asyncio
async def test_retrieve_vector_hits_vector_applied() -> None:
    db = MagicMock()
    chunks = MagicMock()
    vec = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks, vector_index=vec)
    vhit = KnowledgeSearchHit(
        doc_id=2,
        chunk_index=1,
        text_snippet="v",
        filename="b.txt",
        match_type="vector",
    )
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(
                retrieval_type="vector",
                embedding_model_config_id=5,
            ),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fetch_keyword_hits",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.knowledge.facades._settings",
            SimpleNamespace(milvus_configured=True),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.try_vector_search",
            new_callable=AsyncMock,
            return_value=([vhit], None),
        ),
    ):
        out = await facade.retrieve(1, q="q", limit=5)
    assert out.applied_retrieval == "vector"
    assert out.total == 1
    assert out.items[0].match_type == "vector"


@pytest.mark.asyncio
async def test_retrieve_override_keyword_on_hybrid_kb() -> None:
    """单次请求可覆盖为关键词，不走向量。"""
    db = MagicMock()
    chunks = MagicMock()
    vec = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks, vector_index=vec)
    kh = KnowledgeSearchHit(
        doc_id=1, chunk_index=0, text_snippet="b", filename=None, match_type="keyword"
    )
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(
                retrieval_type="hybrid",
                embedding_model_config_id=3,
            ),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fetch_keyword_hits",
            new_callable=AsyncMock,
            return_value=[kh],
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.try_vector_search",
            new_callable=AsyncMock,
        ) as mock_vec,
        patch(
            "app.knowledge.facades._settings",
            SimpleNamespace(milvus_configured=True),
        ),
    ):
        out = await facade.retrieve(
            1,
            q="q",
            limit=10,
            retrieval_override="keyword",
        )
    assert out.configured_retrieval == "hybrid"
    assert out.applied_retrieval == "keyword"
    mock_vec.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_hybrid_merges() -> None:
    db = MagicMock()
    chunks = MagicMock()
    vec = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks, vector_index=vec)
    vh = KnowledgeSearchHit(
        doc_id=1, chunk_index=0, text_snippet="a", filename=None, match_type="vector"
    )
    kh = KnowledgeSearchHit(
        doc_id=2, chunk_index=0, text_snippet="b", filename=None, match_type="keyword"
    )
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(
                retrieval_type="hybrid",
                embedding_model_config_id=3,
            ),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fetch_keyword_hits",
            new_callable=AsyncMock,
            return_value=[kh],
        ),
        patch(
            "app.knowledge.facades._settings",
            SimpleNamespace(milvus_configured=True),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.try_vector_search",
            new_callable=AsyncMock,
            return_value=([vh], None),
        ),
        patch(
            "app.knowledge.adapters.retrieval_pipeline.fuse_hybrid_rrf",
            return_value=[vh, kh],
        ),
    ):
        out = await facade.retrieve(1, q="q", limit=10)
    assert out.configured_retrieval == "hybrid"
    assert out.applied_retrieval == "hybrid"
    assert len(out.items) == 2


@pytest.mark.asyncio
async def test_retrieve_doc_id_wrong_kb_raises() -> None:
    db = MagicMock()
    chunks = MagicMock()
    facade = KnowledgeRetrievalFacade(db, chunks)
    with (
        patch.object(
            KnowledgeRetrievalFacade,
            "_require_kb",
            new_callable=AsyncMock,
            return_value=_kb(),
        ),
        patch(
            "app.knowledge.facades.KnowledgeDocumentRepository.get_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        with pytest.raises(LookupError, match="文档不存在"):
            await facade.retrieve(1, q="x", doc_id=99)
