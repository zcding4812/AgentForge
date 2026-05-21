"""``retrieval_pipeline.assemble_search_data``：纯函数分支与降级文案回归。"""

from __future__ import annotations

import pytest

from app.knowledge.adapters.retrieval import fuse_hybrid_rrf
from app.knowledge.adapters.retrieval_pipeline import (
    DOWNGRADE_NOTE,
    assemble_search_data,
    normalize_retrieval_type,
)
from app.schemas.knowledge import KnowledgeSearchHit


def _hit(doc_id: int, mt: str = "keyword") -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        doc_id=doc_id,
        chunk_index=0,
        text_snippet="x",
        filename=None,
        match_type=mt,  # type: ignore[arg-type]
    )


def test_normalize_retrieval_type() -> None:
    assert normalize_retrieval_type("VECTOR") == "vector"
    assert normalize_retrieval_type(None) == "keyword"
    assert normalize_retrieval_type("unknown") == "keyword"


def test_assemble_keyword_only() -> None:
    kw = [_hit(1)]
    out = assemble_search_data(
        kb_retrieval="keyword",
        effective="keyword",
        kw_items=kw,
        vec_items=[],
        vec_err=None,
        can_ann=False,
        limit=10,
    )
    assert out.configured_retrieval == "keyword"
    assert out.applied_retrieval == "keyword"
    assert out.total == 1
    assert out.note is None


def test_assemble_vector_downgrade() -> None:
    kw = [_hit(1)]
    out = assemble_search_data(
        kb_retrieval="vector",
        effective="vector",
        kw_items=kw,
        vec_items=[],
        vec_err=None,
        can_ann=False,
        limit=10,
    )
    assert out.applied_retrieval == "keyword"
    assert out.note == DOWNGRADE_NOTE


def test_assemble_vector_success() -> None:
    kw = [_hit(1)]
    vv = [_hit(2, "vector")]
    out = assemble_search_data(
        kb_retrieval="vector",
        effective="vector",
        kw_items=kw,
        vec_items=vv,
        vec_err=None,
        can_ann=True,
        limit=10,
    )
    assert out.applied_retrieval == "vector"
    assert out.total == 1
    assert out.items[0].match_type == "vector"


def test_assemble_vector_fallback_note() -> None:
    kw = [_hit(1)]
    out = assemble_search_data(
        kb_retrieval="vector",
        effective="vector",
        kw_items=kw,
        vec_items=[],
        vec_err="boom",
        can_ann=True,
        limit=10,
    )
    assert out.applied_retrieval == "keyword"
    assert out.note is not None
    assert "boom" in (out.note or "")


def test_assemble_hybrid_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    kw = [_hit(10)]
    vv = [_hit(20, "vector")]

    def _merge(v, k, *, limit: int = 10, rrf_k: int = 60):
        return v + k

    monkeypatch.setattr(
        "app.knowledge.adapters.retrieval_pipeline.fuse_hybrid_rrf",
        _merge,
    )
    out = assemble_search_data(
        kb_retrieval="hybrid",
        effective="hybrid",
        kw_items=kw,
        vec_items=vv,
        vec_err=None,
        can_ann=True,
        limit=10,
    )
    assert out.applied_retrieval == "hybrid"
    assert len(out.items) == 2


def test_fuse_hybrid_rrf_respects_max_chunks_per_doc() -> None:
    """同一文档多命中时最多保留 ``max_chunks_per_doc`` 条。"""
    kw = [
        KnowledgeSearchHit(
            doc_id=1,
            chunk_index=i,
            text_snippet="x",
            filename=None,
            match_type="keyword",
        )
        for i in range(5)
    ]
    out = fuse_hybrid_rrf([], kw, limit=10, max_chunks_per_doc=2)
    assert len(out) == 2
    assert {h.chunk_index for h in out} <= {0, 1, 2, 3, 4}


def test_fuse_hybrid_rrf_prefers_both_lists() -> None:
    """两路同一 doc/chunk 只保留一条，分数为 RRF 合分。"""
    a = KnowledgeSearchHit(
        doc_id=1,
        chunk_index=0,
        text_snippet="alpha beta",
        filename="x",
        match_type="vector",
    )
    b = KnowledgeSearchHit(
        doc_id=1,
        chunk_index=0,
        text_snippet="alpha beta",
        filename="x",
        match_type="keyword",
    )
    c = KnowledgeSearchHit(
        doc_id=2,
        chunk_index=0,
        text_snippet="gamma",
        filename="y",
        match_type="keyword",
    )
    out = fuse_hybrid_rrf([a], [b, c], limit=10)
    assert len(out) == 2
    keys = {(h.doc_id, h.chunk_index) for h in out}
    assert keys == {(1, 0), (2, 0)}
    d1 = next(h for h in out if h.doc_id == 1)
    assert d1.match_type == "vector"
    assert d1.score is not None and d1.score > 0


def test_assemble_hybrid_no_vector_keyword_only() -> None:
    kw = [_hit(1)]
    out = assemble_search_data(
        kb_retrieval="hybrid",
        effective="hybrid",
        kw_items=kw,
        vec_items=[],
        vec_err=None,
        can_ann=True,
        limit=10,
    )
    assert out.applied_retrieval == "keyword"
    assert "无命中" in (out.note or "")
