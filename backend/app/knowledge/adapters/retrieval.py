"""知识库检索子路：Mongo 关键词、混合合并、节点映射、向量 ANN（嵌入 + Milvus + Mongo 正文）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.schema import TextNode

from app.config import get_settings
from app.core.constants.knowledge import (
    HYBRID_KEYWORD_RRF_WEIGHT,
    HYBRID_MAX_CHUNKS_PER_DOC,
    HYBRID_RRF_K,
    default_milvus_collection_name,
)
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.knowledge.adapters.embedding_adapter import EmbeddingAdapter
from app.knowledge.kernel.ports import VectorIndexPort
from app.knowledge.kernel.types import RetrievalHit
from app.models.knowledge_mod import KnowledgeBase
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.knowledge_repo import KnowledgeDocumentRepository
from app.schemas.knowledge import KnowledgeSearchHit


# ------------------------------------------------------------------------------
# Mongo 关键词子路
# ------------------------------------------------------------------------------
async def build_keyword_search_hits(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
    kb_id: int,
    query: str,
    limit: int,
    doc_id: int | None,
) -> list[KnowledgeSearchHit]:
    """知识库 Mongo 全文/关键词子路（与门面历史行为一致）。"""
    raw_hits = await chunk_repository.search_chunks(
        kb_id,
        query,
        limit=limit,
        doc_id=doc_id,
    )
    doc_ids = list({h["doc_id"] for h in raw_hits})
    names: dict[int, str] = {}
    for did in doc_ids:
        row = await KnowledgeDocumentRepository.get_by_id(did, db_manager=db_manager)
        if row is not None:
            names[did] = row.filename
    items: list[KnowledgeSearchHit] = []
    for h in raw_hits:
        text = str(h.get("text", ""))
        snip = text[:200] + ("…" if len(text) > 200 else "")
        items.append(
            KnowledgeSearchHit(
                doc_id=int(h["doc_id"]),
                chunk_index=int(h["chunk_index"]) if h.get("chunk_index") is not None else None,
                text_snippet=snip,
                filename=names.get(int(h["doc_id"])),
                match_type="keyword",
            ),
        )
    return items


# ------------------------------------------------------------------------------
# 混合：向量路 + 关键词路合并
# ------------------------------------------------------------------------------
def _hybrid_hit_key(h: KnowledgeSearchHit) -> tuple[int, int]:
    return (h.doc_id, h.chunk_index if h.chunk_index is not None else -1)


def _diversify_scored_hits(
    ranked: list[tuple[float, KnowledgeSearchHit]],
    *,
    limit: int,
    max_per_doc: int,
) -> list[KnowledgeSearchHit]:
    """按 RRF 分从高到低取条，同一 ``doc_id`` 至多保留 ``max_per_doc`` 条。"""
    if max_per_doc <= 0:
        out: list[KnowledgeSearchHit] = []
        for score, h in ranked[:limit]:
            out.append(h.model_copy(update={"score": round(float(score), 6)}))
        return out
    counts: dict[int, int] = {}
    out2: list[KnowledgeSearchHit] = []
    for score, h in ranked:
        if len(out2) >= limit:
            break
        if counts.get(h.doc_id, 0) >= max_per_doc:
            continue
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
        out2.append(h.model_copy(update={"score": round(float(score), 6)}))
    return out2


def fuse_hybrid_rrf(
    vector_hits: list[KnowledgeSearchHit],
    keyword_hits: list[KnowledgeSearchHit],
    *,
    limit: int,
    rrf_k: int = HYBRID_RRF_K,
    keyword_rrf_weight: float = HYBRID_KEYWORD_RRF_WEIGHT,
    max_chunks_per_doc: int = HYBRID_MAX_CHUNKS_PER_DOC,
) -> list[KnowledgeSearchHit]:
    """混合检索：两路 **RRF（倒数排名融合）**，不做神经 / 交叉编码重排。

    同键 ``(doc_id, chunk_index)`` 优先保留向量路条目；``score`` 为 RRF 合分。
    关键词子路可加权；融合后可按文档限制分片条数，减轻同文档重复刷屏。
    """
    if limit <= 0:
        return []
    kw_w = float(keyword_rrf_weight) if keyword_rrf_weight > 0 else 1.0
    scores: dict[tuple[int, int], float] = {}
    for rank, h in enumerate(vector_hits):
        key = _hybrid_hit_key(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, h in enumerate(keyword_hits):
        key = _hybrid_hit_key(h)
        scores[key] = scores.get(key, 0.0) + kw_w * (1.0 / (rrf_k + rank + 1))

    hits_map: dict[tuple[int, int], KnowledgeSearchHit] = {}
    for h in vector_hits:
        hits_map[_hybrid_hit_key(h)] = h
    for h in keyword_hits:
        k = _hybrid_hit_key(h)
        if k not in hits_map:
            hits_map[k] = h

    ranked: list[tuple[float, KnowledgeSearchHit]] = [
        (scores[k], hits_map[k]) for k in scores if k in hits_map
    ]
    ranked.sort(key=lambda t: -t[0])

    return _diversify_scored_hits(
        ranked,
        limit=limit,
        max_per_doc=max_chunks_per_doc,
    )


def merge_hybrid_hits(
    vector_first: list[KnowledgeSearchHit],
    keyword_hits: list[KnowledgeSearchHit],
    limit: int,
) -> list[KnowledgeSearchHit]:
    """兼容：等价于 :func:`fuse_hybrid_rrf`。"""
    return fuse_hybrid_rrf(vector_first, keyword_hits, limit=limit)


# 别名：与历史命名一致
fuse_hybrid_recall_rerank = fuse_hybrid_rrf


# ------------------------------------------------------------------------------
# 防腐层：DTO ↔ LlamaIndex 节点（Milvus / Retriever 路径）
# ------------------------------------------------------------------------------
META_KB_DOC_ID = "kb_doc_id"
META_CHUNK_INDEX = "chunk_index"
META_CONTENT_VERSION = "content_version"


def _snippet_from_text(text: str, *, max_len: int = 200) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[:max_len] + "…"


def text_node_to_retrieval_hit(
    node: TextNode,
    *,
    score: float | None = None,
) -> RetrievalHit:
    """``TextNode`` → :class:`~app.knowledge.kernel.types.RetrievalHit`。"""
    md = node.metadata or {}
    out: RetrievalHit = {
        "node_id": str(node.node_id),
        "snippet": _snippet_from_text(node.get_content() or ""),
        "score": float(score) if score is not None else 0.0,
    }
    if META_KB_DOC_ID in md:
        try:
            out["doc_id"] = int(md[META_KB_DOC_ID])
        except (TypeError, ValueError):
            pass
    if META_CHUNK_INDEX in md:
        try:
            out["chunk_index"] = int(md[META_CHUNK_INDEX])
        except (TypeError, ValueError):
            out["chunk_index"] = None
    cv = md.get(META_CONTENT_VERSION)
    if cv is not None:
        out["content_version"] = str(cv)
    return out


def text_node_to_knowledge_search_hit(
    node: TextNode,
    *,
    match_type: Literal["chunk", "filename", "keyword", "vector"] = "vector",
    filename: str | None = None,
) -> KnowledgeSearchHit:
    """``TextNode`` → HTTP 契约 :class:`KnowledgeSearchHit`。"""
    md = node.metadata or {}
    doc_id = int(md[META_KB_DOC_ID]) if META_KB_DOC_ID in md else 0
    cidx_raw = md.get(META_CHUNK_INDEX)
    chunk_index: int | None
    try:
        chunk_index = int(cidx_raw) if cidx_raw is not None else None
    except (TypeError, ValueError):
        chunk_index = None
    snip = _snippet_from_text(node.get_content() or "")
    return KnowledgeSearchHit(
        doc_id=doc_id,
        chunk_index=chunk_index,
        text_snippet=snip,
        filename=filename,
        match_type=match_type,
    )


def vector_node_pairs_to_ann_tuples(
    pairs: Sequence[tuple[Any, float]],
) -> list[tuple[int, int, str, float]]:
    """Milvus ``(节点, distance)`` → 与 :meth:`VectorIndexPort.search_vectors` 相同的 ANN 行元组。"""
    out: list[tuple[int, int, str, float]] = []
    for node, dist in pairs:
        md = getattr(node, "metadata", None) or {}
        did = int(md.get(META_KB_DOC_ID, 0))
        cidx = int(md.get(META_CHUNK_INDEX, 0))
        cv = str(md.get(META_CONTENT_VERSION) or "")
        out.append((did, cidx, cv, float(dist)))
    return out


def text_nodes_to_knowledge_search_hits(
    nodes: Sequence[TextNode],
    *,
    filenames_by_doc_id: dict[int, str | None] | None = None,
    match_type: Literal["chunk", "filename", "keyword", "vector"] = "vector",
) -> list[KnowledgeSearchHit]:
    """批量映射；``filenames_by_doc_id`` 用于补全 ``filename``。"""
    names = filenames_by_doc_id or {}
    out: list[KnowledgeSearchHit] = []
    for node in nodes:
        md = node.metadata or {}
        did = int(md[META_KB_DOC_ID]) if META_KB_DOC_ID in md else 0
        fn = names.get(did) if did else None
        out.append(
            text_node_to_knowledge_search_hit(
                node,
                match_type=match_type,
                filename=fn,
            ),
        )
    return out


# ------------------------------------------------------------------------------
# 向量 ANN
# ------------------------------------------------------------------------------
async def _mongo_backed_vector_hits_from_ann_rows(
    *,
    kb_id: int,
    rows: list[tuple[int, int, str, float]],
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
) -> list[KnowledgeSearchHit]:
    """ANN 行 ``(doc_id, chunk_index, content_version, distance)`` + Mongo 正文 → ``KnowledgeSearchHit``。"""
    names: dict[int, str | None] = {}
    items: list[KnowledgeSearchHit] = []
    for did, cidx, cv, _dist in rows:
        text = await chunk_repository.get_chunk_text(
            kb_id=kb_id,
            doc_id=did,
            content_version=cv,
            chunk_index=cidx,
        )
        if text is None:
            text = ""
        snip = text[:200] + ("…" if len(text) > 200 else "")
        if did not in names:
            row = await KnowledgeDocumentRepository.get_by_id(did, db_manager=db_manager)
            names[did] = row.filename if row is not None else None
        fn = names.get(did)
        items.append(
            KnowledgeSearchHit(
                doc_id=did,
                chunk_index=cidx,
                text_snippet=snip,
                filename=fn,
                match_type="vector",
            ),
        )
    return items


async def run_vector_ann_search(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
    kb_row: KnowledgeBase,
    query: str,
    limit: int,
    doc_id: int | None,
    vector_index: VectorIndexPort,
) -> list[KnowledgeSearchHit]:
    """查询向量嵌入 + ANN，再从 Mongo 取分片文本构造命中项。"""
    settings = get_settings()
    if not settings.milvus_configured:
        return []
    if kb_row.embedding_model_config_id is None:
        return []

    embedder = await EmbeddingAdapter.client_for_model_config(
        db_manager,
        kb_row.embedding_model_config_id,
    )
    q = query.strip()
    vecs = await embedder.embed_texts([q])
    if not vecs:
        return []
    query_vector = vecs[0]
    dim = len(query_vector)
    coll = (kb_row.milvus_collection or "").strip() or default_milvus_collection_name(kb_row.id)

    if settings.knowledge_vector_search_use_li_nodes:
        pairs = await vector_index.search_vector_nodes(
            collection_name=coll,
            dim=dim,
            kb_id=kb_row.id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )
        rows = vector_node_pairs_to_ann_tuples(pairs)
    else:
        rows = await vector_index.search_vectors(
            collection_name=coll,
            dim=dim,
            kb_id=kb_row.id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )

    return await _mongo_backed_vector_hits_from_ann_rows(
        kb_id=kb_row.id,
        rows=rows,
        db_manager=db_manager,
        chunk_repository=chunk_repository,
    )
