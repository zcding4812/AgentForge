"""知识库检索编排：关键词 / 向量 / 混合分支与降级（纯函数 + 子路调用）。

门面 :class:`~app.knowledge.facades.KnowledgeRetrievalFacade` 仅负责加载 KB、校验 ``doc_id``、
注入依赖并调用本模块，便于单测与后续接 RetrieverBuilder / Postprocessor 链。
"""

from __future__ import annotations

from app.core.constants.knowledge import HYBRID_RRF_K
from app.core.logger import get_logger
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.knowledge.adapters.retrieval import (
    build_keyword_search_hits,
    fuse_hybrid_rrf,
    run_vector_ann_search,
)
from app.knowledge.kernel.ports import VectorIndexPort
from app.models.knowledge_mod import KnowledgeBase
from app.repositories.chunk_repo import ChunkRepository
from app.schemas.knowledge import KnowledgeSearchData, KnowledgeSearchHit, RetrievalType

logger = get_logger(__name__)

DOWNGRADE_NOTE = "未配置 MILVUS_URI 或知识库未绑定嵌入模型，已使用 Mongo 分片关键词匹配。"


def normalize_retrieval_type(raw: str | None) -> RetrievalType:
    s = (raw or "keyword").strip().lower()
    return s if s in ("keyword", "vector", "hybrid") else "keyword"  # type: ignore[return-value]


def assemble_search_data(
    *,
    kb_retrieval: RetrievalType,
    effective: RetrievalType,
    kw_items: list[KnowledgeSearchHit],
    vec_items: list[KnowledgeSearchHit],
    vec_err: str | None,
    can_ann: bool,
    limit: int,
    rrf_k: int | None = None,
) -> KnowledgeSearchData:
    """由已算好的关键词 / 向量子路结果组装 ``KnowledgeSearchData``。

    ``kb_retrieval`` 为知识库配置（响应 ``configured_retrieval``）；``effective`` 为本次生效策略（分支逻辑）。
    """
    if effective == "keyword":
        sliced = kw_items[:limit]
        return KnowledgeSearchData(
            items=sliced,
            total=len(sliced),
            configured_retrieval=kb_retrieval,
            applied_retrieval="keyword",
            note=None,
        )

    if not can_ann:
        if effective == "vector":
            sliced = kw_items[:limit]
            return KnowledgeSearchData(
                items=sliced,
                total=len(sliced),
                configured_retrieval=kb_retrieval,
                applied_retrieval="keyword",
                note=DOWNGRADE_NOTE,
            )
        sliced = kw_items[:limit]
        return KnowledgeSearchData(
            items=sliced,
            total=len(sliced),
            configured_retrieval=kb_retrieval,
            applied_retrieval="keyword",
            note=DOWNGRADE_NOTE,
        )

    if effective == "vector":
        if vec_items:
            sliced = vec_items[:limit]
            return KnowledgeSearchData(
                items=sliced,
                total=len(sliced),
                configured_retrieval=kb_retrieval,
                applied_retrieval="vector",
                note=None,
            )
        note = "向量检索无命中；已使用关键词匹配。"
        if vec_err:
            note = f"向量检索失败：{vec_err}；已使用关键词匹配。"
        sliced = kw_items[:limit]
        return KnowledgeSearchData(
            items=sliced,
            total=len(sliced),
            configured_retrieval=kb_retrieval,
            applied_retrieval="keyword",
            note=note,
        )

    if vec_items:
        _k = int(rrf_k) if rrf_k is not None else HYBRID_RRF_K
        merged = fuse_hybrid_rrf(vec_items, kw_items, limit=limit, rrf_k=_k)
        hybrid_note = None
        if vec_err:
            hybrid_note = f"向量侧异常：{vec_err}；结果已合并但可能不完整。"
        return KnowledgeSearchData(
            items=merged,
            total=len(merged),
            configured_retrieval=kb_retrieval,
            applied_retrieval="hybrid",
            note=hybrid_note,
        )

    hybrid_note = "向量侧无命中；以下为关键词子路结果。"
    if vec_err:
        hybrid_note = f"向量检索失败：{vec_err}；以下为关键词子路结果。"
    sliced = kw_items[:limit]
    return KnowledgeSearchData(
        items=sliced,
        total=len(sliced),
        configured_retrieval=kb_retrieval,
        applied_retrieval="keyword",
        note=hybrid_note,
    )


async def fetch_keyword_hits(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
    kb_id: int,
    query: str,
    limit: int,
    doc_id: int | None,
) -> list[KnowledgeSearchHit]:
    """Mongo 关键词子路。"""
    return await build_keyword_search_hits(
        db_manager=db_manager,
        chunk_repository=chunk_repository,
        kb_id=kb_id,
        query=query,
        limit=limit,
        doc_id=doc_id,
    )


class KnowledgeRetrieverBuilder:
    """子路组装：关键词 Mongo 检索 + 向量 ANN（供门面注入，便于后续扩展为完整 Builder 图）。"""

    def __init__(
        self,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        chunk_repository: ChunkRepository,
        vector_index: VectorIndexPort,
    ) -> None:
        self._db = db_manager
        self._chunk_repository = chunk_repository
        self._vector_index = vector_index

    async def keyword_hits(
        self,
        *,
        kb_id: int,
        query: str,
        limit: int,
        doc_id: int | None,
    ) -> list[KnowledgeSearchHit]:
        return await fetch_keyword_hits(
            db_manager=self._db,
            chunk_repository=self._chunk_repository,
            kb_id=kb_id,
            query=query,
            limit=limit,
            doc_id=doc_id,
        )

    async def vector_try(
        self,
        *,
        kb_row: KnowledgeBase,
        query: str,
        kb_id: int,
        limit: int,
        doc_id: int | None,
    ) -> tuple[list[KnowledgeSearchHit], str | None]:
        return await try_vector_search(
            db_manager=self._db,
            chunk_repository=self._chunk_repository,
            vector_index=self._vector_index,
            kb_row=kb_row,
            query=query,
            kb_id=kb_id,
            limit=limit,
            doc_id=doc_id,
        )


async def try_vector_search(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
    vector_index: VectorIndexPort,
    kb_row: KnowledgeBase,
    query: str,
    kb_id: int,
    limit: int,
    doc_id: int | None,
) -> tuple[list[KnowledgeSearchHit], str | None]:
    """向量 ANN 子路；异常时返回空列表与截断错误信息。"""
    try:
        items = await run_vector_ann_search(
            db_manager=db_manager,
            chunk_repository=chunk_repository,
            kb_row=kb_row,
            query=query,
            limit=limit,
            doc_id=doc_id,
            vector_index=vector_index,
        )
        return items, None
    except Exception as e:
        logger.exception("knowledge.vector.search.failed kb_id=%s", kb_id)
        return [], str(e)[:240]


class HybridMergePostprocessor:
    """混合路后处理：仅 RRF 融合（委托 :func:`~app.knowledge.adapters.retrieval.fuse_hybrid_rrf`）。"""

    @staticmethod
    def merge(
        vector_first: list[KnowledgeSearchHit],
        keyword_hits: list[KnowledgeSearchHit],
        limit: int,
    ) -> list[KnowledgeSearchHit]:
        return fuse_hybrid_rrf(vector_first, keyword_hits, limit=limit)
