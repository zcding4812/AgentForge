"""知识库 **对外唯一出口**：Agent / HTTP / 测试应从此模块导入门面。

本模块实现 :class:`KnowledgeRetrievalFacade`（唯一检索入口）；并再导出
:class:`~app.knowledge.adapters.embedding_adapter.EmbeddingAdapter` 便于发现。

**禁止**在业务代码中绕过门面直接访问向量库或 Mongo 检索接口。
"""

from __future__ import annotations

from app.config import get_settings
from app.core.constants.knowledge import HYBRID_RECALL_CAP
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.knowledge.adapters import (
    EmbeddingAdapter,  # noqa: F401
    KnowledgeRetrieverBuilder,
    assemble_search_data,
    get_vector_index_port,
    normalize_retrieval_type,
)
from app.knowledge.kernel.ports import VectorIndexPort
from app.models.knowledge_mod import KnowledgeBase
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.knowledge_repo import (
    KnowledgeBaseRepository,
    KnowledgeDocumentRepository,
)
from app.schemas.knowledge import (
    KnowledgeSearchData,
    KnowledgeSearchEnvelope,
    RetrievalType,
)

_settings = get_settings()


def _compute_sub_limit(
    effective: RetrievalType,
    limit: int,
    can_ann: bool,
    recall_limit: int | None,
) -> int:
    """各子路从存储拉取的条数上限；最终仍由 ``assemble_search_data`` 截断到 ``limit``。"""
    cap = HYBRID_RECALL_CAP
    if recall_limit is not None:
        return max(limit, min(int(recall_limit), cap))
    if effective == "hybrid" and can_ann:
        return min(cap, max(limit * 3, limit + 10))
    return limit


class KnowledgeRetrievalFacade:
    """**唯一**检索门面：禁止在 Agent 内核或路由中绕过此类直接访问向量库 / Mongo。

    实现组合关键词、向量、混合策略；返回已定型的 ``KnowledgeSearchData``。
    编排委托 :mod:`app.knowledge.adapters.retrieval_pipeline`（子路 + 纯函数组装），**不**依赖 LlamaIndex 类型。
    """

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager,
        chunk_repository: ChunkRepository,
        vector_index: VectorIndexPort | None = None,
    ) -> None:
        self._db = db_manager
        self._chunk_repository = chunk_repository
        self._vector_index: VectorIndexPort = (
            vector_index if vector_index is not None else get_vector_index_port(get_settings())
        )
        self._kb_repo = KnowledgeBaseRepository
        self._doc_repo = KnowledgeDocumentRepository
        self._retrievers = KnowledgeRetrieverBuilder(
            db_manager=self._db,
            chunk_repository=self._chunk_repository,
            vector_index=self._vector_index,
        )

    async def _require_kb(self, kb_id: int) -> KnowledgeBase:
        kb = await self._kb_repo.get_active_by_id(kb_id, db_manager=self._db)
        if kb is None:
            raise LookupError("知识库不存在或已删除")
        return kb

    async def retrieve(
        self,
        kb_id: int,
        *,
        q: str,
        limit: int = 20,
        doc_id: int | None = None,
        retrieval_override: RetrievalType | None = None,
        recall_limit: int | None = None,
        rrf_k: int | None = None,
    ) -> KnowledgeSearchData:
        """知识库检索：与历史 ``KnowledgeDocumentService.search_in_kb`` 行为一致。"""
        kb_row = await self._require_kb(kb_id)
        kb_retrieval = normalize_retrieval_type(kb_row.retrieval_type)
        effective = (
            normalize_retrieval_type(retrieval_override)
            if retrieval_override is not None
            else kb_retrieval
        )

        query = q.strip()
        if not query:
            return KnowledgeSearchData(
                items=[],
                total=0,
                configured_retrieval=kb_retrieval,
                applied_retrieval="keyword",
                note=None,
            )

        if doc_id is not None:
            row = await self._doc_repo.get_by_id(int(doc_id), db_manager=self._db)
            if row is None or row.kb_id != kb_id:
                raise LookupError("文档不存在或不属于该知识库")

        can_ann = _settings.milvus_configured and kb_row.embedding_model_config_id is not None
        sub_limit = _compute_sub_limit(effective, limit, can_ann, recall_limit)

        kw_items = await self._retrievers.keyword_hits(
            kb_id=kb_id,
            query=query,
            limit=sub_limit,
            doc_id=doc_id,
        )

        if effective == "keyword":
            data = assemble_search_data(
                kb_retrieval=kb_retrieval,
                effective="keyword",
                kw_items=kw_items,
                vec_items=[],
                vec_err=None,
                can_ann=can_ann,
                limit=limit,
                rrf_k=rrf_k,
            )
            return data

        if not can_ann:
            data = assemble_search_data(
                kb_retrieval=kb_retrieval,
                effective=effective,
                kw_items=kw_items,
                vec_items=[],
                vec_err=None,
                can_ann=False,
                limit=limit,
                rrf_k=rrf_k,
            )
            return data

        vec_items, vec_err = await self._retrievers.vector_try(
            kb_row=kb_row,
            query=query,
            kb_id=kb_id,
            limit=sub_limit,
            doc_id=doc_id,
        )

        data = assemble_search_data(
            kb_retrieval=kb_retrieval,
            effective=effective,
            kw_items=kw_items,
            vec_items=vec_items,
            vec_err=vec_err,
            can_ann=True,
            limit=limit,
            rrf_k=rrf_k,
        )
        return data

    async def retrieve_as_message_envelope(
        self,
        kb_id: int,
        *,
        q: str,
        limit: int = 20,
        doc_id: int | None = None,
        retrieval_override: RetrievalType | None = None,
        recall_limit: int | None = None,
        rrf_k: int | None = None,
    ) -> KnowledgeSearchEnvelope:
        """返回定型信封；需 plain dict 时调用 :meth:`KnowledgeSearchEnvelope.model_dump`。"""
        data = await self.retrieve(
            kb_id,
            q=q,
            limit=limit,
            doc_id=doc_id,
            retrieval_override=retrieval_override,
            recall_limit=recall_limit,
            rrf_k=rrf_k,
        )
        return KnowledgeSearchEnvelope(data=data)
