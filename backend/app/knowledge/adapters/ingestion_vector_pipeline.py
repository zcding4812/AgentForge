"""Pipeline②：嵌入 + Milvus（仅 vector/hybrid 且已配置嵌入时执行）。

与 Pipeline① 分离；**不**更新任务表——由 :class:`~app.services.knowledge.ingest.IngestTaskHandler` 在 Pipeline 外推进。
实现委托 :func:`app.services.knowledge.vector.upsert_document_chunks_to_milvus`。
"""

from __future__ import annotations

from app.config import get_settings
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.knowledge_mod import KnowledgeBase
from app.services.knowledge.vector import upsert_document_chunks_to_milvus


async def run_vector_ingestion_pipeline(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    kb: KnowledgeBase,
    doc_id: int,
    content_version: str,
    chunks: list[str],
) -> None:
    """Pipeline②：Mongo 已写入的分片文本 → 向量 → Milvus（条件与旧 ``_try_upsert_milvus`` 一致）。"""
    settings = get_settings()
    if not settings.milvus_configured:
        return
    rt = (kb.retrieval_type or "keyword").strip().lower()
    if rt not in ("vector", "hybrid") or kb.embedding_model_config_id is None:
        return
    await upsert_document_chunks_to_milvus(
        db_manager=db_manager,
        kb=kb,
        doc_id=doc_id,
        content_version=content_version,
        chunks=chunks,
    )
