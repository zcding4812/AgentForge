"""索引流水线：Pipeline①（切块）→ Mongo；向量见 :mod:`app.knowledge.adapters.ingestion_vector_pipeline`。

幂等键 ``(kb_id, doc_id, content_version)`` 与 :func:`~app.knowledge.kernel.rules.ingest_idempotency_key`、
MySQL / ``knowledge_task`` 对齐；任务状态由 ingest worker 在 Pipeline 外更新。
"""

from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.core.constants.knowledge import DOCUMENT_STATUS_INDEXED, default_milvus_collection_name
from app.domain.knowledge.chunk_config import effective_chunk_strategy_for_ingest
from app.knowledge.adapters import (
    DocumentFactory,
    NodeParserFactory,
    chunk_strings_from_pipeline_nodes,
    get_vector_index_port,
    run_chunk_pipeline_for_documents,
)
from app.knowledge.kernel.exceptions import DocumentTextExtractionError
from app.knowledge.kernel.ports import VectorIndexPort
from app.knowledge.kernel.rules import ingest_idempotency_key
from app.models.knowledge_mod import KnowledgeBase, KnowledgeDocument
from app.repositories.chunk_repo import ChunkRepository

logger = logging.getLogger(__name__)


def kb_requires_milvus_vector_upsert_for_ingest(kb: KnowledgeBase) -> bool:
    """向量/混合检索且已配置嵌入时，ingest 需写 Milvus；无法用行级元数据廉价验证向量是否已写齐。"""
    s = get_settings()
    if not s.milvus_configured:
        return False
    rt = (kb.retrieval_type or "keyword").strip().lower()
    if rt not in ("vector", "hybrid"):
        return False
    return kb.embedding_model_config_id is not None


_MILVUS_COUNT_QUERY_CAP = 65536


async def should_skip_ingest_derived_writes(
    chunk_repository: ChunkRepository,
    *,
    kb: KnowledgeBase,
    doc: KnowledgeDocument,
    content_version: str,
    vector_index: VectorIndexPort | None = None,
) -> bool:
    """若权威行与 Mongo 已一致，且非向量路径或 Milvus 行数与 ``chunk_count`` 一致，则短路 ingest。

    向量/混合路径：在 Milvus 可查询且计数与 ``knowledge_document.chunk_count`` 一致时允许跳过，
    避免重复写 Mongo/Milvus；无法查询 Milvus（返回 ``-1``）或命中枚举上界时不短路。
    """
    if doc.status != DOCUMENT_STATUS_INDEXED:
        return False
    cv_doc = (doc.content_version or "").strip()
    cv_task = (content_version or "").strip()
    if not cv_doc or cv_doc != cv_task:
        return False
    if doc.chunk_count <= 0:
        return False
    mongo_total = await chunk_repository.count_chunks_for_document_version(
        kb_id=kb.id,
        doc_id=doc.id,
        content_version=cv_task,
    )
    if mongo_total != doc.chunk_count or mongo_total == 0:
        logger.info(
            "ingest 短路跳过：Mongo 分片数与权威 chunk_count 不一致",
            extra={
                "doc_id": doc.id,
                "chunk_count_mysql": doc.chunk_count,
                "mongo_total": mongo_total,
            },
        )
        return False
    if kb_requires_milvus_vector_upsert_for_ingest(kb):
        if vector_index is None:
            vector_index = get_vector_index_port(get_settings())
        coll = (kb.milvus_collection or "").strip() or default_milvus_collection_name(kb.id)
        milvus_n = await vector_index.count_vectors_for_doc_version(
            collection_name=coll,
            kb_id=kb.id,
            doc_id=doc.id,
            content_version=cv_task,
        )
        if milvus_n < 0:
            logger.info(
                "ingest 不短路：Milvus 计数不可用",
                extra={"kb_id": kb.id, "doc_id": doc.id},
            )
            return False
        if milvus_n >= _MILVUS_COUNT_QUERY_CAP:
            logger.info(
                "ingest 不短路：Milvus 命中数达到枚举上限，无法确认是否写齐",
                extra={"kb_id": kb.id, "doc_id": doc.id, "milvus_n": milvus_n},
            )
            return False
        if milvus_n != doc.chunk_count:
            logger.info(
                "ingest 不短路：Milvus 行数与权威 chunk_count 不一致",
                extra={
                    "kb_id": kb.id,
                    "doc_id": doc.id,
                    "chunk_count_mysql": doc.chunk_count,
                    "milvus_n": milvus_n,
                },
            )
            return False
        logger.info(
            "ingest 短路：Mongo + Milvus 与权威一致（向量路径）",
            extra={"kb_id": kb.id, "doc_id": doc.id, "content_version": cv_task},
        )
        return True

    logger.info(
        "ingest 短路：派生已完整（关键词路径，Mongo 与权威一致）",
        extra={"kb_id": kb.id, "doc_id": doc.id, "content_version": cv_task},
    )
    return True


async def persist_upload_bytes_as_chunks(
    chunk_repository: ChunkRepository,
    *,
    kb: KnowledgeBase,
    doc: KnowledgeDocument,
    content_version: str,
    raw: bytes,
) -> list[str]:
    """ingest 主路径：**bytes →** :class:`~app.knowledge.adapters.document_factory.DocumentFactory` **→ Pipeline① → Mongo**。

    抽取与切块在同步块内完成（建议在 worker 内 ``asyncio.to_thread`` 调用本函数）。
    """
    _ = ingest_idempotency_key(kb_id=kb.id, doc_id=doc.id, content_version=content_version)
    strategy = effective_chunk_strategy_for_ingest(kb, doc)

    def _sync() -> list[str]:
        doc_li = DocumentFactory.from_upload_bytes(
            raw,
            mime=doc.mime,
            filename=doc.filename or "",
        )
        nodes = run_chunk_pipeline_for_documents(
            [doc_li],
            strategy,
            source_filename=doc.filename,
        )
        return chunk_strings_from_pipeline_nodes(nodes)

    try:
        chunks = await asyncio.to_thread(_sync)
    except DocumentTextExtractionError as e:
        raise RuntimeError(f"文本抽取失败：{e}") from e
    await chunk_repository.replace_document_chunks(
        kb_id=kb.id,
        doc_id=doc.id,
        content_version=content_version,
        chunks=chunks,
    )
    return chunks


async def persist_extracted_text_as_chunks(
    chunk_repository: ChunkRepository,
    *,
    kb: KnowledgeBase,
    doc: KnowledgeDocument,
    content_version: str,
    text: str,
) -> list[str]:
    """按知识库切块策略切分 ``text``，并对 ``replace_document_chunks`` 全量替换 Mongo 分片。

    与 Pipeline① 算法一致：经 :meth:`~app.knowledge.adapters.node_parser_factory.NodeParserFactory.split_text_for_ingest`。

    调用方须保证 ``content_version`` 与当前 ingest 任务及 MySQL 文档行一致。
    """
    _ = ingest_idempotency_key(kb_id=kb.id, doc_id=doc.id, content_version=content_version)
    chunks = NodeParserFactory.split_text_for_ingest(text, kb=kb, doc=doc)
    await chunk_repository.replace_document_chunks(
        kb_id=kb.id,
        doc_id=doc.id,
        content_version=content_version,
        chunks=chunks,
    )
    return chunks
