"""知识库向量写入编排。

职责：按知识库 embedding 配置调用多后端 HTTP 嵌入，经 :class:`~app.knowledge.kernel.ports.VectorIndexPort` 写入 Milvus。
ANN 检索由 ``knowledge.adapters.retrieval`` / ``KnowledgeRetrievalFacade`` 承担，本模块不实现检索。

对外导出 ``upsert_document_chunks_to_milvus``、``rebuild_kb_vectors_from_mongo``，供 ingest 与重建向量调用。
"""

from __future__ import annotations

from app.config import get_settings
from app.core.constants.knowledge import default_milvus_collection_name
from app.core.logger import get_logger
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.knowledge.adapters import get_vector_index_port
from app.knowledge.adapters.embedding_adapter import EmbeddingAdapter
from app.knowledge.kernel.ports import VectorIndexPort
from app.models.knowledge_mod import KnowledgeBase
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.knowledge_repo import KnowledgeDocumentRepository
from app.schemas.knowledge import KnowledgeVectorBuildData

logger = get_logger(__name__)


class KnowledgeVectorService:
    """向量写入：拉取 ``sys_model`` + 提供商构造嵌入客户端；经 ``VectorIndexPort`` 带 ``kb_id`` / ``doc_id`` / ``content_version`` 写入 Milvus。"""

    _EMBED_BATCH = 32

    def __init__(self, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._db = db_manager
        self._vector_index: VectorIndexPort = get_vector_index_port(get_settings())

    async def upsert_document_chunks_to_milvus(
        self,
        *,
        kb: KnowledgeBase,
        doc_id: int,
        content_version: str,
        chunks: list[str],
    ) -> None:
        """Mongo 分片写入成功后，按知识库配置将向量写入 Milvus（需 ``MILVUS_URI`` 与 vector/hybrid + embedding）。"""
        settings = get_settings()
        if not settings.milvus_configured:
            return
        rt = (kb.retrieval_type or "keyword").strip().lower()
        if rt not in ("vector", "hybrid"):
            return
        if kb.embedding_model_config_id is None:
            logger.warning(
                "kb vector mode but embedding_model_config_id is null | kb_id=%s",
                kb.id,
            )
            return

        embedder = await EmbeddingAdapter.client_for_model_config(
            self._db,
            kb.embedding_model_config_id,
        )
        coll = (kb.milvus_collection or "").strip() or default_milvus_collection_name(kb.id)

        if not chunks:
            await self._vector_index.delete_by_doc_version(
                collection_name=coll,
                kb_id=kb.id,
                doc_id=doc_id,
                content_version=content_version,
            )
            return

        all_vecs: list[list[float]] = []
        for i in range(0, len(chunks), self._EMBED_BATCH):
            batch = chunks[i : i + self._EMBED_BATCH]
            part = await embedder.embed_texts(batch)
            all_vecs.extend(part)

        dim = len(all_vecs[0])
        await self._vector_index.delete_by_doc_version(
            collection_name=coll,
            kb_id=kb.id,
            doc_id=doc_id,
            content_version=content_version,
        )
        await self._vector_index.insert_vectors(
            collection_name=coll,
            dim=dim,
            kb_id=kb.id,
            doc_id=doc_id,
            content_version=content_version,
            vectors=all_vecs,
            texts=chunks,
        )

    async def rebuild_kb_vectors_from_mongo(
        self,
        *,
        chunk_repository: ChunkRepository,
        kb: KnowledgeBase,
    ) -> KnowledgeVectorBuildData:
        """在 Mongo 已有分片的前提下，按当前知识库配置调用 embedding 并写入 Milvus（不重新 ingest）。"""
        settings = get_settings()
        if not settings.milvus_configured:
            raise ValueError("未配置 MILVUS_URI，无法写入向量库")
        rt = (kb.retrieval_type or "keyword").strip().lower()
        if rt not in ("vector", "hybrid"):
            raise ValueError("检索类型须为 vector 或 hybrid")
        if kb.embedding_model_config_id is None:
            raise ValueError("须先配置 embedding 模型（embedding_model_config_id）")

        docs = await KnowledgeDocumentRepository.list_indexed_with_chunks_by_kb(
            kb.id,
            db_manager=self._db,
        )
        built = 0
        skipped = 0
        errors: list[str] = []

        for doc in docs:
            cv = (doc.content_version or "").strip()
            if not cv:
                skipped += 1
                continue
            texts = await chunk_repository.list_all_chunk_texts_for_version(
                kb_id=kb.id,
                doc_id=doc.id,
                content_version=cv,
            )
            if not texts:
                skipped += 1
                continue
            try:
                await self.upsert_document_chunks_to_milvus(
                    kb=kb,
                    doc_id=doc.id,
                    content_version=cv,
                    chunks=texts,
                )
                built += 1
            except Exception as e:
                logger.exception("rebuild_vectors doc failed kb_id=%s doc_id=%s", kb.id, doc.id)
                errors.append(f"文档 #{doc.id}（{doc.filename}）：{e}")

        return KnowledgeVectorBuildData(built=built, skipped=skipped, errors=errors)


# --- 模块级 API（供 ``knowledge_svc`` / ``ingest`` 导入） ---


async def upsert_document_chunks_to_milvus(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    kb: KnowledgeBase,
    doc_id: int,
    content_version: str,
    chunks: list[str],
) -> None:
    await KnowledgeVectorService(db_manager).upsert_document_chunks_to_milvus(
        kb=kb,
        doc_id=doc_id,
        content_version=content_version,
        chunks=chunks,
    )


async def rebuild_kb_vectors_from_mongo(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    chunk_repository: ChunkRepository,
    kb: KnowledgeBase,
) -> KnowledgeVectorBuildData:
    return await KnowledgeVectorService(db_manager).rebuild_kb_vectors_from_mongo(
        chunk_repository=chunk_repository,
        kb=kb,
    )
