"""知识库 ingest：``knowledge_task`` 与后台协程衔接。

入队（幂等）、状态机、:func:`process_document_ingest` 编排；实现为 ``asyncio.create_task``。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.core.constants.knowledge import (
    DOCUMENT_STATUS_FAILED,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_PROCESSING,
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCEEDED,
)
from app.core.logger import get_logger
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.infrastructure.minio import MinioObjectStore
from app.knowledge.adapters.ingestion_vector_pipeline import run_vector_ingestion_pipeline
from app.knowledge.application.indexing import (
    persist_upload_bytes_as_chunks,
    should_skip_ingest_derived_writes,
)
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.knowledge_repo import (
    KnowledgeBaseRepository,
    KnowledgeDocumentRepository,
    KnowledgeTaskRepository,
)

logger = get_logger(__name__)
settings = get_settings()

# ------------------------------------------------------------------------------
# 对外入口：任务入队
# ------------------------------------------------------------------------------


async def enqueue_ingest_after_upload(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    object_store: MinioObjectStore,
    chunk_repository: ChunkRepository,
    kb_id: int,
    doc_id: int,
    content_version: str,
    ingest_task_payload: dict[str, Any] | None = None,
) -> str | None:
    """上传完成后入队 ingest；同一 ``content_version`` 幂等去重。

    ``ingest_task_payload`` 可含 ``skip_milvus_upsert: true``（仅写 Mongo 分片，不调用 embedding/Milvus）。

    :returns: 新建任务的 ``task_id``；若幂等未创建则 ``None``。
    """
    task = await KnowledgeTaskRepository.create_ingest_task(
        kb_id=kb_id,
        doc_id=doc_id,
        content_version=content_version,
        payload=ingest_task_payload,
        db_manager=db_manager,
    )
    if task is None:
        logger.info(
            "ingest 任务已幂等去重",
            extra={"doc_id": doc_id, "content_version": content_version},
        )
        return None

    coro = process_document_ingest(
        db_manager=db_manager,
        object_store=object_store,
        chunk_repository=chunk_repository,
        task_id=task.task_id,
    )
    handle = asyncio.create_task(coro)
    handle.add_done_callback(_log_ingest_task_exception)

    logger.info(
        "ingest 任务已调度",
        extra={"task_id": task.task_id, "doc_id": doc_id},
    )
    return task.task_id


def _log_ingest_task_exception(t: asyncio.Task[Any]) -> None:
    try:
        exc = t.exception()
        if exc is not None:
            logger.exception(
                "ingest 后台任务未捕获异常（应在 IngestTaskHandler 内处理）",
                exc_info=exc,
            )
    except asyncio.CancelledError:
        pass


# ------------------------------------------------------------------------------
# 任务执行入口
# ------------------------------------------------------------------------------


async def process_document_ingest(
    *,
    db_manager: SQLAlchemyDatabaseManager,
    object_store: MinioObjectStore,
    chunk_repository: ChunkRepository,
    task_id: str,
) -> None:
    """执行单条 ingest 任务（供 ``asyncio.create_task`` 或单测直接调用）。"""
    handler = IngestTaskHandler(
        db_manager=db_manager,
        object_store=object_store,
        chunk_repo=chunk_repository,
        task_id=task_id,
    )
    await handler.run()


# ------------------------------------------------------------------------------
# 核心处理器
# ------------------------------------------------------------------------------


class IngestTaskHandler:
    """单任务编排：步骤拆分，便于阅读与单测打桩。"""

    def __init__(
        self,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        object_store: MinioObjectStore,
        chunk_repo: ChunkRepository,
        task_id: str,
    ) -> None:
        self._db = db_manager
        self._object_store = object_store
        self._chunk_repo = chunk_repo
        self._task_id = task_id

        self.task: Any = None
        self.doc: Any = None
        self.kb: Any = None

        self.doc_id: int | None = None
        self.kb_id: int | None = None
        self.content_version: str | None = None
        self._skip_milvus_upsert: bool = False

    async def run(self) -> None:
        try:
            if not await self._load_task():
                return
            self._skip_milvus_upsert = bool((self.task.payload or {}).get("skip_milvus_upsert"))

            await self._load_entities()
            assert self.content_version is not None and self.kb is not None and self.doc is not None
            if await should_skip_ingest_derived_writes(
                self._chunk_repo,
                kb=self.kb,
                doc=self.doc,
                content_version=self.content_version,
            ):
                await self._succeed_skip_derived_complete()
                return

            await self._mark_processing()
            raw_bytes = await self._read_from_minio()
            chunks = await persist_upload_bytes_as_chunks(
                self._chunk_repo,
                kb=self.kb,
                doc=self.doc,
                content_version=self.content_version,
                raw=raw_bytes,
            )

            if not chunks:
                await self._fail("分片结果为空：文件无文本、解析结果为空或切块后无有效片段")
                return

            await self._try_upsert_milvus(chunks)
            await self._succeed(len(chunks))
        except Exception as e:
            logger.exception(
                "ingest 执行失败 | task_id=%s doc_id=%s",
                self._task_id,
                self.doc_id,
            )
            await self._fail(str(e)[:1024])

    async def _load_task(self) -> bool:
        self.task = await KnowledgeTaskRepository.get_by_task_id(
            self._task_id,
            db_manager=self._db,
        )
        if self.task is None or self.task.doc_id is None:
            logger.warning("ingest 任务不存在", extra={"task_id": self._task_id})
            return False

        self.doc_id = self.task.doc_id
        self.kb_id = self.task.kb_id
        self.content_version = self.task.content_version
        return True

    async def _mark_processing(self) -> None:
        assert self.task is not None
        nxt = int(self.task.attempts or 0) + 1
        await KnowledgeTaskRepository.update_fields_by_task_id(
            self._task_id,
            db_manager=self._db,
            status=TASK_STATUS_RUNNING,
            started_at=datetime.now(),
            attempts=nxt,
            last_heartbeat_at=datetime.now(),
        )
        assert self.doc_id is not None
        await KnowledgeDocumentRepository.update_fields_by_id(
            self.doc_id,
            db_manager=self._db,
            status=DOCUMENT_STATUS_PROCESSING,
        )

    async def _load_entities(self) -> None:
        assert self.doc_id is not None and self.kb_id is not None
        self.doc = await KnowledgeDocumentRepository.get_by_id(
            self.doc_id,
            db_manager=self._db,
        )
        self.kb = await KnowledgeBaseRepository.get_active_by_id(
            self.kb_id,
            db_manager=self._db,
        )
        if self.doc is None or self.doc.deleted_at is not None or self.kb is None:
            raise RuntimeError("文档或知识库已删除/不存在")
        if not self.doc.object_key:
            raise RuntimeError("object_key 为空，无法读取对象")

    async def _read_from_minio(self) -> bytes:
        assert self.doc is not None
        return await self._object_store.get_object(self.doc.object_key)

    async def _try_upsert_milvus(self, chunks: list[str]) -> None:
        assert self.kb is not None and self.doc_id is not None and self.content_version is not None
        if self._skip_milvus_upsert:
            logger.info(
                "ingest 跳过向量写入（任务 skip_milvus_upsert）；向量/混合检索请在索引步骤执行「从 Mongo 构建向量」",
                extra={"task_id": self._task_id, "doc_id": self.doc_id},
            )
            return
        await run_vector_ingestion_pipeline(
            db_manager=self._db,
            kb=self.kb,
            doc_id=self.doc_id,
            content_version=self.content_version,
            chunks=chunks,
        )

    async def _succeed(self, chunk_count: int) -> None:
        assert self.doc_id is not None and self.kb_id is not None
        await KnowledgeDocumentRepository.update_fields_by_id(
            self.doc_id,
            db_manager=self._db,
            status=DOCUMENT_STATUS_INDEXED,
            chunk_count=chunk_count,
        )
        await KnowledgeBaseRepository.patch_by_id(
            self.kb_id,
            db_manager=self._db,
            status="ready",
        )

        vector_eligible = (
            settings.milvus_configured
            and self.kb is not None
            and (self.kb.retrieval_type or "keyword").strip().lower() in ("vector", "hybrid")
            and self.kb.embedding_model_config_id is not None
        )
        if self._skip_milvus_upsert:
            milvus_payload = "deferred" if vector_eligible else "skipped"
        elif vector_eligible:
            milvus_payload = "ok"
        else:
            milvus_payload = "skipped"

        await KnowledgeTaskRepository.update_fields_by_task_id(
            self._task_id,
            db_manager=self._db,
            status=TASK_STATUS_SUCCEEDED,
            finished_at=datetime.now(),
            payload={
                "milvus": milvus_payload,
                "mongo_written": chunk_count > 0,
                "chunk_count": chunk_count,
            },
        )
        logger.info(
            "ingest 完成",
            extra={"task_id": self._task_id, "doc_id": self.doc_id, "chunks": chunk_count},
        )

    async def _fail(self, msg: str) -> None:
        err = msg[:1024]
        if self.doc_id is not None:
            await KnowledgeDocumentRepository.update_fields_by_id(
                self.doc_id,
                db_manager=self._db,
                status=DOCUMENT_STATUS_FAILED,
            )
        await KnowledgeTaskRepository.update_fields_by_task_id(
            self._task_id,
            db_manager=self._db,
            status=TASK_STATUS_FAILED,
            finished_at=datetime.now(),
            error_msg=err,
            checkpoint_json={"phase": "ingest"},
        )

    async def _succeed_skip_derived_complete(self) -> None:
        """权威行 + Mongo 已对齐且非向量 Milvus 路径时跳过重复写入。"""
        assert self.doc is not None
        await KnowledgeTaskRepository.update_fields_by_task_id(
            self._task_id,
            db_manager=self._db,
            status=TASK_STATUS_SUCCEEDED,
            finished_at=datetime.now(),
            last_heartbeat_at=datetime.now(),
            payload={
                "skipped": True,
                "reason": "derived_already_complete",
                "chunk_count": self.doc.chunk_count,
            },
        )
        logger.info(
            "ingest 跳过（派生已完整）",
            extra={
                "task_id": self._task_id,
                "doc_id": self.doc.id,
                "content_version": self.content_version,
            },
        )
