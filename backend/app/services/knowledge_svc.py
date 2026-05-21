"""知识库 HTTP 侧编排（与 ``provider_svc`` / ``agent_svc`` 同级）：元数据 CRUD、文档上传与列表、分片预览与检索；路由经 ``api/knowledge_api`` ``Depends`` 注入。

文档 ingest 入队与向量重建见 :mod:`app.services.knowledge.ingest`、:mod:`app.services.knowledge.vector`。
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.constants.knowledge import (
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_PENDING,
    DOCUMENT_STATUS_PROCESSING,
    KNOWLEDGE_UPLOAD_MAX_BYTES,
    MAX_KNOWLEDGE_UPLOAD_FILENAME_LENGTH,
    MAX_KNOWLEDGE_UPLOAD_MIME_LENGTH,
    default_milvus_collection_name,
)
from app.core.logger import get_logger
from app.domain.knowledge.chunk_config import (
    _normalize_method as normalize_doc_chunk_method,
)
from app.domain.knowledge.chunk_config import (
    apply_chunk_strategy_to_updates,
    orm_updates_from_chunk_strategy,
)
from app.domain.knowledge.mime import normalize_upload_mime
from app.domain.knowledge.object_key import build_knowledge_document_object_key
from app.domain.knowledge.slug import is_valid_slug, propose_slug_from_name
from app.domain.knowledge.upload_constraints import validate_upload_byte_size
from app.domain.knowledge.upload_filename import safe_upload_filename
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.infrastructure.minio import MinioObjectStore
from app.knowledge.adapters import get_vector_index_port
from app.knowledge.facades import KnowledgeRetrievalFacade
from app.models.knowledge_mod import KnowledgeBase, KnowledgeDocument
from app.models.sys_model_mod import SysModel
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.knowledge_repo import (
    KnowledgeBaseRepository,
    KnowledgeDocumentRepository,
    KnowledgeTaskRepository,
)
from app.repositories.namespace_repo import WorkspaceNamespaceRepository
from app.schemas.knowledge import (
    KnowledgeBaseOut,
    KnowledgeChunkOut,
    KnowledgeCreateBody,
    KnowledgeDocumentChunksData,
    KnowledgeDocumentIngestAccepted,
    KnowledgeDocumentListData,
    KnowledgeDocumentOut,
    KnowledgeDocumentPatchBody,
    KnowledgeListData,
    KnowledgeSearchData,
    KnowledgeUpdateBody,
    KnowledgeVectorBuildData,
    RetrievalType,
)
from app.services.knowledge.ingest import enqueue_ingest_after_upload
from app.services.knowledge.vector import rebuild_kb_vectors_from_mongo

logger = get_logger(__name__)


def get_chunk_repository(request: Request) -> ChunkRepository:
    """从 ``app.state`` 取进程内注册的 :class:`~app.repositories.chunk_repo.ChunkRepository`。"""
    repo = getattr(request.app.state, "chunk_repository", None)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="app.state.chunk_repository 未初始化",
        )
    return repo


ChunkRepositoryDep = Annotated[ChunkRepository, Depends(get_chunk_repository)]

# ------------------------------------------------------------------------------
# 工具函数（纯函数）
# ------------------------------------------------------------------------------


_CHUNK_EMPTY_HINTS: dict[str, str] = {
    "pending": "文档仍在排队或尚未开始 ingest，请稍后刷新预览；若长时间无变化，请使用「重新触发索引」。",
    "processing": "正在从对象存储读取并切块写入 Mongo，请稍候再试。",
    "failed": "索引失败，请查看服务端日志或删除后重新上传，也可尝试「重新触发索引」。",
    "indexed": "状态为 indexed 但当前 content_version 下无分片记录；可能曾为空文本、或仅修改切块策略后尚未重建。",
}


def _effective_chunk_signature(
    *, kb: KnowledgeBase, doc: KnowledgeDocument
) -> tuple[str, int, int, str]:
    """文档在 ingest 时生效的切块列字段（覆盖优先，否则知识库默认）。"""
    dm = (doc.chunk_method or "").strip()
    km = (kb.chunk_method or "").strip()
    raw_m = dm or km or "length"
    m = normalize_doc_chunk_method(raw_m)
    method = m
    if m == "length":
        method = "char"
    size = int(doc.chunk_size if doc.chunk_size is not None else kb.chunk_size)
    overlap = int(doc.chunk_overlap if doc.chunk_overlap is not None else kb.chunk_overlap)
    sep_src = doc.chunk_separator if doc.chunk_separator is not None else kb.chunk_separator
    sep = (sep_src or "").strip()
    return (method, size, overlap, sep)


# ------------------------------------------------------------------------------
# 知识库（元数据）
# ------------------------------------------------------------------------------


class KnowledgeBaseService:
    """知识库元数据：列表、创建、更新、软删；检索类型与切块策略校验。"""

    def __init__(self, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._db = db_manager
        self._repo = KnowledgeBaseRepository

    async def _get_active_kb_or_404(self, kb_id: int) -> KnowledgeBase:
        row = await self._repo.get_active_by_id(kb_id, db_manager=self._db)
        if row is None:
            raise LookupError("知识库不存在或已删除")
        return row

    async def _ensure_milvus_collection(self, row: KnowledgeBase) -> KnowledgeBase:
        """若 ``milvus_collection`` 为空则写入默认 ``kb_{id}_vectors``（幂等、便于旧数据补全）。"""
        if (row.milvus_collection or "").strip():
            return row
        patched = await self._repo.patch_by_id(
            row.id,
            db_manager=self._db,
            milvus_collection=default_milvus_collection_name(row.id),
        )
        return patched if patched is not None else row

    @staticmethod
    def _validate_create_retrieval_embedding(body: KnowledgeCreateBody) -> None:
        if (
            str(body.retrieval_type) in ("vector", "hybrid")
            and body.embedding_model_config_id is None
        ):
            raise ValueError("向量或混合检索须配置 embedding_model_config_id")

    async def _validate_update_retrieval_embedding(
        self, kb_id: int, body: KnowledgeUpdateBody
    ) -> None:
        fs = body.model_fields_set
        if "retrieval_type" not in fs and "embedding_model_config_id" not in fs:
            return
        current = await self._repo.get_active_by_id(kb_id, db_manager=self._db)
        if current is None:
            return
        rt = body.retrieval_type if body.retrieval_type is not None else current.retrieval_type
        emb = (
            body.embedding_model_config_id
            if "embedding_model_config_id" in fs
            else current.embedding_model_config_id
        )
        if str(rt) in ("vector", "hybrid") and emb is None:
            raise ValueError("向量或混合检索须配置 embedding_model_config_id")

    async def _embedding_model_exists(self, model_id: int) -> bool:
        async def _read(session: AsyncSession) -> bool:
            row = await session.get(SysModel, model_id)
            return row is not None

        return await self._repo.run_read(_read, self._db)

    async def _apply_chunk_strategy_updates(
        self, kb_id: int, body: KnowledgeUpdateBody, updates: dict[str, Any]
    ) -> dict[str, Any]:
        if body.chunk_strategy is None:
            updates.pop("chunk_strategy", None)
            return updates
        cs = body.chunk_strategy
        current = await self._get_active_kb_or_404(kb_id)
        client_cj = updates.pop("config_json", None)
        updates.pop("chunk_strategy", None)
        for k in ("chunk_method", "chunk_size", "chunk_overlap"):
            updates.pop(k, None)
        merged_cj = apply_chunk_strategy_to_updates(
            cfg=cs,
            base_config_json=current.config_json if isinstance(current.config_json, dict) else None,
            client_config_json=client_cj if isinstance(client_cj, dict) else None,
        )
        updates["config_json"] = merged_cj
        updates.update(orm_updates_from_chunk_strategy(cs))
        return updates

    def _generate_valid_slug(self, body: KnowledgeCreateBody) -> str:
        if body.slug:
            if not is_valid_slug(body.slug):
                raise ValueError("slug 须为小写字母、数字与连字符，长度 2～128")
            return body.slug
        return propose_slug_from_name(body.name)

    async def list_knowledge_bases(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        status: str | None = None,
        workspace_namespace: str | None = None,
    ) -> KnowledgeListData:
        page_res = await self._repo.list_page(
            page=page,
            page_size=page_size,
            q=q,
            status=status,
            workspace_namespace=workspace_namespace,
            db_manager=self._db,
        )
        items = [KnowledgeBaseOut.model_validate(r) for r in page_res.items]
        return KnowledgeListData(
            items=items,
            total=page_res.total,
            page=page_res.page,
            page_size=page_res.page_size,
        )

    async def get_knowledge_base(self, kb_id: int) -> KnowledgeBaseOut:
        row = await self._get_active_kb_or_404(kb_id)
        row = await self._ensure_milvus_collection(row)
        return KnowledgeBaseOut.model_validate(row)

    async def create_knowledge_base(self, body: KnowledgeCreateBody) -> KnowledgeBaseOut:
        if body.embedding_model_config_id is not None:
            if not await self._embedding_model_exists(body.embedding_model_config_id):
                raise ValueError("embedding_model_config_id 不存在")
        self._validate_create_retrieval_embedding(body)

        slug = self._generate_valid_slug(body)

        wn = await WorkspaceNamespaceRepository.get_or_create_by_slug(
            body.workspace_namespace,
            db_manager=self._db,
        )
        if await self._repo.exists_by(
            slug=slug,
            namespace_id=wn.id,
            deleted_at=None,
            db_manager=self._db,
        ):
            raise ValueError("slug 已存在，请更换名称或手动指定其它 slug")

        payload = body.model_dump(exclude={"slug", "workspace_namespace"}, exclude_unset=False)
        payload["slug"] = slug
        payload["namespace_id"] = wn.id
        row = await self._repo.create(db_manager=self._db, **payload)
        row = await self._repo.get_active_by_id(row.id, db_manager=self._db) or row
        row = await self._ensure_milvus_collection(row)
        return KnowledgeBaseOut.model_validate(row)

    async def update_knowledge_base(
        self, kb_id: int, body: KnowledgeUpdateBody
    ) -> KnowledgeBaseOut:
        if body.embedding_model_config_id is not None and not await self._embedding_model_exists(
            body.embedding_model_config_id
        ):
            raise ValueError("embedding_model_config_id 不存在")
        await self._validate_update_retrieval_embedding(kb_id, body)

        updates = body.model_dump(exclude_unset=True, mode="python")
        updates = await self._apply_chunk_strategy_updates(kb_id, body, updates)

        if not updates:
            return await self.get_knowledge_base(kb_id)

        row = await self._repo.patch_by_id(
            kb_id,
            db_manager=self._db,
            **updates,
        )
        if row is None:
            raise LookupError("知识库不存在或已删除")
        fs = body.model_fields_set
        if not (row.milvus_collection or "").strip() and "milvus_collection" not in fs:
            row = await self._ensure_milvus_collection(row)
        return KnowledgeBaseOut.model_validate(row)

    async def delete_knowledge_base(self, kb_id: int) -> None:
        ok = await self._repo.soft_delete_kb(kb_id, db_manager=self._db)
        if not ok:
            raise LookupError("知识库不存在或已删除")
        logger.warning(
            "知识库已软删，派生存储清理待后台任务",
            extra={"kb_id": kb_id, "reason": "placeholder_pending_worker"},
        )


# ------------------------------------------------------------------------------
# 知识库文档（上传、分片、检索）
# ------------------------------------------------------------------------------


class KnowledgeDocumentService:
    """知识库文档：列表、切块覆盖、重新 ingest、上传、分片预览、检索。"""

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager,
        object_store: MinioObjectStore,
        chunk_repository: ChunkRepository,
    ) -> None:
        self._db = db_manager
        self._object_store = object_store
        self._chunk_repository = chunk_repository
        self._kb_repo = KnowledgeBaseRepository
        self._doc_repo = KnowledgeDocumentRepository
        self._retrieval = KnowledgeRetrievalFacade(db_manager, chunk_repository)

    async def _require_kb(self, kb_id: int) -> KnowledgeBase:
        kb = await self._kb_repo.get_active_by_id(kb_id, db_manager=self._db)
        if kb is None:
            raise LookupError("知识库不存在或已删除")
        return kb

    async def _get_doc_or_404(self, doc_id: int, kb_id: int) -> KnowledgeDocument:
        doc = await self._doc_repo.get_by_id(doc_id, db_manager=self._db)
        if doc is None or doc.kb_id != kb_id:
            raise LookupError("文档不存在或已删除")
        return doc

    async def _purge_document_derived_for_reingest(
        self,
        kb: KnowledgeBase,
        doc_id: int,
        content_version: str,
    ) -> None:
        """重新 ingest 前清空当前版本派生数据：Mongo 分片；向量/混合时尽力删 Milvus，避免排队期间仍命中旧向量。"""
        cv = (content_version or "").strip()
        if not cv:
            return
        await self._chunk_repository.replace_document_chunks(
            kb_id=kb.id,
            doc_id=doc_id,
            content_version=cv,
            chunks=[],
        )
        settings = get_settings()
        if not settings.milvus_configured:
            return
        rt = (kb.retrieval_type or "keyword").strip().lower()
        if rt not in ("vector", "hybrid") or kb.embedding_model_config_id is None:
            return
        try:
            port = get_vector_index_port(settings)
            coll = (kb.milvus_collection or "").strip() or default_milvus_collection_name(kb.id)
            await port.delete_by_doc_version(
                collection_name=coll,
                kb_id=kb.id,
                doc_id=doc_id,
                content_version=cv,
            )
        except Exception as e:
            logger.warning(
                "reingest 前清理 Milvus 失败（后续 ingest 仍会 delete+insert 覆盖）",
                extra={"kb_id": kb.id, "doc_id": doc_id, "error": str(e)},
            )

    async def list_documents(
        self,
        kb_id: int,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        chunk_left_panel: bool = False,
    ) -> KnowledgeDocumentListData:
        await self._require_kb(kb_id)

        page_res = await self._doc_repo.list_page_by_kb(
            kb_id,
            page=page,
            page_size=page_size,
            q=q,
            chunk_left_panel=chunk_left_panel,
            db_manager=self._db,
        )
        items = [KnowledgeDocumentOut.model_validate(r) for r in page_res.items]
        return KnowledgeDocumentListData(
            items=items,
            total=page_res.total,
            page=page_res.page,
            page_size=page_res.page_size,
        )

    async def patch_document(
        self,
        kb_id: int,
        doc_id: int,
        body: KnowledgeDocumentPatchBody,
    ) -> KnowledgeDocumentOut:
        """文档级切块覆盖；``chunk_method`` 显式 ``null`` 表示清除覆盖（恢复知识库默认）。"""
        kb = await self._require_kb(kb_id)
        doc = await self._get_doc_or_404(doc_id, kb_id)

        fs = body.model_fields_set
        if "chunk_method" in fs and body.chunk_method is None:
            sig_before = _effective_chunk_signature(kb=kb, doc=doc)
            row = await self._doc_repo.update_fields_by_id(
                doc_id,
                db_manager=self._db,
                chunk_method=None,
                chunk_size=None,
                chunk_overlap=None,
                chunk_separator=None,
            )
            if row is None:
                raise LookupError("文档不存在或已删除")
            sig_after = _effective_chunk_signature(kb=kb, doc=row)
            needs_reingest = (
                sig_before != sig_after
                and row.status
                in (
                    DOCUMENT_STATUS_INDEXED,
                    DOCUMENT_STATUS_PROCESSING,
                )
                and (row.object_key or "").strip()
                and (row.content_version or "").strip()
            )
            if needs_reingest:
                logger.info(
                    "knowledge.document.clear_chunk_override_requeue_ingest",
                    extra={
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "sig_before": sig_before,
                        "sig_after": sig_after,
                        "prior_status": row.status,
                    },
                )
                await self.requeue_document_ingest(
                    kb_id,
                    doc_id,
                    skip_milvus_upsert=True,
                )
                row = await self._doc_repo.get_by_id(doc_id, db_manager=self._db)
                if row is None:
                    raise LookupError("文档不存在或已删除")
            return KnowledgeDocumentOut.model_validate(row)

        if not fs:
            return KnowledgeDocumentOut.model_validate(doc)

        if "chunk_method" in fs and body.chunk_method is not None:
            method = normalize_doc_chunk_method(body.chunk_method)
        elif doc.chunk_method:
            method = normalize_doc_chunk_method(doc.chunk_method)
        else:
            method = normalize_doc_chunk_method(kb.chunk_method)

        size = (
            body.chunk_size
            if "chunk_size" in fs
            else (doc.chunk_size if doc.chunk_size is not None else kb.chunk_size)
        )
        overlap = (
            body.chunk_overlap
            if "chunk_overlap" in fs
            else (doc.chunk_overlap if doc.chunk_overlap is not None else kb.chunk_overlap)
        )
        sep_raw = (
            body.chunk_separator
            if "chunk_separator" in fs
            else (doc.chunk_separator if doc.chunk_separator is not None else kb.chunk_separator)
        )
        sep = (sep_raw or "").strip() or None

        if int(overlap) >= int(size):
            raise ValueError("chunk_overlap 须小于 chunk_size")

        row = await self._doc_repo.update_fields_by_id(
            doc_id,
            db_manager=self._db,
            chunk_method=method,
            chunk_size=int(size),
            chunk_overlap=int(overlap),
            chunk_separator=sep,
        )
        if row is None:
            raise LookupError("文档不存在或已删除")
        return KnowledgeDocumentOut.model_validate(row)

    async def requeue_document_ingest(
        self,
        kb_id: int,
        doc_id: int,
        *,
        skip_milvus_upsert: bool = False,
    ) -> KnowledgeDocumentIngestAccepted:
        """删除同版本 ingest 幂等任务并重新入队；入队前先清空当前版本 Mongo 分片（及向量库中同版本行）。

        ``skip_milvus_upsert=True``：仅重切并写 Mongo，不调用 embedding（用于「恢复默认」等场景；向量需另做「从 Mongo 构建向量」）。
        """
        kb = await self._require_kb(kb_id)
        doc = await self._get_doc_or_404(doc_id, kb_id)
        object_key = (doc.object_key or "").strip()
        if not object_key:
            raise ValueError("object_key 为空，无法重新索引")
        cv = (doc.content_version or "").strip()
        if not cv:
            raise ValueError("content_version 为空，无法重新索引")

        await self._purge_document_derived_for_reingest(kb, doc_id, cv)

        idempotency_key = f"ingest:{doc_id}:{cv}"
        await KnowledgeTaskRepository.delete_by_idempotency_key(
            idempotency_key,
            db_manager=self._db,
        )
        await self._doc_repo.update_fields_by_id(
            doc_id,
            db_manager=self._db,
            status=DOCUMENT_STATUS_PENDING,
            chunk_count=0,
        )
        ingest_payload = {"skip_milvus_upsert": True} if skip_milvus_upsert else None
        task_id = await enqueue_ingest_after_upload(
            db_manager=self._db,
            object_store=self._object_store,
            chunk_repository=self._chunk_repository,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=cv,
            ingest_task_payload=ingest_payload,
        )
        row = await self._doc_repo.get_by_id(doc_id, db_manager=self._db)
        if row is None:
            raise LookupError("文档不存在或已删除")
        return KnowledgeDocumentIngestAccepted(
            document=KnowledgeDocumentOut.model_validate(row),
            task_id=task_id,
        )

    async def build_vectors_from_mongo_chunks(self, kb_id: int) -> KnowledgeVectorBuildData:
        """从 Mongo 已有分片构建向量并写入 Milvus（不重跑 ingest、不读 MinIO）。"""
        kb = await self._require_kb(kb_id)
        return await rebuild_kb_vectors_from_mongo(
            db_manager=self._db,
            chunk_repository=self._chunk_repository,
            kb=kb,
        )

    async def delete_document(self, kb_id: int, doc_id: int) -> None:
        """软删文档，并尽力清理 Mongo 分片与 MinIO 对象。"""
        await self._require_kb(kb_id)
        doc = await self._get_doc_or_404(doc_id, kb_id)
        object_key = (doc.object_key or "").strip() or None

        ok = await self._doc_repo.soft_delete_by_id(doc_id, db_manager=self._db)
        if not ok:
            raise LookupError("文档不存在或已删除")

        removed = await self._chunk_repository.delete_all_chunks_for_document(
            kb_id=kb_id,
            doc_id=doc_id,
        )
        logger.info(
            "knowledge.document.deleted",
            extra={"kb_id": kb_id, "doc_id": doc_id, "mongo_chunks_removed": removed},
        )

        if object_key and self._object_store.is_configured:
            try:
                await self._object_store.remove_object(object_key)
            except Exception as e:
                logger.warning(
                    "删除 MinIO 对象失败（文档已软删）",
                    extra={"object_key": object_key, "error": str(e)},
                )

    async def list_document_chunks(
        self,
        kb_id: int,
        doc_id: int,
        *,
        content_version: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> KnowledgeDocumentChunksData:
        await self._require_kb(kb_id)
        doc = await self._get_doc_or_404(doc_id, kb_id)
        cv = (content_version or "").strip() or (doc.content_version or "")
        page_n = max(1, page)
        size_n = max(1, min(page_size, 100))
        skip = (page_n - 1) * size_n
        if not cv:
            return KnowledgeDocumentChunksData(
                items=[],
                total=0,
                page=page_n,
                page_size=size_n,
                doc_id=doc_id,
                filename=doc.filename,
                content_version="",
                document_status=doc.status,
                empty_hint="content_version 为空，无法关联 Mongo 中的分片版本。",
            )
        raw, total = await self._chunk_repository.list_document_chunks(
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=cv,
            skip=skip,
            limit=size_n,
        )
        items: list[KnowledgeChunkOut] = []
        for r in raw:
            items.append(
                KnowledgeChunkOut(
                    chunk_index=int(r.get("chunk_index", 0)),
                    text=str(r.get("text") or ""),
                    created_at=r.get("created_at"),
                )
            )

        def _chunks_empty_hint(document_status: str | None, *, total: int) -> str | None:
            if total > 0:
                return None
            s = (document_status or "").strip().lower()
            return _CHUNK_EMPTY_HINTS.get(s, "暂无分片数据。")

        return KnowledgeDocumentChunksData(
            items=items,
            total=total,
            page=page_n,
            page_size=size_n,
            doc_id=doc_id,
            filename=doc.filename,
            content_version=cv,
            document_status=doc.status,
            empty_hint=_chunks_empty_hint(doc.status, total=total),
        )

    async def search_in_kb(
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
        """委托 :class:`~app.knowledge.facades.KnowledgeRetrievalFacade`。"""
        return await self._retrieval.retrieve(
            kb_id,
            q=q,
            limit=limit,
            doc_id=doc_id,
            retrieval_override=retrieval_override,
            recall_limit=recall_limit,
            rrf_k=rrf_k,
        )

    async def upload_document(
        self,
        kb_id: int,
        *,
        original_filename: str,
        content_type: str | None,
        data: bytes,
    ) -> KnowledgeDocumentIngestAccepted:
        self._validate_upload_preconditions(data)
        kb = await self._require_kb(kb_id)

        safe_name = safe_upload_filename(
            original_filename, max_len=MAX_KNOWLEDGE_UPLOAD_FILENAME_LENGTH
        )
        sha256_hex = hashlib.sha256(data).hexdigest()
        object_key = build_knowledge_document_object_key(
            minio_prefix=kb.minio_prefix,
            kb_id=kb_id,
            safe_filename=safe_name,
            folder_hex=uuid.uuid4().hex,
        )
        mime = normalize_upload_mime(
            content_type,
            max_len=MAX_KNOWLEDGE_UPLOAD_MIME_LENGTH,
        )

        await self._upload_to_minio(object_key, data, mime)

        try:
            out = await self._persist_document_row(
                kb_id=kb_id,
                filename=safe_name,
                object_key=object_key,
                size_bytes=len(data),
                mime=mime,
                sha256_hex=sha256_hex,
            )
            task_id = await enqueue_ingest_after_upload(
                db_manager=self._db,
                object_store=self._object_store,
                chunk_repository=self._chunk_repository,
                kb_id=kb_id,
                doc_id=out.id,
                content_version=out.content_version,
            )
        except Exception:
            await self._object_store.remove_object(object_key)
            logger.warning(
                "文档入库失败，已回滚 MinIO 对象",
                extra={"object_key": object_key},
            )
            raise

        logger.info(
            "knowledge.document.uploaded",
            extra={"kb_id": kb_id, "object_key": object_key, "doc_filename": safe_name},
        )
        return KnowledgeDocumentIngestAccepted(document=out, task_id=task_id)

    def _validate_upload_preconditions(self, data: bytes) -> None:
        validate_upload_byte_size(len(data), max_bytes=KNOWLEDGE_UPLOAD_MAX_BYTES)
        if not self._object_store.is_configured:
            raise ValueError(
                "MinIO 未配置：请设置 MINIO_ENDPOINT、MINIO_ACCESS_KEY、MINIO_SECRET_KEY",
            )

    async def _upload_to_minio(
        self,
        object_key: str,
        data: bytes,
        mime: str | None,
    ) -> None:
        try:
            await self._object_store.put_object(
                object_key=object_key,
                data=data,
                content_type=mime,
            )
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "MinIO 上传失败",
                extra={"object_key": object_key, "error": str(e)},
            )
            raise RuntimeError(f"上传对象到 MinIO 失败：{e}") from e

    async def _persist_document_row(
        self,
        *,
        kb_id: int,
        filename: str,
        object_key: str,
        size_bytes: int,
        mime: str | None,
        sha256_hex: str,
    ) -> KnowledgeDocumentOut:
        row = await self._doc_repo.create(
            db_manager=self._db,
            kb_id=kb_id,
            filename=filename,
            object_key=object_key,
            size_bytes=size_bytes,
            mime=mime,
            sha256=sha256_hex,
            status=DOCUMENT_STATUS_PENDING,
            content_version=sha256_hex,
            chunk_count=0,
        )
        return KnowledgeDocumentOut.model_validate(row)
