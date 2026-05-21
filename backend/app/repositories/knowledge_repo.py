"""``knowledge_base`` / ``knowledge_document`` 表读写。"""

from __future__ import annotations

import uuid
from datetime import datetime
from math import ceil
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.constants.knowledge import DOCUMENT_STATUS_INDEXED
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.knowledge_mod import KnowledgeBase, KnowledgeDocument, KnowledgeTask
from app.models.workspace_mod import WorkspaceNamespace
from app.repositories.base_repo import BaseRepository, Page


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    model = KnowledgeBase

    @classmethod
    async def list_page(
        cls,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        status: str | None = None,
        workspace_namespace: str | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Page[KnowledgeBase]:
        normalized_page = max(1, page)
        normalized_page_size = max(1, min(page_size, 100))

        async def _read(session: AsyncSession) -> tuple[list[KnowledgeBase], int]:
            filters: list[Any] = [KnowledgeBase.deleted_at.is_(None)]
            st = (status or "").strip()
            if st:
                filters.append(KnowledgeBase.status == st)
            if workspace_namespace is not None and str(workspace_namespace).strip():
                ns = str(workspace_namespace).strip()
                filters.append(
                    KnowledgeBase.namespace_id.in_(
                        select(WorkspaceNamespace.id).where(WorkspaceNamespace.slug == ns)
                    )
                )
            if q and q.strip():
                like = f"%{q.strip()}%"
                filters.append(
                    or_(
                        KnowledgeBase.name.like(like),
                        KnowledgeBase.slug.like(like),
                    )
                )
            base = select(KnowledgeBase).options(selectinload(KnowledgeBase.namespace))
            count_q = select(func.count()).select_from(KnowledgeBase)
            for f in filters:
                base = base.where(f)
                count_q = count_q.where(f)
            total = int(await session.scalar(count_q) or 0)
            offset = (normalized_page - 1) * normalized_page_size
            base = base.order_by(KnowledgeBase.id.desc()).offset(offset).limit(normalized_page_size)
            items = list((await session.scalars(base)).all())
            return items, total

        items, total = await cls.run_read(_read, db_manager)
        total_pages = ceil(total / normalized_page_size) if total > 0 else 0
        return Page(
            items=items,
            page=normalized_page,
            page_size=normalized_page_size,
            total=total,
            total_pages=total_pages,
        )

    @classmethod
    async def get_active_by_id(
        cls,
        kb_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> KnowledgeBase | None:
        async def _read(session: AsyncSession) -> KnowledgeBase | None:
            stmt = (
                select(KnowledgeBase)
                .options(selectinload(KnowledgeBase.namespace))
                .where(KnowledgeBase.id == kb_id)
            )
            row = (await session.scalars(stmt)).first()
            if row is None or row.deleted_at is not None:
                return None
            return row

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def patch_by_id(
        cls,
        kb_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **fields: Any,
    ) -> KnowledgeBase | None:
        if not fields:
            return await cls.get_active_by_id(kb_id, db_manager=db_manager)

        async def _write(session: AsyncSession) -> KnowledgeBase | None:
            row = await session.get(KnowledgeBase, kb_id)
            if row is None or row.deleted_at is not None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.flush()
            stmt = (
                select(KnowledgeBase)
                .options(selectinload(KnowledgeBase.namespace))
                .where(KnowledgeBase.id == kb_id)
            )
            out = (await session.scalars(stmt)).first()
            return out if out is None or out.deleted_at is None else None

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def soft_delete_kb(
        cls,
        kb_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """软删知识库及其未删文档行；释放 ``slug`` 唯一约束后供占位清理任务使用。"""

        async def _write(session: AsyncSession) -> bool:
            row = await session.get(KnowledgeBase, kb_id)
            if row is None or row.deleted_at is not None:
                return False
            now = datetime.now()
            new_slug = f"d{kb_id}-{row.slug}"[:128]
            row.deleted_at = now
            row.slug = new_slug
            await session.execute(
                update(KnowledgeDocument)
                .where(
                    KnowledgeDocument.kb_id == kb_id,
                    KnowledgeDocument.deleted_at.is_(None),
                )
                .values(deleted_at=now)
            )
            await session.flush()
            await session.refresh(row)
            return True

        return await cls.run_write(_write, db_manager)


class KnowledgeDocumentRepository(BaseRepository[KnowledgeDocument]):
    model = KnowledgeDocument

    @classmethod
    async def list_page_by_kb(
        cls,
        kb_id: int,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        chunk_left_panel: bool = False,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Page[KnowledgeDocument]:
        normalized_page = max(1, page)
        normalized_page_size = max(1, min(page_size, 100))

        async def _read(session: AsyncSession) -> tuple[list[KnowledgeDocument], int]:
            filters: list[Any] = [
                KnowledgeDocument.kb_id == kb_id,
                KnowledgeDocument.deleted_at.is_(None),
            ]
            if chunk_left_panel:
                # 左侧「待配置」：已 indexed 且已设文档级切块覆盖的不在此列；恢复默认后 chunk_method 为空则回到左侧
                filters.append(
                    or_(
                        KnowledgeDocument.status != "indexed",
                        KnowledgeDocument.chunk_method.is_(None),
                        KnowledgeDocument.chunk_method == "",
                    ),
                )
            if q and q.strip():
                like = f"%{q.strip()}%"
                filters.append(KnowledgeDocument.filename.like(like))
            base = select(KnowledgeDocument)
            count_q = select(func.count()).select_from(KnowledgeDocument)
            for f in filters:
                base = base.where(f)
                count_q = count_q.where(f)
            total = int(await session.scalar(count_q) or 0)
            offset = (normalized_page - 1) * normalized_page_size
            base = (
                base.order_by(KnowledgeDocument.id.desc())
                .offset(offset)
                .limit(normalized_page_size)
            )
            items = list((await session.scalars(base)).all())
            return items, total

        items, total = await cls.run_read(_read, db_manager)
        total_pages = ceil(total / normalized_page_size) if total > 0 else 0
        return Page(
            items=items,
            page=normalized_page,
            page_size=normalized_page_size,
            total=total,
            total_pages=total_pages,
        )

    @classmethod
    async def list_indexed_with_chunks_by_kb(
        cls,
        kb_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[KnowledgeDocument]:
        """已 indexed 且 ``chunk_count > 0`` 的文档（从 Mongo 分片重建向量、不重跑 ingest）。"""

        async def _read(session: AsyncSession) -> list[KnowledgeDocument]:
            q = (
                select(KnowledgeDocument)
                .where(
                    KnowledgeDocument.kb_id == kb_id,
                    KnowledgeDocument.deleted_at.is_(None),
                    KnowledgeDocument.status == DOCUMENT_STATUS_INDEXED,
                    KnowledgeDocument.chunk_count > 0,
                )
                .order_by(KnowledgeDocument.id.asc())
            )
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def soft_delete_by_id(
        cls,
        doc_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        async def _write(session: AsyncSession) -> bool:
            row = await session.get(KnowledgeDocument, doc_id)
            if row is None or row.deleted_at is not None:
                return False
            row.deleted_at = datetime.now()
            await session.flush()
            return True

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def get_by_id(
        cls,
        doc_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> KnowledgeDocument | None:
        async def _read(session: AsyncSession) -> KnowledgeDocument | None:
            row = await session.get(KnowledgeDocument, doc_id)
            if row is None or row.deleted_at is not None:
                return None
            return row

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def update_fields_by_id(
        cls,
        doc_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **fields: Any,
    ) -> KnowledgeDocument | None:
        if not fields:
            return None

        async def _write(session: AsyncSession) -> KnowledgeDocument | None:
            row = await session.get(KnowledgeDocument, doc_id)
            if row is None or row.deleted_at is not None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.flush()
            await session.refresh(row)
            return row

        return await cls.run_write(_write, db_manager)


class KnowledgeTaskRepository(BaseRepository[KnowledgeTask]):
    model = KnowledgeTask

    @classmethod
    async def get_by_task_id(
        cls,
        task_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> KnowledgeTask | None:
        async def _read(session: AsyncSession) -> KnowledgeTask | None:
            q = select(KnowledgeTask).where(KnowledgeTask.task_id == task_id).limit(1)
            return (await session.execute(q)).scalar_one_or_none()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def update_fields_by_task_id(
        cls,
        task_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **fields: Any,
    ) -> KnowledgeTask | None:
        if not fields:
            return await cls.get_by_task_id(task_id, db_manager=db_manager)

        async def _write(session: AsyncSession) -> KnowledgeTask | None:
            q = select(KnowledgeTask).where(KnowledgeTask.task_id == task_id).limit(1)
            row = (await session.execute(q)).scalar_one_or_none()
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.flush()
            await session.refresh(row)
            return row

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def create_ingest_task(
        cls,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
        payload: dict[str, Any] | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> KnowledgeTask | None:
        from app.core.constants.knowledge import (
            TASK_STATUS_QUEUED,
            TASK_TYPE_INGEST,
        )

        task_id = uuid.uuid4().hex
        idempotency_key = f"ingest:{doc_id}:{content_version}"
        try:
            return await cls.create(
                db_manager=db_manager,
                task_id=task_id,
                kb_id=kb_id,
                doc_id=doc_id,
                content_version=content_version,
                task_type=TASK_TYPE_INGEST,
                status=TASK_STATUS_QUEUED,
                idempotency_key=idempotency_key,
                payload=payload,
            )
        except IntegrityError:
            return None

    @classmethod
    async def delete_by_idempotency_key(
        cls,
        idempotency_key: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> int:
        """删除幂等键对应的任务行（用于重新入队 ingest）。"""

        async def _write(session: AsyncSession) -> int:
            result = await session.execute(
                delete(KnowledgeTask).where(KnowledgeTask.idempotency_key == idempotency_key),
            )
            await session.flush()
            return int(result.rowcount or 0)

        return await cls.run_write(_write, db_manager)
