"""``namespace`` 表读写。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.workspace_mod import WorkspaceNamespace
from app.repositories.base_repo import BaseRepository


class WorkspaceNamespaceRepository(BaseRepository[WorkspaceNamespace]):
    model = WorkspaceNamespace

    @classmethod
    async def get_by_slug(
        cls,
        slug: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> WorkspaceNamespace | None:
        s = (slug or "").strip()
        if not s:
            return None

        async def _read(session: AsyncSession) -> WorkspaceNamespace | None:
            q = select(WorkspaceNamespace).where(WorkspaceNamespace.slug == s).limit(1)
            return (await session.scalars(q)).first()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def get_by_id(
        cls,
        namespace_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> WorkspaceNamespace | None:
        async def _read(session: AsyncSession) -> WorkspaceNamespace | None:
            return await session.get(WorkspaceNamespace, namespace_id)

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def patch_by_id(
        cls,
        namespace_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **fields: Any,
    ) -> WorkspaceNamespace | None:
        if not fields:
            return await cls.get_by_id(namespace_id, db_manager=db_manager)

        async def _write(session: AsyncSession) -> WorkspaceNamespace | None:
            row = await session.get(WorkspaceNamespace, namespace_id)
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.flush()
            await session.refresh(row)
            return row

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def list_all_by_slug_order(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[WorkspaceNamespace]:
        async def _read(session: AsyncSession) -> list[WorkspaceNamespace]:
            q = select(WorkspaceNamespace).order_by(WorkspaceNamespace.slug.asc())
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def get_or_create_by_slug(
        cls,
        slug: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> WorkspaceNamespace:
        """按 slug 查找；不存在则插入（并发下遇唯一冲突则重查）。"""
        s = (slug or "").strip() or "default"
        existing = await cls.get_by_slug(s, db_manager=db_manager)
        if existing is not None:
            return existing
        try:
            return await cls.create(db_manager=db_manager, slug=s)
        except IntegrityError:
            got = await cls.get_by_slug(s, db_manager=db_manager)
            if got is not None:
                return got
            raise

    @classmethod
    async def set_workbench_agent_id(
        cls,
        namespace_id: int,
        workbench_agent_id: int | None,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> WorkspaceNamespace | None:
        return await cls.patch_by_id(
            namespace_id,
            db_manager=db_manager,
            workbench_agent_id=workbench_agent_id,
        )
