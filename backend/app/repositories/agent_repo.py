"""``agent_entity`` 表读写（ORM：`AgentEntity`）。"""

from __future__ import annotations

from math import ceil
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.agent_mod import AgentEntity
from app.models.workspace_mod import WorkspaceNamespace
from app.repositories.base_repo import BaseRepository, Page


class AgentRepository(BaseRepository[AgentEntity]):
    model = AgentEntity

    @classmethod
    async def list_page(
        cls,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        agent_kinds: list[str] | None = None,
        exclude_agent_kinds: list[str] | None = None,
        workspace_namespace: str | None = None,
        namespace_id: int | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Page[AgentEntity]:
        normalized_page = max(1, page)
        normalized_page_size = max(1, min(page_size, 100))

        async def _read(session: AsyncSession) -> tuple[list[AgentEntity], int]:
            filters: list[Any] = []
            kinds = [k.strip() for k in (agent_kinds or []) if k and k.strip()]
            if kinds:
                filters.append(AgentEntity.agent_kind.in_(kinds))
            excluded = [k.strip() for k in (exclude_agent_kinds or []) if k and k.strip()]
            if excluded:
                filters.append(AgentEntity.agent_kind.notin_(excluded))
            if namespace_id is not None:
                filters.append(AgentEntity.namespace_id == namespace_id)
            elif workspace_namespace is not None and str(workspace_namespace).strip():
                ns = str(workspace_namespace).strip()
                filters.append(
                    AgentEntity.namespace_id.in_(
                        select(WorkspaceNamespace.id).where(WorkspaceNamespace.slug == ns)
                    )
                )
            if q and q.strip():
                like = f"%{q.strip()}%"
                filters.append(
                    or_(
                        AgentEntity.name.like(like),
                        AgentEntity.description.like(like),
                        AgentEntity.system_prompt.like(like),
                    )
                )
            base = select(AgentEntity).options(selectinload(AgentEntity.namespace))
            count_q = select(func.count()).select_from(AgentEntity)
            if filters:
                for f in filters:
                    base = base.where(f)
                    count_q = count_q.where(f)
            total = int(await session.scalar(count_q) or 0)
            offset = (normalized_page - 1) * normalized_page_size
            base = base.order_by(AgentEntity.id.desc()).offset(offset).limit(normalized_page_size)
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
    async def list_in_workspace_namespace(
        cls,
        *,
        workspace_namespace: str,
        limit: int = 500,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[AgentEntity]:
        """按命名空间 slug 列出 Agent（id 升序）；每次调用即查库，无应用层缓存。"""

        ns = str(workspace_namespace).strip()
        if not ns:
            return []

        cap = max(1, min(int(limit), 2000))

        async def _read(session: AsyncSession) -> list[AgentEntity]:
            stmt = (
                select(AgentEntity)
                .options(selectinload(AgentEntity.namespace))
                .where(
                    AgentEntity.namespace_id.in_(
                        select(WorkspaceNamespace.id).where(WorkspaceNamespace.slug == ns)
                    )
                )
                .order_by(AgentEntity.id.asc())
                .limit(cap)
            )
            return list((await session.scalars(stmt)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def get_by_id(
        cls,
        agent_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> AgentEntity | None:
        async def _read(session: AsyncSession) -> AgentEntity | None:
            stmt = (
                select(AgentEntity)
                .options(selectinload(AgentEntity.namespace))
                .where(AgentEntity.id == agent_id)
            )
            return (await session.scalars(stmt)).first()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def patch_by_id(
        cls,
        agent_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **fields: Any,
    ) -> AgentEntity | None:
        if not fields:
            return await cls.get_by_id(agent_id, db_manager=db_manager)

        async def _write(session: AsyncSession) -> AgentEntity | None:
            row = await session.get(AgentEntity, agent_id)
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.flush()
            stmt = (
                select(AgentEntity)
                .options(selectinload(AgentEntity.namespace))
                .where(AgentEntity.id == agent_id)
            )
            return (await session.scalars(stmt)).first()

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def delete_by_id(
        cls,
        agent_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """物理删除 ``agent_entity`` 行；关联 ``conversation_session`` 由 FK CASCADE 级联删除。"""

        async def _write(session: AsyncSession) -> bool:
            row = await session.get(AgentEntity, agent_id)
            if row is None:
                return False
            await session.delete(row)
            await session.flush()
            return True

        return await cls.run_write(_write, db_manager)
