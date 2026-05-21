"""`sys_model_provider` / `sys_model` 持久化（异步 ``AsyncSession``）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.sys_model_mod import SysModel, SysModelProvider
from app.repositories.base_repo import BaseRepository


class ProviderRepository(BaseRepository[SysModelProvider]):
    model = SysModelProvider

    @staticmethod
    async def _resolve_provider(session: AsyncSession, key: str) -> SysModelProvider | None:
        key = key.strip()
        if key.isdigit():
            return await session.get(SysModelProvider, int(key))
        return await session.scalar(
            select(SysModelProvider).where(SysModelProvider.provider_code == key)
        )

    @classmethod
    async def get_provider_in_session(
        cls, session: AsyncSession, key: str
    ) -> SysModelProvider | None:
        return await cls._resolve_provider(session, key)

    @classmethod
    async def get_provider_by_key(
        cls,
        provider_key: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> SysModelProvider | None:
        async def _read(session: AsyncSession) -> SysModelProvider | None:
            return await cls._resolve_provider(session, provider_key)

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def provider_counts_and_types(
        cls,
        provider_ids: list[int],
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> tuple[dict[int, int], dict[int, set[str]]]:
        if not provider_ids:
            return {}, {}

        async def _read(session: AsyncSession) -> tuple[dict[int, int], dict[int, set[str]]]:
            result = await session.execute(
                select(SysModel.provider_id, SysModel.model_type).where(
                    SysModel.provider_id.in_(provider_ids)
                )
            )
            rows = result.all()
            types_map: dict[int, set[str]] = {pid: set() for pid in provider_ids}
            count_map: dict[int, int] = {pid: 0 for pid in provider_ids}
            for pid, mt in rows:
                count_map[pid] = count_map.get(pid, 0) + 1
                types_map.setdefault(pid, set()).add(mt)
            return count_map, types_map

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_providers_paginated(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        q: str | None,
        status: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[SysModelProvider], int]:
        async def _read(session: AsyncSession) -> tuple[list[SysModelProvider], int]:
            filters: list[Any] = []
            if q and q.strip():
                like = f"%{q.strip()}%"
                filters.append(
                    or_(
                        SysModelProvider.provider_code.like(like),
                        SysModelProvider.provider_name.like(like),
                    )
                )
            if status == "enabled":
                filters.append(SysModelProvider.status == 1)
            elif status == "disabled":
                filters.append(SysModelProvider.status == 0)

            count_q = select(func.count()).select_from(SysModelProvider)
            if filters:
                count_q = count_q.where(*filters)
            total = int(await session.scalar(count_q) or 0)

            data_q = select(SysModelProvider).order_by(
                SysModelProvider.status.desc(),
                SysModelProvider.id.asc(),
            )
            if filters:
                data_q = data_q.where(*filters)
            offset = (page - 1) * page_size
            items = list((await session.scalars(data_q.offset(offset).limit(page_size))).all())
            return items, total

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def insert_provider(
        cls,
        row: SysModelProvider,
        *,
        db_manager: SQLAlchemyDatabaseManager,
    ) -> SysModelProvider:
        async def _write(session: AsyncSession) -> SysModelProvider:
            if await cls._resolve_provider(session, row.provider_code):
                raise ValueError("供应商编码已存在")
            session.add(row)
            await session.flush()
            return row

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def update_provider(
        cls,
        provider_key: str,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        apply_updates: Callable[[SysModelProvider], None],
        after_flush: Callable[[SysModelProvider], None] | None = None,
    ) -> SysModelProvider:
        async def _write(session: AsyncSession) -> SysModelProvider:
            p = await cls._resolve_provider(session, provider_key)
            if not p:
                raise KeyError("not found")
            apply_updates(p)
            await session.flush()
            if after_flush:
                after_flush(p)
            return p

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def delete_provider_by_key(
        cls,
        provider_key: str,
        *,
        db_manager: SQLAlchemyDatabaseManager,
    ) -> None:
        async def _write(session: AsyncSession) -> None:
            p = await cls._resolve_provider(session, provider_key)
            if not p:
                raise KeyError("not found")
            await session.delete(p)

        await cls.run_write(_write, db_manager)

    @classmethod
    async def list_models_paginated(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        q: str | None,
        provider_id: str | None,
        model_types: tuple[str, ...] | None,
        status: str | None,
        provider_enabled_only: bool,
        page: int,
        page_size: int,
    ) -> tuple[list[tuple[SysModel, SysModelProvider]], int]:
        async def _read(
            session: AsyncSession,
        ) -> tuple[list[tuple[SysModel, SysModelProvider]], int]:
            filters: list[Any] = []
            if provider_enabled_only:
                filters.append(SysModelProvider.status == 1)
            if q and q.strip():
                like = f"%{q.strip()}%"
                filters.append(
                    or_(
                        SysModel.model_code.like(like),
                        SysModel.model_name.like(like),
                    )
                )
            if provider_id and provider_id.strip():
                prov = await cls._resolve_provider(session, provider_id.strip())
                if prov:
                    filters.append(SysModel.provider_id == prov.id)
                else:
                    filters.append(SysModel.id == -1)
            if model_types:
                filters.append(SysModel.model_type.in_(model_types))
            if status == "enabled":
                filters.append(SysModel.is_enabled == 1)
            elif status == "disabled":
                filters.append(SysModel.is_enabled == 0)

            inner = select(SysModel.id).join(
                SysModelProvider, SysModel.provider_id == SysModelProvider.id
            )
            if filters:
                inner = inner.where(*filters)
            total = int(
                await session.scalar(select(func.count()).select_from(inner.subquery())) or 0
            )

            data_q = (
                select(SysModel, SysModelProvider)
                .join(SysModelProvider, SysModel.provider_id == SysModelProvider.id)
                .order_by(SysModel.id.asc())
            )
            if filters:
                data_q = data_q.where(*filters)
            offset = (page - 1) * page_size
            result = await session.execute(data_q.offset(offset).limit(page_size))
            rows = list(result.all())
            return rows, total

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def create_model_row(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        provider_id_key: str,
        model_code: str,
        model_name: str,
        model_type: str,
        endpoint: str | None,
        timeout: int,
        is_enabled: bool,
        validate_enabled_on_provider: Callable[[SysModelProvider], None] | None,
    ) -> tuple[SysModel, SysModelProvider]:
        async def _write(session: AsyncSession) -> tuple[SysModel, SysModelProvider]:
            prov = await cls._resolve_provider(session, provider_id_key.strip())
            if not prov:
                raise ValueError("供应商不存在")
            code = model_code.strip()
            dup = (
                await session.scalars(
                    select(SysModel).where(
                        SysModel.provider_id == prov.id,
                        SysModel.model_code == code,
                    )
                )
            ).first()
            if dup:
                raise ValueError("该供应商下模型编码已存在")
            if is_enabled and validate_enabled_on_provider:
                validate_enabled_on_provider(prov)
            row = SysModel(
                provider_id=prov.id,
                model_code=code,
                model_name=model_name.strip(),
                model_type=model_type.strip(),
                endpoint=endpoint,
                timeout=timeout,
                is_enabled=1 if is_enabled else 0,
            )
            session.add(row)
            await session.flush()
            return row, prov

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def update_model(
        cls,
        model_pk: int,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        apply_updates: Callable[[SysModel, SysModelProvider, AsyncSession], Awaitable[None]],
        after_flush: Callable[[SysModel, SysModelProvider], None] | None = None,
    ) -> tuple[SysModel, SysModelProvider]:
        async def _write(session: AsyncSession) -> tuple[SysModel, SysModelProvider]:
            m = await session.get(SysModel, model_pk)
            if not m:
                raise KeyError("not found")
            prov = await session.get(SysModelProvider, m.provider_id)
            if not prov:
                raise KeyError("not found")
            await apply_updates(m, prov, session)
            await session.flush()
            prov_final = await session.get(SysModelProvider, m.provider_id)
            if not prov_final:
                raise KeyError("not found")
            if after_flush:
                after_flush(m, prov_final)
            return m, prov_final

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def delete_model_by_pk(
        cls,
        model_pk: int,
        *,
        db_manager: SQLAlchemyDatabaseManager,
    ) -> None:
        async def _write(session: AsyncSession) -> None:
            m = await session.get(SysModel, model_pk)
            if not m:
                raise KeyError("not found")
            await session.delete(m)

        await cls.run_write(_write, db_manager)

    @classmethod
    async def get_model_with_provider(
        cls,
        model_pk: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> tuple[SysModel, SysModelProvider] | None:
        async def _read(session: AsyncSession) -> tuple[SysModel, SysModelProvider] | None:
            m = await session.get(SysModel, model_pk)
            if not m:
                return None
            p = await session.get(SysModelProvider, m.provider_id)
            if not p:
                return None
            return m, p

        return await cls.run_read(_read, db_manager)
