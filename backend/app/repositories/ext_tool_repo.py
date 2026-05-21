"""``runtime_external_tool`` 及 HTTP/MCP 子表访问（类表继承）。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectin_polymorphic

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.ext_tool_mod import (
    RuntimeExternalTool,
    RuntimeHttpTool,
    RuntimeMcpTool,
)
from app.repositories.base_repo import BaseRepository


class RuntimeExternalToolRepository(BaseRepository[RuntimeExternalTool]):
    model = RuntimeExternalTool

    @classmethod
    async def list_all_order_by_name(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[RuntimeExternalTool]:
        async def _read(session: AsyncSession) -> list[RuntimeExternalTool]:
            q = (
                select(RuntimeExternalTool)
                .options(
                    selectin_polymorphic(RuntimeExternalTool, [RuntimeHttpTool, RuntimeMcpTool]),
                )
                .order_by(RuntimeExternalTool.name.asc())
            )
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_enabled_order_by_name(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[RuntimeExternalTool]:
        async def _read(session: AsyncSession) -> list[RuntimeExternalTool]:
            q = (
                select(RuntimeExternalTool)
                .options(
                    selectin_polymorphic(RuntimeExternalTool, [RuntimeHttpTool, RuntimeMcpTool]),
                )
                .where(RuntimeExternalTool.enabled.is_(True))
                .order_by(RuntimeExternalTool.name.asc())
            )
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def get_by_name(
        cls,
        name: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RuntimeExternalTool | None:
        async def _read(session: AsyncSession) -> RuntimeExternalTool | None:
            q = (
                select(RuntimeExternalTool)
                .options(
                    selectin_polymorphic(RuntimeExternalTool, [RuntimeHttpTool, RuntimeMcpTool]),
                )
                .where(RuntimeExternalTool.name == name)
            )
            return (await session.scalars(q)).first()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def create_http_tool(
        cls,
        *,
        name: str,
        description: str,
        url: str,
        method: str,
        headers_json: dict[str, Any] | None,
        request_body_template: str | None,
        timeout_ms: int,
        enabled: bool = True,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RuntimeHttpTool:
        async def _write(session: AsyncSession) -> RuntimeHttpTool:
            row = RuntimeHttpTool(
                name=name,
                description=description,
                enabled=enabled,
                input_schema=input_schema,
                output_schema=output_schema,
                url=url,
                method=method,
                headers_json=headers_json,
                request_body_template=request_body_template,
                timeout_ms=timeout_ms,
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return row

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def create_mcp_tool(
        cls,
        *,
        name: str,
        description: str,
        transport_type: str,
        server_url: str,
        connection_config_json: dict[str, Any] | None,
        tools_config_json: dict[str, Any] | None,
        enabled: bool = True,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RuntimeMcpTool:
        async def _write(session: AsyncSession) -> RuntimeMcpTool:
            row = RuntimeMcpTool(
                name=name,
                description=description,
                enabled=enabled,
                input_schema=input_schema,
                output_schema=output_schema,
                transport_type=transport_type,
                server_url=server_url,
                connection_config_json=connection_config_json,
                tools_config_json=tools_config_json,
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return row

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def update_http_by_name(
        cls,
        name: str,
        *,
        expected_version: int | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        url: str | None = None,
        method: str | None = None,
        headers_json: dict[str, Any] | None = None,
        request_body_template: str | None = None,
        timeout_ms: int | None = None,
        unset_headers: bool = False,
        unset_body: bool = False,
        unset_input_schema: bool = False,
        unset_output_schema: bool = False,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RuntimeHttpTool | None:
        async def _write(session: AsyncSession) -> RuntimeHttpTool | None:
            q = (
                select(RuntimeExternalTool)
                .options(
                    selectin_polymorphic(RuntimeExternalTool, [RuntimeHttpTool, RuntimeMcpTool]),
                )
                .where(RuntimeExternalTool.name == name)
            )
            obj = (await session.scalars(q)).first()
            if obj is None or not isinstance(obj, RuntimeHttpTool):
                return None
            if expected_version is not None and obj.version != expected_version:
                raise ValueError(
                    f"工具版本冲突：期望 version={expected_version}，当前为 {obj.version}"
                )
            if description is not None:
                obj.description = description
            if enabled is not None:
                obj.enabled = enabled
            if unset_input_schema:
                obj.input_schema = None
            elif input_schema is not None:
                obj.input_schema = input_schema
            if unset_output_schema:
                obj.output_schema = None
            elif output_schema is not None:
                obj.output_schema = output_schema
            if url is not None:
                obj.url = url
            if method is not None:
                obj.method = method
            if unset_headers:
                obj.headers_json = None
            elif headers_json is not None:
                obj.headers_json = headers_json
            if unset_body:
                obj.request_body_template = None
            elif request_body_template is not None:
                obj.request_body_template = request_body_template
            if timeout_ms is not None:
                obj.timeout_ms = timeout_ms
            await session.flush()
            await session.refresh(obj)
            return obj

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def update_mcp_by_name(
        cls,
        name: str,
        *,
        expected_version: int | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        transport_type: str | None = None,
        server_url: str | None = None,
        connection_config_json: dict[str, Any] | None = None,
        tools_config_json: dict[str, Any] | None = None,
        unset_input_schema: bool = False,
        unset_output_schema: bool = False,
        unset_connection_config: bool = False,
        unset_tools_config: bool = False,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RuntimeMcpTool | None:
        async def _write(session: AsyncSession) -> RuntimeMcpTool | None:
            q = (
                select(RuntimeExternalTool)
                .options(
                    selectin_polymorphic(RuntimeExternalTool, [RuntimeHttpTool, RuntimeMcpTool]),
                )
                .where(RuntimeExternalTool.name == name)
            )
            obj = (await session.scalars(q)).first()
            if obj is None or not isinstance(obj, RuntimeMcpTool):
                return None
            if expected_version is not None and obj.version != expected_version:
                raise ValueError(
                    f"工具版本冲突：期望 version={expected_version}，当前为 {obj.version}"
                )
            if description is not None:
                obj.description = description
            if enabled is not None:
                obj.enabled = enabled
            if unset_input_schema:
                obj.input_schema = None
            elif input_schema is not None:
                obj.input_schema = input_schema
            if unset_output_schema:
                obj.output_schema = None
            elif output_schema is not None:
                obj.output_schema = output_schema
            if transport_type is not None:
                obj.transport_type = transport_type
            if server_url is not None:
                obj.server_url = server_url
            if unset_connection_config:
                obj.connection_config_json = None
            elif connection_config_json is not None:
                obj.connection_config_json = connection_config_json
            if unset_tools_config:
                obj.tools_config_json = None
            elif tools_config_json is not None:
                obj.tools_config_json = tools_config_json
            await session.flush()
            await session.refresh(obj)
            return obj

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def delete_by_name(
        cls,
        name: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> int:
        async def _write(session: AsyncSession) -> int:
            q = select(RuntimeExternalTool).where(RuntimeExternalTool.name == name)
            obj = (await session.scalars(q)).first()
            if obj is None:
                return 0
            await session.delete(obj)
            await session.flush()
            return 1

        return await cls.run_write(_write, db_manager)
