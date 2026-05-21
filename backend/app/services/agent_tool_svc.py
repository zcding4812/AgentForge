"""Agent 工具注册表查询（进程内 LangChain 工具）。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool

from app.agent.adapters.tools.builtins import BUILTIN_TOOL_NAMES
from app.agent.adapters.tools.registry import tool_registry
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.ext_tool_mod import (
    RuntimeExternalTool,
    RuntimeHttpTool,
    RuntimeMcpTool,
)
from app.repositories.ext_tool_repo import RuntimeExternalToolRepository
from app.schemas.agent import (
    AgentHttpToolConfigOut,
    AgentRegisteredToolOut,
    AgentRegisteredToolsData,
)


class AgentToolService:
    """进程内工具注册表与库表外部工具并集查询（供 API 编排调用）。"""

    def __init__(self, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._db = db_manager

    @staticmethod
    def _parameters_summary_from_json_schema(schema: dict[str, Any] | None) -> str:
        if not schema:
            return "（无参数）"
        props = schema.get("properties")
        if not isinstance(props, dict) or not props:
            return "（无参数）"
        parts: list[str] = []
        for key, spec in props.items():
            if not isinstance(spec, dict):
                parts.append(f"{key}: any")
                continue
            typ = spec.get("type")
            if isinstance(typ, list):
                typ_s = "|".join(str(t) for t in typ)
            elif typ is None:
                typ_s = "any"
            else:
                typ_s = str(typ)
            parts.append(f"{key}: {typ_s}")
        return ", ".join(parts)

    @staticmethod
    def _json_schema_for_tool(tool: BaseTool) -> dict[str, Any] | None:
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is None:
            return None
        if hasattr(schema_cls, "model_json_schema"):
            return schema_cls.model_json_schema()  # type: ignore[no-any-return]
        return None

    @staticmethod
    def _headers_for_out(raw: dict[str, Any] | None) -> dict[str, str] | None:
        if not raw or not isinstance(raw, dict):
            return None
        return {str(k): v if isinstance(v, str) else str(v) for k, v in raw.items()}

    @staticmethod
    def _tool_row_to_http_mcp(
        row: RuntimeExternalTool,
    ) -> tuple[
        Literal["http", "mcp"],
        AgentHttpToolConfigOut | None,
        dict[str, Any] | None,
    ]:
        """由持久化行得到 origin 与展示用 http/mcp 摘要。"""
        if isinstance(row, RuntimeMcpTool):
            mcp_cfg: dict[str, Any] = {
                "server_url": row.server_url,
                "transport_type": row.transport_type,
                **(row.connection_config_json or {}),
            }
            if row.tools_config_json:
                mcp_cfg["tools_config"] = row.tools_config_json
            return "mcp", None, mcp_cfg
        if isinstance(row, RuntimeHttpTool):
            http_req = AgentHttpToolConfigOut(
                method=row.method or "GET",
                url=row.url or "",
                headers=AgentToolService._headers_for_out(row.headers_json),
                body=row.request_body_template,
                timeout_seconds=row.timeout_ms / 1000.0,
            )
            return "http", http_req, None
        raise RuntimeError(f"unexpected persisted external tool type: {type(row)!r}")

    async def list_registered(
        self,
        *,
        namespace: str = "default",
    ) -> AgentRegisteredToolsData:
        """列出工具：名称为 **注册表键 ∪ 库表名** 的并集。

        合并库表后，以 ``runtime_external_tool.name`` 展示 HTTP/MCP（含仅库表存在、
        尚未在进程内注册的工具，如 ``enabled=false`` 未重放者）。
        """
        ext_rows = await RuntimeExternalToolRepository.list_all_order_by_name(
            db_manager=self._db,
        )
        ext_by_name: dict[str, RuntimeExternalTool] = {r.name: r for r in ext_rows}

        reg_by_name: dict[str, BaseTool] = dict(tool_registry.list_tool_entries(namespace))
        all_names = sorted(set(reg_by_name.keys()) | set(ext_by_name.keys()))

        items: list[AgentRegisteredToolOut] = []
        for name in all_names:
            t = reg_by_name.get(name)
            row = ext_by_name.get(name)
            js = self._json_schema_for_tool(t) if t is not None else None

            if name in BUILTIN_TOOL_NAMES:
                origin: Literal["builtin", "runtime", "http", "mcp"] = "builtin"
                http_req = None
                mcp_cfg = None
            elif row is not None:
                origin, http_req, mcp_cfg = self._tool_row_to_http_mcp(row)
            else:
                origin = "runtime"
                http_req = None
                mcp_cfg = None

            desc = ""
            if name in BUILTIN_TOOL_NAMES and t is not None:
                d = getattr(t, "description", None)
                desc = d if isinstance(d, str) else ""
            elif row is not None:
                desc = row.description or ""
            elif t is not None:
                d = getattr(t, "description", None)
                desc = d if isinstance(d, str) else ""

            js_from_db = row.input_schema if row is not None and row.input_schema else None
            js_effective = js_from_db or js
            items.append(
                AgentRegisteredToolOut(
                    name=name,
                    namespace=namespace,
                    description=desc,
                    parameters_summary=self._parameters_summary_from_json_schema(js_effective),
                    parameters_json_schema=js_effective,
                    origin=origin,
                    http_request=http_req,
                    mcp_config=mcp_cfg,
                )
            )
        return AgentRegisteredToolsData(items=items, namespace=namespace)
