"""持久化外部工具（HTTP / MCP）：落库、进程内注册、启动重放。"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from sqlalchemy.exc import IntegrityError

from app.agent.adapters.tools.dynamic.http_fetch import make_http_request_tool
from app.agent.adapters.tools.dynamic.mcp_streamable_http import (
    MCP_PROTOCOL_VERSION_DEFAULT,
    McpSseTransport,
    McpStreamableHttpClient,
    _headers_from_connection,
    make_mcp_streamable_http_tool,
)
from app.agent.adapters.tools.dynamic.mcp_transport import (
    McpProtocolError,
    McpTransport,
    McpTransportError,
)
from app.agent.adapters.tools.registry import register_tool, unregister_tool
from app.agent.kernel.tool_runtime import assert_mcp_http_url_allowed_async
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.ext_tool_mod import (
    RuntimeExternalTool,
    RuntimeHttpTool,
    RuntimeMcpTool,
)
from app.repositories.ext_tool_repo import RuntimeExternalToolRepository
from app.schemas.agent import (
    AgentExternalToolPersistedOut,
    AgentRegisterHttpToolBody,
    AgentRegisterMcpToolBody,
    AgentRuntimeExternalToolsListData,
    AgentToolRegisterResultData,
    AgentUpdateHttpToolBody,
    AgentUpdateMcpToolBody,
    McpProbeListToolsBody,
    McpProbeListToolsData,
)

logger = logging.getLogger(__name__)


def _timeout_ms_from_seconds(seconds: float) -> int:
    ms = int(round(float(seconds) * 1000))
    return max(1000, min(120_000_000, ms))


def _mcp_merged_config(row: RuntimeMcpTool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "server_url": row.server_url,
        "transport_type": row.transport_type,
    }
    if isinstance(row.connection_config_json, dict):
        for k, v in row.connection_config_json.items():
            if k not in cfg:
                cfg[k] = v
    if isinstance(row.tools_config_json, dict) and row.tools_config_json:
        cfg["tools_config"] = row.tools_config_json
    return cfg


def _register_in_memory(row: RuntimeExternalTool) -> None:
    if isinstance(row, RuntimeMcpTool):
        merged = _mcp_merged_config(row)
        t = make_mcp_streamable_http_tool(
            logical_name=row.name,
            description=row.description,
            mcp_config=merged,
        )
        register_tool(row.name, t, overwrite=True)
        return
    if isinstance(row, RuntimeHttpTool):
        t = make_http_request_tool(
            logical_name=row.name,
            description=row.description,
            url=row.url,
            method=row.method,
            headers=row.headers_json,
            body=row.request_body_template,
            timeout_s=row.timeout_ms / 1000.0,
        )
        register_tool(row.name, t, overwrite=True)
        return
    raise TypeError(f"未知外部工具类型: {type(row)!r}")


def _row_to_persisted_out(row: RuntimeExternalTool) -> AgentExternalToolPersistedOut:
    base = {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "kind": row.kind.value,
        "enabled": row.enabled,
        "version": row.version,
        "input_schema": row.input_schema,
        "output_schema": row.output_schema,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    if isinstance(row, RuntimeHttpTool):
        hdrs: dict[str, str] | None = None
        if isinstance(row.headers_json, dict) and row.headers_json:
            hdrs = {
                str(k): v if isinstance(v, str) else str(v) for k, v in row.headers_json.items()
            }
        return AgentExternalToolPersistedOut(
            **base,
            url=row.url,
            method=row.method,
            headers=hdrs,
            body=row.request_body_template,
            timeout_ms=row.timeout_ms,
            timeout_seconds=row.timeout_ms / 1000.0,
            transport_type=None,
            server_url=None,
            connection_config=None,
            tools_config=None,
            mcp_config=None,
        )
    if isinstance(row, RuntimeMcpTool):
        merged = _mcp_merged_config(row)
        return AgentExternalToolPersistedOut(
            **base,
            url=None,
            method=None,
            headers=None,
            body=None,
            timeout_ms=None,
            timeout_seconds=None,
            transport_type=row.transport_type,
            server_url=row.server_url,
            connection_config=row.connection_config_json,
            tools_config=row.tools_config_json,
            mcp_config=merged,
        )
    raise TypeError(f"未知外部工具类型: {type(row)!r}")


async def replay_persisted_external_tools(db_manager: SQLAlchemyDatabaseManager) -> None:
    """进程启动时从库中加载并注册到 LangChain 注册表（仅 enabled=true）。"""
    try:
        rows = await RuntimeExternalToolRepository.list_enabled_order_by_name(db_manager=db_manager)
    except Exception:
        logger.exception("ext_tool.replay_failed")
        raise
    for row in rows:
        try:
            _register_in_memory(row)
        except Exception:
            logger.exception(
                "ext_tool.replay_row_failed",
                extra={"name": row.name, "kind": getattr(row.kind, "value", row.kind)},
            )


async def create_persisted_http_tool(
    body: AgentRegisterHttpToolBody,
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> AgentToolRegisterResultData:
    timeout_ms = _timeout_ms_from_seconds(body.timeout_seconds)
    try:
        row = await RuntimeExternalToolRepository.create_http_tool(
            db_manager=db_manager,
            name=body.name,
            description=body.description,
            url=body.url,
            method=body.method,
            headers_json=body.headers,
            request_body_template=body.body,
            timeout_ms=timeout_ms,
            enabled=body.enabled,
            input_schema=body.input_schema,
            output_schema=body.output_schema,
        )
    except IntegrityError as e:
        raise ValueError(f"工具名已存在: {body.name!r}") from e
    _register_in_memory(row)
    return AgentToolRegisterResultData(name=body.name)


async def create_persisted_mcp_tool(
    body: AgentRegisterMcpToolBody,
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> AgentToolRegisterResultData:
    try:
        row = await RuntimeExternalToolRepository.create_mcp_tool(
            db_manager=db_manager,
            name=body.name,
            description=body.description,
            transport_type=body.transport_type,
            server_url=body.server_url,
            connection_config_json=body.connection_config,
            tools_config_json=body.tools_config,
            enabled=body.enabled,
            input_schema=body.input_schema,
            output_schema=body.output_schema,
        )
    except IntegrityError as e:
        raise ValueError(f"工具名已存在: {body.name!r}") from e
    _register_in_memory(row)
    return AgentToolRegisterResultData(name=body.name)


async def update_persisted_http_tool(
    name: str,
    body: AgentUpdateHttpToolBody,
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> AgentExternalToolPersistedOut:
    d = body.model_dump(exclude_unset=True)
    clear_headers = bool(d.pop("clear_headers", False))
    clear_body = bool(d.pop("clear_body", False))
    expected_version = d.pop("version", None)
    kwargs: dict[str, Any] = {}
    if expected_version is not None:
        kwargs["expected_version"] = int(expected_version)
    if "description" in d:
        kwargs["description"] = d["description"]
    if "enabled" in d:
        kwargs["enabled"] = d["enabled"]
    if "input_schema" in d:
        kwargs["input_schema"] = d["input_schema"]
    if "output_schema" in d:
        kwargs["output_schema"] = d["output_schema"]
    if "url" in d:
        kwargs["url"] = d["url"]
    if "method" in d:
        kwargs["method"] = d["method"]
    if "timeout_seconds" in d:
        kwargs["timeout_ms"] = _timeout_ms_from_seconds(d["timeout_seconds"])
    if clear_headers:
        kwargs["unset_headers"] = True
    elif "headers" in d:
        kwargs["headers_json"] = d["headers"]
    if clear_body:
        kwargs["unset_body"] = True
    elif "body" in d:
        kwargs["request_body_template"] = d["body"]
    if not kwargs:
        row = await RuntimeExternalToolRepository.get_by_name(name, db_manager=db_manager)
        if row is None:
            raise LookupError(f"未找到工具: {name!r}")
        if not isinstance(row, RuntimeHttpTool):
            raise LookupError(f"未找到 HTTP 工具: {name!r}")
        return _row_to_persisted_out(row)

    row = await RuntimeExternalToolRepository.update_http_by_name(
        name, db_manager=db_manager, **kwargs
    )
    if row is None:
        raise LookupError(f"未找到工具: {name!r}")
    _register_in_memory(row)
    return _row_to_persisted_out(row)


async def update_persisted_mcp_tool(
    name: str,
    body: AgentUpdateMcpToolBody,
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> AgentExternalToolPersistedOut:
    d = body.model_dump(exclude_unset=True)
    clear_cc = bool(d.pop("clear_connection_config", False))
    clear_tc = bool(d.pop("clear_tools_config", False))
    expected_version = d.pop("version", None)
    kwargs: dict[str, Any] = {}
    if expected_version is not None:
        kwargs["expected_version"] = int(expected_version)
    if "description" in d:
        kwargs["description"] = d["description"]
    if "enabled" in d:
        kwargs["enabled"] = d["enabled"]
    if "input_schema" in d:
        kwargs["input_schema"] = d["input_schema"]
    if "output_schema" in d:
        kwargs["output_schema"] = d["output_schema"]
    if "transport_type" in d:
        kwargs["transport_type"] = d["transport_type"]
    if "server_url" in d:
        kwargs["server_url"] = d["server_url"]
    if clear_cc:
        kwargs["unset_connection_config"] = True
    elif "connection_config" in d:
        kwargs["connection_config_json"] = d["connection_config"]
    if clear_tc:
        kwargs["unset_tools_config"] = True
    elif "tools_config" in d:
        kwargs["tools_config_json"] = d["tools_config"]
    if not kwargs:
        row = await RuntimeExternalToolRepository.get_by_name(name, db_manager=db_manager)
        if row is None:
            raise LookupError(f"未找到工具: {name!r}")
        if not isinstance(row, RuntimeMcpTool):
            raise LookupError(f"未找到 MCP 工具: {name!r}")
        return _row_to_persisted_out(row)

    row = await RuntimeExternalToolRepository.update_mcp_by_name(
        name, db_manager=db_manager, **kwargs
    )
    if row is None:
        raise LookupError(f"未找到工具: {name!r}")
    _register_in_memory(row)
    return _row_to_persisted_out(row)


async def delete_persisted_http_tool(
    name: str,
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> None:
    n = await RuntimeExternalToolRepository.delete_by_name(name, db_manager=db_manager)
    if n == 0:
        raise LookupError(f"未找到工具: {name!r}")
    unregister_tool(name)


async def list_persisted_external_tools(
    *,
    db_manager: SQLAlchemyDatabaseManager,
) -> AgentRuntimeExternalToolsListData:
    rows = await RuntimeExternalToolRepository.list_all_order_by_name(db_manager=db_manager)
    items = [_row_to_persisted_out(r) for r in rows]
    return AgentRuntimeExternalToolsListData(items=items)


async def probe_mcp_list_tools(body: McpProbeListToolsBody) -> McpProbeListToolsData:
    """不落库：对 MCP 端点执行 ``initialize`` + ``tools/list``，供控制台探测。"""
    await assert_mcp_http_url_allowed_async(body.server_url.strip())
    if body.transport_type not in ("http", "sse"):
        raise ValueError("探测仅支持 transport_type=http（Streamable HTTP）或 sse（HTTP+SSE）")

    mc: dict[str, Any] = {
        "server_url": body.server_url.strip(),
        **(body.connection_config or {}),
    }
    protocol_version = str(mc.get("protocol_version") or MCP_PROTOCOL_VERSION_DEFAULT).strip()
    verify_ssl = bool(mc.get("verify_ssl", True))
    timeout_s = float(mc.get("timeout_seconds") or 60.0)
    timeout_s = max(1.0, min(300.0, timeout_s))
    extra_headers = _headers_from_connection(mc)
    htt = httpx.Timeout(timeout_s, connect=min(30.0, timeout_s))
    client: McpTransport
    if body.transport_type == "sse":
        client = McpSseTransport(
            body.server_url.strip(),
            protocol_version=protocol_version,
            extra_headers=extra_headers,
            verify_ssl=verify_ssl,
            timeout=htt,
        )
    else:
        client = McpStreamableHttpClient(
            body.server_url.strip(),
            protocol_version=protocol_version,
            extra_headers=extra_headers,
            verify_ssl=verify_ssl,
            timeout=htt,
        )
    try:
        tools = await client.list_tools(
            platform_session_id=None,
            logical_tool_name="probe-list",
        )
        return McpProbeListToolsData(tools=tools)
    except (McpProtocolError, McpTransportError) as e:
        raise ValueError(f"MCP 探测失败: {e}") from e
    finally:
        await client.aclose()
