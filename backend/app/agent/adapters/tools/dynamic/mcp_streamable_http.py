"""MCP Streamable HTTP 客户端 + LangChain 工具。

传输层使用 LangChain 官方推荐的 ``langchain-mcp-adapters``（``create_session`` +
``transport: streamable_http``）与 Python ``mcp`` SDK 的 :class:`mcp.ClientSession`，
不再自研 JSON-RPC/httpx 传输。

**配置与安全**：工具构造时校验 ``server_url``（``assert_mcp_http_url_allowed``）；``mcp_config``
含超时、TLS、协议扩展头等。

**能力发现**：``list_tools`` 取自 SDK ``ClientSession.list_tools``，结果转为与旧实现一致的
``dict`` 列表（含 ``inputSchema``）；同实例内对 ``tools/list`` 做短时 TTL 缓存。

**调度模式**：仍要求应用层先 ``list_tools`` 再 ``call_tool``（见 ``McpStreamableHttpDispatchTool``）。

**结果**：``tools/call`` 结果以 dict 形式交给 ``_extract_call_result_text``；若 ``isError`` 则抛
:class:`McpProtocolError`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, ClassVar, Literal, Self

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import create_session
from mcp.shared.exceptions import McpError
from mcp.types import Implementation
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from app.agent.adapters.tools.dynamic.mcp_transport import (
    McpNotImplementedTransport,
    McpProtocolError,
    McpSessionExpiredError,
    McpTransport,
    McpTransportError,
)
from app.agent.kernel.tool_runtime import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    assert_mcp_http_url_allowed,
    build_external_tool_error_json,
    build_external_tool_json,
    get_current_tool_invocation,
)

logger = logging.getLogger(__name__)

# 连接层失败（对端无响应、拒绝、路由不可达等），与 JSON-RPC 业务错误区分
_MCP_CONN_FAIL_HINT = (
    " 常见原因：MCP 未启动、URL 主机/端口/路径错误、防火墙；"
    "若后端在容器内、MCP 在宿主机，请用 host.docker.internal 或宿主机 IP，勿用 127.0.0.1。"
)


def _initialize_http_error_hint(
    status: int,
    resp: httpx.Response,
    *,
    transport: Literal["streamable_http", "sse"] = "streamable_http",
) -> str:
    """对非 2xx 的补充说明（与外层「HTTP {status}」连用，勿再重复状态码）。"""
    if status == 405:
        allow_raw = (resp.headers.get("Allow") or resp.headers.get("allow") or "").strip()
        allow_part = f"（Allow: {allow_raw}）" if allow_raw else ""
        extra = ""
        if allow_raw:
            tokens = allow_raw.replace(",", " ").split()
            methods = {t.strip().upper() for t in tokens if t.strip()}
            if "POST" not in methods and ("GET" in methods or "HEAD" in methods):
                extra = " 多为浏览器页面（仅 GET/HEAD），非 MCP；"
        if transport == "sse":
            path_hint = (
                "请填写 MCP 的 SSE 入口（路径常见为 /sse）；"
                "若 Allow 仅有 GET/HEAD，多为文档页而非 MCP。"
            )
        else:
            path_hint = (
                "请填写 Streamable HTTP 的 JSON-RPC 入口（路径示例：/mcp、/message）；"
                "勿填仅支持 GET 的文档首页。"
            )
        return f"：该路径不接受 POST。{extra}{path_hint}{allow_part}"
    if status == 404:
        return "：路径不存在；请核对文档中的 MCP HTTP 入口路径与完整 URL。"
    return ""


def _trace_id_for_log() -> str:
    inv = get_current_tool_invocation()
    if inv is None or not inv.trace_id:
        return "-"
    tid = str(inv.trace_id).strip()
    return tid[:48] if len(tid) > 48 else tid


def _arguments_preview(arguments: dict[str, Any], *, max_len: int = 1200) -> str:
    try:
        s = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(arguments)
    if len(s) > max_len:
        return f"{s[:max_len]}...(truncated, len={len(s)})"
    return s


def _leaf_exception(exc: BaseException) -> BaseException:
    """展开 ``ExceptionGroup``（含 asyncio TaskGroup 的「unhandled errors in a TaskGroup」）取首个子异常。"""
    e: BaseException = exc
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


# 与 DB/控制台默认一致；请求头 ``MCP-Protocol-Version`` 使用本值（底层握手由 mcp SDK 完成）。
MCP_PROTOCOL_VERSION_DEFAULT = "2025-06-18"
# ``tools/list`` 短时缓存（秒）：同传输实例内减少重复往返。
_TOOLS_LIST_CACHE_TTL_SEC = 120.0


def _normalize_endpoint_url(url: str) -> str:
    return url.strip().rstrip("/")


def _rpc_id_matches(body_id: Any, expect_id: int) -> bool:
    if body_id is None:
        return False
    try:
        return int(body_id) == int(expect_id)
    except (TypeError, ValueError):
        return body_id == expect_id


def _parse_sse_buffer_for_id(text: str, expect_id: int) -> dict[str, Any]:
    """从已缓冲的 SSE 正文中找出匹配 ``id`` 的 JSON-RPC 对象（单测/静态解析）。"""
    found: dict[str, Any] | None = None
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            s = line.strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("id") is None:
                continue
            if _rpc_id_matches(obj.get("id"), expect_id):
                found = obj
                break
        if found is not None:
            break
    if found is None:
        raise McpProtocolError("SSE 响应中未找到匹配的 JSON-RPC 结果")
    return found


# 兼容旧单测：曾清理进程内 MCP 会话缓存；现由 mcp SDK 管理连接，保留空 dict 仅支持 ``.clear()``。
_mcp_session_store: dict[tuple[str, str], Any] = {}


class McpStreamableHttpTransport:
    """MCP Streamable HTTP；底层为 ``langchain_mcp_adapters.sessions.create_session`` + ``mcp.ClientSession``。"""

    _mcp_http_error_transport: ClassVar[Literal["streamable_http", "sse"]] = "streamable_http"

    def __init__(
        self,
        server_url: str,
        *,
        protocol_version: str = MCP_PROTOCOL_VERSION_DEFAULT,
        extra_headers: dict[str, str] | None = None,
        verify_ssl: bool = True,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.server_url = _normalize_endpoint_url(server_url)
        self.protocol_version = protocol_version
        self.extra_headers = dict(extra_headers or {})
        self.verify_ssl = verify_ssl
        self.timeout = timeout or httpx.Timeout(60.0, connect=30.0)
        self._tools_list_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        #: 按 ``platform_session_id`` 记录是否已成功执行过 ``tools/list``（调度模式前置）
        self._list_tools_completed: set[str] = set()

    def _timeout_timedelta(self) -> timedelta:
        t = self.timeout
        if isinstance(t, httpx.Timeout):
            ro = t.read
            sec = float(ro) if ro is not None else 60.0
        else:
            sec = 60.0
        sec = max(1.0, min(300.0, sec))
        return timedelta(seconds=sec)

    def _streamable_connection(self) -> dict[str, Any]:
        headers: dict[str, Any] = {
            "MCP-Protocol-Version": self.protocol_version,
            **self.extra_headers,
        }
        return {
            "transport": "streamable_http",
            "url": self.server_url,
            "headers": headers,
            "timeout": self._timeout_timedelta(),
            "sse_read_timeout": timedelta(seconds=300.0),
            "httpx_client_factory": self._httpx_client_factory(),
            "session_kwargs": {
                "client_info": Implementation(name="agent-forge-backend", version="1.0.0"),
            },
        }

    def _httpx_client_factory(self):
        verify = self.verify_ssl

        def factory(
            headers: dict[str, str] | None,
            timeout: httpx.Timeout | None,
            auth: httpx.Auth | None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                auth=auth,
                verify=verify,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

        return factory

    async def _run_with_session(self, fn: Any, *, retry_404: bool = False) -> Any:
        """在已 ``initialize`` 的 ClientSession 上执行 ``fn``。"""

        async def _once() -> Any:
            async with create_session(self._streamable_connection()) as session:
                await session.initialize()
                return await fn(session)

        try:
            return await _once()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            hint = _initialize_http_error_hint(
                status,
                e.response,
                transport=type(self)._mcp_http_error_transport,
            )
            if status == 404 and retry_404:
                return await _once()
            if status == 404:
                raise McpSessionExpiredError(f"MCP session 失效: HTTP 404{hint}") from e
            raise McpTransportError(f"MCP HTTP 请求失败: HTTP {status}{hint}") from e
        except McpProtocolError:
            raise
        except McpError as e:
            ed = e.error
            raise McpProtocolError(f"MCP error ({ed.code}): {ed.message}") from e
        except httpx.RequestError as e:
            raise McpTransportError(f"MCP 网络错误: {e}{_MCP_CONN_FAIL_HINT}") from e
        except Exception as e:
            # mcp / anyio 常将真实错误包在 ExceptionGroup 里，探测接口否则会只看到 TaskGroup 文案
            leaf = _leaf_exception(e)
            if leaf is not e:
                logger.warning(
                    "mcp.exception_group.unwrapped",
                    extra={
                        "server_url": self.server_url,
                        "wrapper_type": type(e).__name__,
                        "wrapper_msg": str(e)[:800],
                        "leaf_type": type(leaf).__name__,
                        "leaf_msg": str(leaf)[:800],
                    },
                )
            if isinstance(leaf, McpProtocolError):
                raise leaf from e
            if isinstance(leaf, httpx.HTTPStatusError):
                status = leaf.response.status_code
                hint = _initialize_http_error_hint(
                    status,
                    leaf.response,
                    transport=type(self)._mcp_http_error_transport,
                )
                if status == 404 and retry_404:
                    return await _once()
                if status == 404:
                    raise McpSessionExpiredError(f"MCP session 失效: HTTP 404{hint}") from e
                raise McpTransportError(f"MCP HTTP 请求失败: HTTP {status}{hint}") from e
            if isinstance(leaf, McpError):
                ed = leaf.error
                raise McpProtocolError(f"MCP error ({ed.code}): {ed.message}") from e
            if isinstance(leaf, httpx.RequestError):
                raise McpTransportError(f"MCP 网络错误: {leaf}{_MCP_CONN_FAIL_HINT}") from e
            raise McpTransportError(
                f"MCP 调用失败: {type(leaf).__name__}: {leaf}",
            ) from e

    async def aclose(self) -> None:
        self._tools_list_cache.clear()
        self._list_tools_completed.clear()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[McpStreamableHttpTransport, None]:
        """异步上下文：退出时调用 ``aclose``。"""
        try:
            yield self
        finally:
            await self.aclose()

    def has_completed_list_tools(self, platform_session_id: str | None) -> bool:
        return (platform_session_id or "") in self._list_tools_completed

    def _mark_list_tools_completed(self, platform_session_id: str | None) -> None:
        self._list_tools_completed.add(platform_session_id or "")

    async def initialize(self, *, platform_session_id: str | None = None) -> None:
        async def _noop(_session: Any) -> None:
            return None

        await self._run_with_session(_noop)

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        platform_session_id: str | None = None,
    ) -> Any:
        """有限的 JSON-RPC 方法映射（优先使用 ``list_tools`` / ``call_tool``）。"""

        async def _send(session: Any) -> Any:
            if method == "tools/list":
                r = await session.list_tools()
                return {
                    "tools": [t.model_dump(by_alias=True, exclude_none=True) for t in r.tools],
                }
            if method == "tools/call":
                p = params or {}
                name = p.get("name")
                if not name or not str(name).strip():
                    raise McpProtocolError("tools/call 缺少 name")
                arguments = p.get("arguments") if isinstance(p.get("arguments"), dict) else {}
                raw = await session.call_tool(str(name).strip(), arguments)
                out = raw.model_dump(mode="json", exclude_none=True)
                if raw.isError:
                    raise McpProtocolError(
                        f"tools/call 返回错误: {_extract_call_result_text(out)}",
                    )
                return out
            raise McpTransportError(
                f"MCP send_request 未实现的 method: {method!r}（请使用 SDK 封装方法或扩展映射）",
            )

        return await self._run_with_session(_send, retry_404=True)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> Any:
        t_rpc = time.perf_counter()

        async def _call(session: Any) -> Any:
            raw = await session.call_tool(tool_name, arguments)
            out = raw.model_dump(mode="json", exclude_none=True)
            if raw.isError:
                raise McpProtocolError(
                    f"tools/call 返回错误: {_extract_call_result_text(out)}",
                )
            return out

        try:
            return await self._run_with_session(_call, retry_404=True)
        finally:
            elapsed_ms = (time.perf_counter() - t_rpc) * 1000.0
            logger.info(
                "MCP tools/call logical=%s remote=%s server=%s ms=%.2f trace=%s args=%s",
                logical_tool_name or "-",
                tool_name,
                self.server_url,
                elapsed_ms,
                _trace_id_for_log(),
                _arguments_preview(arguments),
            )

    async def list_tools(
        self,
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        pk = platform_session_id or ""
        now = time.time()
        hit = self._tools_list_cache.get(pk)
        if hit is not None and now - hit[0] <= _TOOLS_LIST_CACHE_TTL_SEC:
            tools = hit[1]
            self._mark_list_tools_completed(platform_session_id)
            logger.info(
                "MCP tools/list (cache) logical=%s server=%s count=%s trace=%s",
                logical_tool_name or "-",
                self.server_url,
                len(tools),
                _trace_id_for_log(),
            )
            return tools

        async def _list(session: Any) -> list[dict[str, Any]]:
            r = await session.list_tools()
            return [t.model_dump(by_alias=True, exclude_none=True) for t in r.tools]

        tools = await self._run_with_session(_list, retry_404=True)
        self._tools_list_cache[pk] = (time.time(), tools)
        self._mark_list_tools_completed(platform_session_id)
        logger.info(
            "MCP tools/list logical=%s server=%s count=%s trace=%s",
            logical_tool_name or "-",
            self.server_url,
            len(tools),
            _trace_id_for_log(),
        )
        return tools


class McpSseTransport(McpStreamableHttpTransport):
    """MCP HTTP+SSE：``langchain_mcp_adapters.create_session`` 使用 ``transport: sse``（典型 URL 如 ``/sse``）。"""

    _mcp_http_error_transport: ClassVar[Literal["streamable_http", "sse"]] = "sse"

    def _streamable_connection(self) -> dict[str, Any]:
        t = self.timeout
        if isinstance(t, httpx.Timeout):
            sec = float(t.read) if t.read is not None else 60.0
        else:
            sec = 60.0
        sec = max(1.0, min(300.0, sec))
        headers: dict[str, Any] = {
            "MCP-Protocol-Version": self.protocol_version,
            **self.extra_headers,
        }
        return {
            "transport": "sse",
            "url": self.server_url,
            "headers": headers,
            "timeout": sec,
            "sse_read_timeout": max(300.0, sec),
            "httpx_client_factory": self._httpx_client_factory(),
            "session_kwargs": {
                "client_info": Implementation(name="agent-forge-backend", version="1.0.0"),
            },
        }


def _tools_list_from_result(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    tools = result.get("tools")
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict)]


# ---------------------------------------------------------------------------
# tools/list 元数据解析、调用前参数归一化、结果文本化
# ---------------------------------------------------------------------------


def _input_schema_from_tool_entry(tool: dict[str, Any]) -> dict[str, Any] | None:
    sch = tool.get("inputSchema") or tool.get("input_schema")
    if isinstance(sch, dict) and sch:
        return sch
    return None


def _find_tool_entry_by_name(
    tools: list[dict[str, Any]], remote_name: str
) -> dict[str, Any] | None:
    rn = remote_name.strip()
    for t in tools:
        if str(t.get("name", "")).strip() == rn:
            return t
    return None


def _pick_remote_tool_name_from_list(tools: list[dict[str, Any]], logical_name: str) -> str:
    """直连模式：未配置 ``mcp_tool_name`` 时，根据 ``tools/list`` 解析远端 ``tools/call`` 的 ``name``。"""
    ln = logical_name.strip()
    names: list[str] = []
    for t in tools:
        n = str(t.get("name", "")).strip()
        if n:
            names.append(n)
    if ln and ln in names:
        return ln
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError("MCP tools/list 未返回任何工具")
    raise ValueError(
        f"无法在 tools/list 中解析远端工具名：logical_name={ln!r}，"
        f"可用远端名={names!r}；请在 tools_config 中设置 mcp_tool_name，"
        "或使注册名与远端 name 一致，或使用仅含单个工具的 MCP 端点。"
    )


def _unwrap_nested_tool_arguments(raw: dict[str, Any]) -> dict[str, Any]:
    """模型偶发把 MCP 参数再包一层 ``{"arguments": {...}}`` 时展开为真正传给远端的 dict。"""
    if len(raw) != 1:
        return dict(raw)
    inner = raw.get("arguments")
    if isinstance(inner, dict):
        return dict(inner)
    return dict(raw)


def _normalize_mcp_tool_call_arguments(call_args: dict[str, Any]) -> dict[str, Any]:
    """调用前仅做通用处理：展开误嵌套的 ``{"arguments": {...}}``，不按具体远端工具名硬编码。"""
    return _unwrap_nested_tool_arguments(call_args)


def _extract_call_result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            if parts:
                return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LangChain 工具入参：仅 ``arguments``，与 MCP ``tools/call.params.arguments`` / inputSchema 一致
# ---------------------------------------------------------------------------


class McpStreamableInvokeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="传给 MCP ``tools/call`` 的 ``arguments``；字段以远端 ``tools/list`` 的 inputSchema 为准。",
    )


class McpStreamableDispatchArgs(BaseModel):
    """调度模式：须先 ``list_tools`` 再 ``call_tool``（由服务端强制顺序，不做参数 JSON Schema 校验）。"""

    model_config = ConfigDict(extra="forbid")

    step: Literal["list_tools", "call_tool"] = Field(
        description="list_tools：获取该 MCP 服务 tools/list；call_tool：对 remote_tool_name 执行 tools/call",
    )
    remote_tool_name: str | None = Field(
        default=None,
        max_length=128,
        description="仅 step=call_tool 时需要，对应远端 tools/call 的 name",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="仅 step=call_tool 时使用，传给远端 ``tools/call`` 的 arguments。",
    )

    @model_validator(mode="after")
    def _call_requires_remote_name(self) -> Self:
        if self.step == "call_tool":
            if self.remote_tool_name is None or not str(self.remote_tool_name).strip():
                raise ValueError("call_tool 时必须提供非空 remote_tool_name")
        return self


class McpStreamableToolConfig(BaseModel):
    logical_name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=512)
    server_url: str
    #: 与 ``RuntimeMcpTool.transport_type`` 对齐：``http``=Streamable HTTP，``sse``=HTTP+SSE；其它未实现为 :class:`McpNotImplementedTransport`。
    transport_type: str = Field(default="http", min_length=2, max_length=32)
    #: 调度模式固定为 ``__dispatch__``；直连模式下为显式远端名，``None`` 表示首次调用前经 ``tools/list`` 解析
    remote_tool_name: str | None = Field(default=None, max_length=128)
    protocol_version: str = Field(default=MCP_PROTOCOL_VERSION_DEFAULT, min_length=8, max_length=32)
    timeout_s: float = Field(default=60.0, ge=1.0, le=300.0)
    verify_ssl: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("remote_tool_name", mode="before")
    @classmethod
    def _empty_remote_as_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _validate_mcp_server_url(self) -> Self:
        tt = (self.transport_type or "http").strip().lower()
        if tt in ("http", "sse"):
            assert_mcp_http_url_allowed(self.server_url)
        return self


def create_mcp_transport_for_tool_config(cfg: McpStreamableToolConfig) -> McpTransport:
    """按工具配置构造 MCP 传输（``http``=Streamable HTTP，``sse``=HTTP+SSE）。"""
    tt = (cfg.transport_type or "http").strip().lower()
    if tt == "http":
        return McpStreamableHttpTransport(
            cfg.server_url,
            protocol_version=cfg.protocol_version,
            extra_headers=cfg.extra_headers,
            verify_ssl=cfg.verify_ssl,
            timeout=httpx.Timeout(cfg.timeout_s, connect=min(30.0, cfg.timeout_s)),
        )
    if tt == "sse":
        return McpSseTransport(
            cfg.server_url,
            protocol_version=cfg.protocol_version,
            extra_headers=cfg.extra_headers,
            verify_ssl=cfg.verify_ssl,
            timeout=httpx.Timeout(cfg.timeout_s, connect=min(30.0, cfg.timeout_s)),
        )
    return McpNotImplementedTransport(
        transport_type=tt,
        server_url=cfg.server_url.strip() or None,
    )


class McpBaseTool(BaseTool, ABC):
    """MCP 工具共享：配置、:class:`McpTransport` 懒加载、关闭、成功/错误信封。"""

    _cfg: McpStreamableToolConfig = PrivateAttr()
    _mcp_client: McpTransport | None = PrivateAttr(default=None)

    def __init__(self, cfg: McpStreamableToolConfig) -> None:
        super().__init__(name=cfg.logical_name, description=cfg.description)
        self._cfg = cfg

    def _create_mcp_client(self) -> McpTransport:
        return create_mcp_transport_for_tool_config(self._cfg)

    def _mcp(self) -> McpTransport:
        if self._mcp_client is None:
            self._mcp_client = self._create_mcp_client()
        return self._mcp_client

    async def aclose(self) -> None:
        if self._mcp_client is not None:
            await self._mcp_client.aclose()
            self._mcp_client = None

    @staticmethod
    def _elapsed_ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000.0, 2)

    def _error_data_base(self, t0: float, **extra: Any) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tool_name": self._cfg.logical_name,
            "server_url": self._cfg.server_url,
            "response_time_ms": self._elapsed_ms(t0),
        }
        data.update(extra)
        return data

    def _success_envelope(
        self,
        t0: float,
        *,
        text: str,
        raw_result: Any,
        mcp_tool_name: str | None,
        step: str | None = None,
    ) -> str:
        result: dict[str, Any] = {
            "tool_name": self._cfg.logical_name,
            "status": "ok",
            "text": text,
            "response_time_ms": self._elapsed_ms(t0),
            "raw_result": raw_result,
        }
        if step is not None:
            result["step"] = step
            result["mcp_tool_name"] = mcp_tool_name
        else:
            result["mcp_tool_name"] = mcp_tool_name
        return build_external_tool_json(kind="mcp", result=result, is_error=False)

    def _invalid_params_envelope(self, t0: float, message: str, **extra: Any) -> str:
        return build_external_tool_error_json(
            kind="mcp",
            code=JSONRPC_INVALID_PARAMS,
            message=message,
            data=self._error_data_base(t0, **extra),
        )

    def _protocol_transport_envelope(
        self,
        t0: float,
        exc: Exception,
        *,
        log_remote: Any = None,
        **data_extra: Any,
    ) -> str:
        logger.warning(
            "mcp_tool.error logical=%s remote=%s type=%s: %s",
            self._cfg.logical_name,
            log_remote,
            type(exc).__name__,
            exc,
        )
        return build_external_tool_error_json(
            kind="mcp",
            code=JSONRPC_INTERNAL_ERROR,
            message=f"MCP 调用失败: {exc!s}",
            data=self._error_data_base(t0, **data_extra),
        )

    def _execution_failed_envelope(
        self, t0: float, exc: Exception, *, log_remote: Any = None
    ) -> str:
        logger.exception(
            "mcp_tool.execution_failed logical=%s remote=%s: %s",
            self._cfg.logical_name,
            log_remote,
            exc,
        )
        return build_external_tool_error_json(
            kind="mcp",
            code=JSONRPC_INTERNAL_ERROR,
            message=f"MCP 工具执行失败: {exc!s}",
            data=self._error_data_base(t0),
        )


class McpStreamableHttpTool(McpBaseTool):
    args_schema: ClassVar[type[BaseModel]] = McpStreamableInvokeArgs
    #: 未配置 ``remote_tool_name`` 时，解析自 ``tools/list`` 后缓存
    _resolved_remote_for_call: str | None = PrivateAttr(default=None)

    async def aclose(self) -> None:
        await super().aclose()
        self._resolved_remote_for_call = None

    async def _effective_remote_tool_name(self, platform_session_id: str | None) -> str:
        cfg = self._cfg
        if cfg.remote_tool_name is not None:
            return str(cfg.remote_tool_name).strip()
        if self._resolved_remote_for_call is not None:
            return self._resolved_remote_for_call
        tools = await self._mcp().list_tools(
            platform_session_id=platform_session_id,
            logical_tool_name=cfg.logical_name,
        )
        resolved = _pick_remote_tool_name_from_list(tools, cfg.logical_name)
        self._resolved_remote_for_call = resolved
        return resolved

    def _run(
        self,
        arguments: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return asyncio.run(self._arun(arguments=arguments, run_manager=None))

    async def _arun(
        self,
        arguments: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        args = arguments if isinstance(arguments, dict) else {}
        return await self._arun_with_mcp_arguments(dict(args))

    async def _arun_with_mcp_arguments(self, call_args: dict[str, Any]) -> str:
        cfg = self._cfg
        t0 = time.perf_counter()

        inv = get_current_tool_invocation()
        platform_sid = inv.session_id if inv else None

        try:
            call_args = _normalize_mcp_tool_call_arguments(call_args)
            remote = await self._effective_remote_tool_name(platform_sid)
            raw = await self._mcp().call_tool(
                remote,
                call_args,
                platform_session_id=platform_sid,
                logical_tool_name=cfg.logical_name,
            )
            text = _extract_call_result_text(raw)
            return self._success_envelope(
                t0,
                text=text,
                raw_result=raw,
                mcp_tool_name=remote,
                step=None,
            )
        except ValueError as e:
            logger.warning(
                "mcp_tool.resolve_remote_failed logical=%s type=%s: %s",
                cfg.logical_name,
                type(e).__name__,
                e,
            )
            return self._invalid_params_envelope(t0, str(e))
        except (McpProtocolError, McpTransportError) as e:
            remote = self._resolved_remote_for_call or cfg.remote_tool_name
            return self._protocol_transport_envelope(
                t0,
                e,
                log_remote=remote,
                mcp_tool_name=remote,
            )
        except Exception as e:
            remote = self._resolved_remote_for_call or cfg.remote_tool_name
            return self._execution_failed_envelope(t0, e, log_remote=remote)


class McpStreamableHttpDispatchTool(McpBaseTool):
    """单 MCP 端点上的调度工具：``list_tools`` / ``call_tool``（``dispatch_mode`` 启用）。

    须先成功执行 ``step=list_tools``（本会话内），再 ``call_tool``；不在服务端做 JSON Schema 参数校验。
    """

    args_schema: ClassVar[type[BaseModel]] = McpStreamableDispatchArgs

    def _run(
        self,
        step: str = "list_tools",
        remote_tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return asyncio.run(
            self._arun(
                step=step,
                remote_tool_name=remote_tool_name,
                arguments=arguments,
                run_manager=None,
            ),
        )

    async def _arun(
        self,
        step: Literal["list_tools", "call_tool"] = "list_tools",
        remote_tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        cfg = self._cfg
        payload = McpStreamableDispatchArgs.model_validate(
            {
                "step": step,
                "remote_tool_name": remote_tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            },
        )
        t0 = time.perf_counter()
        inv = get_current_tool_invocation()
        platform_sid = inv.session_id if inv else None

        try:
            if payload.step == "list_tools":
                tools = await self._mcp().list_tools(
                    platform_session_id=platform_sid,
                    logical_tool_name=cfg.logical_name,
                )
                text = json.dumps(tools, ensure_ascii=False)
                raw: Any = {"tools": tools}
                mcp_name = None
            else:
                name = str(payload.remote_tool_name or "").strip()
                call_in = _normalize_mcp_tool_call_arguments(dict(payload.arguments))
                mcp = self._mcp()
                if not mcp.has_completed_list_tools(platform_sid):
                    return self._invalid_params_envelope(
                        t0,
                        (
                            "调度模式须先在本会话成功调用 step=list_tools 获取 tools/list 后，"
                            "再调用 step=call_tool。"
                        ),
                        mcp_tool_name=name,
                        step=payload.step,
                    )
                raw = await mcp.call_tool(
                    name,
                    call_in,
                    platform_session_id=platform_sid,
                    logical_tool_name=cfg.logical_name,
                )
                text = _extract_call_result_text(raw)
                mcp_name = name

            return self._success_envelope(
                t0,
                text=text,
                raw_result=raw,
                mcp_tool_name=mcp_name,
                step=payload.step,
            )
        except (McpProtocolError, McpTransportError) as e:
            log_remote = payload.remote_tool_name if payload.step == "call_tool" else None
            return self._protocol_transport_envelope(
                t0,
                e,
                log_remote=log_remote,
                mcp_tool_name=(payload.remote_tool_name if payload.step == "call_tool" else None),
                step=payload.step,
            )
        except Exception as e:
            return self._execution_failed_envelope(t0, e)


def _headers_from_connection(conn: dict[str, Any] | None) -> dict[str, str]:
    if not conn:
        return {}
    raw = conn.get("headers")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v if isinstance(v, str) else str(v) for k, v in raw.items()}


def make_mcp_streamable_http_tool(
    *,
    logical_name: str,
    description: str,
    mcp_config: dict[str, Any] | None,
) -> BaseTool:
    """由持久化合并配置构造 Streamable HTTP MCP 工具。

    **推荐：调度模式（须先 list 再 call，由模型按 tools/list 结果填参）**

    在 ``mcp_config`` 或 ``tools_config`` 中设置 ``dispatch_mode: true``，将注册
    ``McpStreamableHttpDispatchTool``。须先 ``step=list_tools`` 成功，再 ``step=call_tool``；
    服务端不做 JSON Schema 参数校验，远端错误由 MCP 返回。

    单端点直连模式（``dispatch_mode: false``）下使用 ``McpStreamableHttpTool``；
    ``tools_config.mcp_tool_name``（或 ``tool_name`` / ``name``）为远端 ``tools/call`` 的 ``name``。
    未设置时由 ``tools/list`` 动态解析：优先与注册名 ``logical_name`` 完全一致的远端 ``name``；
    若仅有一个工具则使用该工具名；多工具且名称均不匹配时须显式配置 ``mcp_tool_name``。
    """
    mc = dict(mcp_config) if mcp_config else {}
    server_url = str(mc.get("server_url") or "").strip()
    if not server_url:
        raise ValueError("MCP server_url 不能为空")

    tools_cfg = mc.get("tools_config") if isinstance(mc.get("tools_config"), dict) else {}
    dispatch_mode = bool(mc.get("dispatch_mode") or tools_cfg.get("dispatch_mode"))

    explicit = tools_cfg.get("mcp_tool_name") or tools_cfg.get("tool_name") or tools_cfg.get("name")
    if dispatch_mode:
        remote = "__dispatch__"
    elif explicit is not None and str(explicit).strip():
        remote = str(explicit).strip()
    else:
        remote = None

    timeout_s = float(mc.get("timeout_seconds") or 60.0)
    timeout_s = max(1.0, min(300.0, timeout_s))
    verify_ssl = bool(mc.get("verify_ssl", True))
    protocol_version = str(mc.get("protocol_version") or MCP_PROTOCOL_VERSION_DEFAULT).strip()
    extra_headers = _headers_from_connection(mc)
    transport_type = str(mc.get("transport_type") or "http").strip()

    cfg = McpStreamableToolConfig(
        logical_name=logical_name.strip(),
        description=description.strip(),
        server_url=server_url,
        transport_type=transport_type,
        remote_tool_name=remote,
        protocol_version=protocol_version,
        timeout_s=timeout_s,
        verify_ssl=verify_ssl,
        extra_headers=extra_headers,
    )
    if dispatch_mode:
        return McpStreamableHttpDispatchTool(cfg=cfg)
    return McpStreamableHttpTool(cfg=cfg)


# 向后兼容：旧模块/测试使用的类名
McpStreamableHttpClient = McpStreamableHttpTransport
McpSseClient = McpSseTransport


# 单测兼容：缓冲解析（非流式）
def _parse_sse_for_id(text: str, expect_id: int) -> dict[str, Any]:
    return _parse_sse_buffer_for_id(text, expect_id)
