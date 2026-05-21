"""Agent 工具侧运行时（刻意单文件）：外部工具信封、调用期 ContextVar、MCP ``server_url`` SSRF、工具生命周期协议。

与 ``app.core.constants.ext_tool.ToolKind``（ORM）区分；此处无 LangChain / LangGraph 依赖。
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import Enum, IntEnum, StrEnum, auto
from typing import Any, Protocol, runtime_checkable
from urllib.parse import ParseResult, urlparse

from app.agent.kernel.ports import ToolInvocationContext

# ---------------------------------------------------------------------------
# 信封契约（JSON-RPC 码与 kind）
# ---------------------------------------------------------------------------


class JsonRpcErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


JSONRPC_PARSE_ERROR = JsonRpcErrorCode.PARSE_ERROR
JSONRPC_INVALID_REQUEST = JsonRpcErrorCode.INVALID_REQUEST
JSONRPC_METHOD_NOT_FOUND = JsonRpcErrorCode.METHOD_NOT_FOUND
JSONRPC_INVALID_PARAMS = JsonRpcErrorCode.INVALID_PARAMS
JSONRPC_INTERNAL_ERROR = JsonRpcErrorCode.INTERNAL_ERROR


class ExternalToolKind(StrEnum):
    HTTP = "http"
    MCP = "mcp"


# ---------------------------------------------------------------------------
# 工具生命周期（进程级资源释放）
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolLifecycle(Protocol):
    """持有外部资源（HTTP 客户端、连接池等）的工具可实现，在进程退出时释放。"""

    async def aclose(self) -> None:
        """异步释放资源；无异步清理需求可实现为空操作。"""
        ...


# ---------------------------------------------------------------------------
# 外部工具调用期 ContextVar（与 ``ports.ToolInvocationContext`` / GraphBuildContext 对齐）
# ---------------------------------------------------------------------------


class ToolInvocationBinder:
    """将 ``ToolInvocationContext`` 绑定到当前任务上下文（``ContextVar``）。"""

    __slots__ = ("_current",)

    def __init__(self, *, var_name: str = "agent_tool_invocation") -> None:
        self._current: ContextVar[ToolInvocationContext | None] = ContextVar(
            var_name,
            default=None,
        )

    def get(self) -> ToolInvocationContext | None:
        return self._current.get()

    def require(self) -> ToolInvocationContext:
        ctx = self._current.get()
        if ctx is None:
            raise RuntimeError(
                "未设置工具调用上下文，请在 `tool_invocation_context` 内执行工具逻辑。"
            )
        return ctx

    def attach(self, ctx: ToolInvocationContext | None) -> Token[ToolInvocationContext | None]:
        return self._current.set(ctx)

    def detach(self, token: Token[ToolInvocationContext | None]) -> None:
        self._current.reset(token)

    @contextmanager
    def scope(self, ctx: ToolInvocationContext | None) -> Generator[None, None, None]:
        token = self.attach(ctx)
        try:
            yield
        finally:
            self.detach(token)


_tool_invocation_binder = ToolInvocationBinder()


def get_current_tool_invocation() -> ToolInvocationContext | None:
    return _tool_invocation_binder.get()


def require_tool_invocation() -> ToolInvocationContext:
    return _tool_invocation_binder.require()


@contextmanager
def tool_invocation_context(
    ctx: ToolInvocationContext | None,
) -> Generator[None, None, None]:
    with _tool_invocation_binder.scope(ctx):
        yield


# ---------------------------------------------------------------------------
# MCP server_url SSRF
# ---------------------------------------------------------------------------


class McpUrlNotAllowedError(ValueError):
    """URL 未通过安全策略（非 http/https、解析到保留/私网地址等）。"""


class McpSsrfPolicy(Protocol):
    def permits_private_targets(self) -> bool: ...
    def forbidden_tcp_ports(self) -> frozenset[int]: ...
    def is_reachable_ip_blocked(
        self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address
    ) -> bool: ...


class EnvBackedMcpSsrfPolicy:
    FORBIDDEN_PORTS = frozenset({22, 23, 3306, 5432, 6379, 27017, 9200})

    def permits_private_targets(self) -> bool:
        """主机名为域名时是否跳过 DNS（固定为否：始终解析后再放行）。"""
        return False

    def forbidden_tcp_ports(self) -> frozenset[int]:
        return self.FORBIDDEN_PORTS

    def is_reachable_ip_blocked(self, _ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """不再按私网/保留地址拦截解析结果（若需收敛访问面请在网络层控制）。"""
        return False


class McpHttpUrlValidator:
    class _HostResolution(Enum):
        COMPLETE = auto()
        NEED_DNS = auto()

    __slots__ = ("_policy",)

    def __init__(self, policy: McpSsrfPolicy | None = None) -> None:
        self._policy = policy or EnvBackedMcpSsrfPolicy()

    @staticmethod
    def _idna_host(host: str) -> str:
        try:
            return host.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            return host

    def _check_port(self, parsed: ParseResult) -> None:
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port if parsed.port is not None else default_port
        if port in self._policy.forbidden_tcp_ports():
            raise McpUrlNotAllowedError(f"禁止访问该端口: {port}（SSRF 防护）")

    def _parse_http_url(self, url: str) -> tuple[str, str]:
        raw = (url or "").strip()
        if not raw:
            raise McpUrlNotAllowedError("server_url 为空")
        parsed = urlparse(raw)
        if parsed.scheme not in ("http", "https"):
            raise McpUrlNotAllowedError("仅允许 http:// 或 https:// 的 MCP 端点")
        host = parsed.hostname
        if not host:
            raise McpUrlNotAllowedError("URL 缺少主机名")
        self._check_port(parsed)
        return raw, host

    def _host_resolution_phase(self, host: str) -> _HostResolution:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            if self._policy.permits_private_targets():
                return self._HostResolution.COMPLETE
            return self._HostResolution.NEED_DNS
        if self._policy.is_reachable_ip_blocked(ip):
            raise McpUrlNotAllowedError("禁止访问该 IP 段（SSRF 防护）")
        return self._HostResolution.COMPLETE

    def _resolve_hostname_and_check_blocked(self, host: str) -> None:
        host_lookup = self._idna_host(host)
        try:
            infos = socket.getaddrinfo(host_lookup, None, type=socket.SOCK_STREAM)
        except OSError as e:
            raise McpUrlNotAllowedError(f"无法解析主机: {host!r}") from e

        for info in infos:
            addr_s = info[4][0]
            try:
                ip = ipaddress.ip_address(addr_s)
            except ValueError:
                continue
            if self._policy.is_reachable_ip_blocked(ip):
                raise McpUrlNotAllowedError(f"主机解析到禁止网段: {addr_s}")

    def assert_allowed(self, url: str) -> str:
        raw, host = self._parse_http_url(url)
        if self._host_resolution_phase(host) is self._HostResolution.NEED_DNS:
            self._resolve_hostname_and_check_blocked(host)
        return raw

    async def assert_allowed_async(self, url: str) -> str:
        raw, host = self._parse_http_url(url)
        if self._host_resolution_phase(host) is self._HostResolution.NEED_DNS:
            await asyncio.to_thread(self._resolve_hostname_and_check_blocked, host)
        return raw


_default_mcp_url_validator = McpHttpUrlValidator()


def assert_mcp_http_url_allowed(url: str) -> str:
    return _default_mcp_url_validator.assert_allowed(url)


async def assert_mcp_http_url_allowed_async(url: str) -> str:
    return await _default_mcp_url_validator.assert_allowed_async(url)


# ---------------------------------------------------------------------------
# 外部工具持久化 JSON 信封（external_v1）
# ---------------------------------------------------------------------------


class ExternalToolEnvelopeBuilder:
    __slots__ = ("_invocation_getter",)

    def __init__(
        self,
        invocation_getter: Callable[[], ToolInvocationContext | None],
    ) -> None:
        self._invocation_getter = invocation_getter

    def _base_payload(self, *, kind: ExternalToolKind | str, is_error: bool) -> dict[str, Any]:
        inv = self._invocation_getter()
        rid = inv.request_id if inv else None
        return {
            "jsonrpc": "2.0",
            "tool_protocol": "external_v1",
            "kind": str(kind),
            "session_id": inv.session_id if inv else None,
            "request_id": rid,
            "trace_id": inv.trace_id if inv else None,
            "isError": is_error,
            "id": rid,
        }

    def build_result_json(
        self,
        *,
        kind: ExternalToolKind | str,
        result: dict[str, Any],
        is_error: bool = False,
    ) -> str:
        payload = self._base_payload(kind=kind, is_error=is_error)
        payload["result"] = result
        return json.dumps(payload, ensure_ascii=False, default=str)

    def build_error_json(
        self,
        *,
        kind: ExternalToolKind | str,
        code: JsonRpcErrorCode | int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        payload = self._base_payload(kind=kind, is_error=True)
        err: dict[str, Any] = {"code": int(code), "message": message}
        if data:
            err["data"] = data
        payload["error"] = err
        return json.dumps(payload, ensure_ascii=False, default=str)


_default_envelope_builder = ExternalToolEnvelopeBuilder(get_current_tool_invocation)


def build_external_tool_json(
    *,
    kind: ExternalToolKind | str,
    result: dict[str, Any],
    is_error: bool = False,
) -> str:
    return _default_envelope_builder.build_result_json(
        kind=kind,
        result=result,
        is_error=is_error,
    )


def build_external_tool_error_json(
    *,
    kind: ExternalToolKind | str,
    code: JsonRpcErrorCode | int,
    message: str,
    data: dict[str, Any] | None = None,
) -> str:
    return _default_envelope_builder.build_error_json(
        kind=kind,
        code=code,
        message=message,
        data=data,
    )
