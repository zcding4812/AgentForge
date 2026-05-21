"""MCP 传输统一抽象：所有传输（HTTP / WebSocket / gRPC）实现同一接口，供工具层调度与会话管理复用。

上层（LangChain 工具、错误信封）只依赖 :class:`McpTransport`，不依赖具体传输实现。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class McpProtocolError(RuntimeError):
    """MCP JSON-RPC 语义错误（id 不匹配、error 对象等）。"""


class McpTransportError(RuntimeError):
    """HTTP 状态、网络或无法解析的传输错误。"""


class McpSessionExpiredError(McpTransportError):
    """服务端返回 404，会话失效，需重新 initialize。"""


@runtime_checkable
class McpTransport(Protocol):
    """MCP 客户端传输抽象：会话建立、JSON-RPC、能力发现、工具调用。"""

    async def aclose(self) -> None:
        """释放连接与缓存。"""

    async def initialize(self, *, platform_session_id: str | None = None) -> None:
        """执行 MCP 握手（``initialize`` / ``notifications/initialized`` 等），建立服务端会话。"""

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        platform_session_id: str | None = None,
    ) -> Any:
        """发送 JSON-RPC 请求（在已初始化会话上）；``method`` 为 MCP 方法名（如 ``resources/list``）。"""

    async def list_tools(
        self,
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """``tools/list``。"""

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> Any:
        """``tools/call``。"""

    def has_completed_list_tools(self, platform_session_id: str | None) -> bool:
        """调度模式：本会话是否已成功执行过 ``tools/list``。"""


class McpNotImplementedTransport:
    """WebSocket / gRPC 等尚未接入时的占位传输：统一接口，调用时抛出明确错误。"""

    __slots__ = ("_transport_type", "_server_url")

    def __init__(self, *, transport_type: str, server_url: str | None = None) -> None:
        self._transport_type = (transport_type or "").strip().lower() or "unknown"
        self._server_url = server_url

    def _msg(self, op: str) -> str:
        u = f" server_url={self._server_url!r}" if self._server_url else ""
        return (
            f"MCP 传输类型 {self._transport_type!r} 的 {op} 尚未实现；"
            f"当前已接入 transport_type=http（Streamable HTTP）与 sse（HTTP+SSE）。{u}"
        )

    async def aclose(self) -> None:
        return None

    async def initialize(self, *, platform_session_id: str | None = None) -> None:
        raise McpTransportError(self._msg("initialize"))

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        platform_session_id: str | None = None,
    ) -> Any:
        raise McpTransportError(self._msg(f"send_request({method!r})"))

    async def list_tools(
        self,
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        raise McpTransportError(self._msg("list_tools"))

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        platform_session_id: str | None = None,
        logical_tool_name: str | None = None,
    ) -> Any:
        raise McpTransportError(self._msg("call_tool"))

    def has_completed_list_tools(self, platform_session_id: str | None) -> bool:
        return False
