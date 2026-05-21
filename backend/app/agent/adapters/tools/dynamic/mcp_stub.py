"""MCP 工具占位：配置已持久化、真实 MCP 会话接入前用于注册与联调。

**契约**

- **输入**：无参数（空 schema）；执行不发起真实 MCP 网络调用。
- **输出**：``build_external_tool_json``（``kind`` 为 ``mcp``）与扁平 ``result``，含
  ``tool_name``、``status``、``message``、``mcp_transport_type``、``mcp_server_url``、
  ``config_keys``、``hint``；保留 ``ok: false`` 便于旧调用方识别。
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, TypedDict

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.agent.kernel.tool_runtime import build_external_tool_json

logger = logging.getLogger(__name__)

_MCP_PLACEHOLDER_STATUS = "placeholder"


class McpPlaceholderArgs(BaseModel):
    """MCP 占位工具：模型侧无调用参数。"""

    model_config = ConfigDict(extra="forbid")


class McpPlaceholderConfig(BaseModel):
    """与 ``RuntimeMcpTool`` / ``_mcp_merged_config`` 对齐的占位配置。"""

    logical_name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=512)
    mcp_config: dict[str, Any] = Field(
        default_factory=dict,
        description="合并后的 MCP 配置（含 server_url、transport_type 等）",
    )


class McpPlaceholderData(TypedDict):
    """``result`` 内业务负载（用于类型检查，运行时为普通 dict）。"""

    tool_name: str
    status: str
    message: str
    ok: bool
    mcp_transport_type: str | None
    mcp_server_url: str | None
    config_keys: list[str]
    hint: str


class McpPlaceholderTool(BaseTool):
    """MCP 占位：注册与联调用；无副作用、同步/异步均可调用。"""

    args_schema: ClassVar[type[BaseModel]] = McpPlaceholderArgs
    _cfg: McpPlaceholderConfig = PrivateAttr()

    def __init__(self, cfg: McpPlaceholderConfig) -> None:
        super().__init__(name=cfg.logical_name, description=cfg.description)
        self._cfg = cfg
        mc = cfg.mcp_config
        logger.info(
            "mcp_placeholder.tool_created",
            extra={
                "tool": cfg.logical_name,
                "transport": mc.get("transport_type"),
                "server_url": mc.get("server_url"),
            },
        )

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._wrap_result(self._get_placeholder_data())

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return self._wrap_result(self._get_placeholder_data())

    def _get_placeholder_data(self) -> McpPlaceholderData:
        cfg = self._cfg
        mc = cfg.mcp_config
        config_keys = list(mc.keys()) if mc else []
        transport = mc.get("transport_type")
        transport_s = str(transport) if transport is not None else "http"
        raw_url = mc.get("server_url")
        server_url: str | None = str(raw_url).strip() if raw_url not in (None, "") else None

        if server_url:
            hint = (
                f"请检查 MCP 客户端适配器是否已接入并连接 {server_url}。"
                f"当前传输类型: {transport_s}。"
            )
        else:
            hint = (
                "请在库表 RuntimeMcpTool 中配置 server_url、transport_type，"
                "并确保 MCP 客户端适配器已初始化。"
            )

        return McpPlaceholderData(
            tool_name=cfg.logical_name,
            status=_MCP_PLACEHOLDER_STATUS,
            message="MCP 工具配置已持久化，进程内执行尚未接入 MCP 客户端；请后续接入适配器。",
            ok=False,
            mcp_transport_type=transport_s,
            mcp_server_url=server_url,
            config_keys=config_keys,
            hint=hint,
        )

    def _wrap_result(self, data: McpPlaceholderData) -> str:
        # ``ExternalToolKind`` 信封仅允许 "http" | "mcp"，勿使用未收录的 kind
        payload: dict[str, Any] = dict(data)
        return build_external_tool_json(kind="mcp", result=payload, is_error=False)


def make_mcp_placeholder_tool(
    *,
    logical_name: str,
    description: str,
    mcp_config: dict[str, Any] | None,
) -> BaseTool:
    """由持久化行构造 MCP 占位工具（与 :func:`_register_in_memory` 调用签名一致）。"""
    cfg = McpPlaceholderConfig(
        logical_name=logical_name.strip(),
        description=description.strip(),
        mcp_config=dict(mcp_config) if mcp_config else {},
    )
    return McpPlaceholderTool(cfg=cfg)
