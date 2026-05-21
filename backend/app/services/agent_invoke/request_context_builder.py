"""与单次调用相关的输出控制、会话 id 解析、落库 metadata。"""

from __future__ import annotations

from typing import Any

from app.agent.kernel import OutputControl
from app.schemas.agent import AgentInvokeRequest


class InvokeRequestContextBuilder:
    """HTTP 请求体上的输出选项与对话落库元数据（无状态，纯静态逻辑）。"""

    @staticmethod
    def output_control(body: AgentInvokeRequest) -> OutputControl:
        o = body.output
        return OutputControl(
            strip_thinking_blocks=o.strip_thinking_blocks,
            include_tool_messages_in_raw=o.include_tool_messages_in_raw,
        )

    @staticmethod
    def conversation_session_id(body: AgentInvokeRequest) -> str:
        return body.conversation_session_id or ""

    @staticmethod
    def conversation_metadata(
        body: AgentInvokeRequest, *, extra: dict | None = None
    ) -> dict[str, Any]:
        m: dict[str, Any] = {"agent_kind": body.agent_kind.value}
        if body.agent_id is not None:
            m["agent_id"] = body.agent_id
        if body.config_id is not None:
            m["config_id"] = body.config_id
        udn = (body.conversation_user_display_name or "").strip()
        if udn:
            m["user_display_name"] = udn[:64]
        if extra:
            m.update(extra)
        return m
