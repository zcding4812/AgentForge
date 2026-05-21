from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike, InputContentFilterConfig, ToolChoicePolicy


@dataclass(frozen=True, slots=True)
class GraphCompileDeps:
    """策略 `compile` 的注入依赖（适配器层；不进入 kernel）。"""

    chat_model: ChatModelLike
    tools: tuple[BaseTool | dict, ...] = ()
    telemetry: AgentGraphTelemetry = field(default_factory=AgentGraphTelemetry)
    tool_choice_policy: ToolChoicePolicy | None = None
    input_content_filter: InputContentFilterConfig = field(
        default_factory=InputContentFilterConfig,
    )
