"""ReAct 图节点：绑定工具 + ReAct 系统提示，等价于原 ``create_agent`` 内模型步。"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.telemetry.usage import (
    chat_model_label,
    extract_token_usage_from_message,
)
from app.agent.adapters.tools.tool_choice_arg import openai_tool_choice_arg
from app.agent.kernel.default_prompts import REACT_AGENT_TOOL_SYSTEM_PROMPT
from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike, ToolChoicePolicy


class ReactToolAwareModelNode:
    __slots__ = ("_chat_model", "_bound_model", "_telemetry")

    def __init__(
        self,
        chat_model: ChatModelLike,
        tools: tuple[BaseTool | dict, ...],
        *,
        telemetry: AgentGraphTelemetry | None = None,
        tool_choice_policy: ToolChoicePolicy | None = None,
    ) -> None:
        self._chat_model = chat_model
        tc = openai_tool_choice_arg(tool_choice_policy)
        tool_list = list(tools)
        if tc is not None and tool_list:
            self._bound_model = chat_model.bind_tools(tool_list, tool_choice=tc)
        else:
            self._bound_model = chat_model.bind_tools(tool_list)
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            raise ValueError("ReAct 模型节点执行前 messages 不能为空")
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.react_agent",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            prompt: list[BaseMessage] = [
                SystemMessage(content=REACT_AGENT_TOOL_SYSTEM_PROMPT),
                *messages,
            ]
            reply = await self._bound_model.ainvoke(prompt)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(reply)
            tel.tracer.record_llm_call(
                phase="react_agent",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._chat_model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            return {"messages": [reply]}
