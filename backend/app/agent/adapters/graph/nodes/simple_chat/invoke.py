from __future__ import annotations

import time

from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.telemetry.usage import (
    chat_model_label,
    extract_token_usage_from_message,
)
from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike


class ChatModelInvokeNode:
    """单轮：用聊天模型对当前 messages 做一次推理（可单测、可复用）。"""

    __slots__ = ("_model", "_telemetry")

    def __init__(
        self,
        model: ChatModelLike,
        *,
        telemetry: AgentGraphTelemetry | None = None,
    ) -> None:
        self._model = model
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def __call__(self, state: LangGraphAgentState) -> dict:
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.simple_chat",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            reply = await self._model.ainvoke(state["messages"])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(reply)
            tel.tracer.record_llm_call(
                phase="simple_chat",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            return {"messages": [reply]}
