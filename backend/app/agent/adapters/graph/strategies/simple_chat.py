from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps
from app.agent.adapters.graph.factory import get_agent_checkpointer
from app.agent.adapters.graph.middleware.input_content_filter import (
    InputContentFilterBeforeAgentMiddleware,
)
from app.agent.adapters.graph.middleware.snapshot_tool_choice import SnapshotToolChoiceMiddleware
from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.telemetry.usage import chat_model_label
from app.agent.kernel.default_prompts import SIMPLE_CHAT_AGENT_SYSTEM_PROMPT
from app.agent.kernel.ports import AgentGraphTelemetry, CompiledAgentGraph
from app.agent.kernel.spec import ChatModelLike


class _MeteredSimpleChatGraph:
    """与 ReAct 计量包装一致：整段耗时 + LLM 占位；内部步进由 LangChain 执行。"""

    __slots__ = ("_inner", "_telemetry", "_model_label")

    def __init__(
        self,
        inner: CompiledAgentGraph,
        *,
        telemetry: AgentGraphTelemetry,
        chat_model: ChatModelLike,
    ) -> None:
        self._inner = inner
        self._telemetry = telemetry
        self._model_label = chat_model_label(chat_model)

    async def ainvoke(
        self,
        state: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.simple_chat",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            try:
                return await self._inner.ainvoke(state, config=config, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                tel.tracer.record_llm_call(
                    phase="simple_chat",
                    duration_ms=duration_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    model_name=self._model_label,
                    request_id=tel.request_id,
                    trace_id=tel.trace_id,
                    trace_parent_span_id=tel.trace_parent_span_id,
                )

    async def astream(
        self,
        state: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.simple_chat",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            try:
                async for chunk in self._inner.astream(state, config=config, **kwargs):
                    yield chunk
            finally:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                tel.tracer.record_llm_call(
                    phase="simple_chat",
                    duration_ms=duration_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    model_name=self._model_label,
                    request_id=tel.request_id,
                    trace_id=tel.trace_id,
                    trace_parent_span_id=tel.trace_parent_span_id,
                )


class SimpleChatStrategy:
    """Simple Chat：与 ReAct 同用 ``create_agent``；``tools`` 来自 ``GraphCompileDeps``（通常为空）。"""

    def compile(self, deps: GraphCompileDeps) -> CompiledAgentGraph:
        mlist: list[AgentMiddleware[Any, Any, Any]] = []
        if InputContentFilterBeforeAgentMiddleware.is_active(deps.input_content_filter):
            mlist.append(InputContentFilterBeforeAgentMiddleware(deps.input_content_filter))
        if deps.tool_choice_policy is not None:
            mlist.append(SnapshotToolChoiceMiddleware(deps.tool_choice_policy))
        g = create_agent(
            deps.chat_model,
            list(deps.tools),
            system_prompt=SIMPLE_CHAT_AGENT_SYSTEM_PROMPT,
            state_schema=LangGraphAgentState,
            checkpointer=get_agent_checkpointer(),
            middleware=tuple(mlist) if mlist else (),
        )
        return _MeteredSimpleChatGraph(
            g,
            telemetry=deps.telemetry,
            chat_model=deps.chat_model,
        )
