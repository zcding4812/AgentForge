from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.nodes.base_node import BaseGraphNode
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.telemetry.usage import (
    chat_model_label,
    extract_token_usage_from_message,
)
from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike


class ExecuteNode(BaseGraphNode):
    """PlanExecute：执行节点。"""

    __slots__ = ("_telemetry",)

    def __init__(
        self,
        chat_model: ChatModelLike,
        *,
        telemetry: AgentGraphTelemetry | None = None,
    ) -> None:
        super().__init__(chat_model)
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def _process(self, state: LangGraphAgentState) -> dict:
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.execute",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            plan = (state.get("plan") or "").strip()
            last_human = ""
            for m in state["messages"]:
                if isinstance(m, HumanMessage):
                    last_human = m.content if isinstance(m.content, str) else str(m.content)
            plan_block = (
                plan if plan else "（规划阶段未产生有效计划，请仅根据下方用户问题直接作答。）"
            )
            exec_system = (
                "你是执行器（Executor）。请严格按「计划要点」的顺序组织回答：逐条覆盖、合并冗余，"
                "给出完整、可直接采用的最终答案。\n"
                "要求：\n"
                "1. 不要逐字复述计划全文；用自然段落或列表呈现结果即可。\n"
                "2. 若计划与问题冲突，以用户问题为准并简要说明取舍。\n"
                "3. 语言与用户问题一致；需要代码/公式时用 Markdown fenced code 即可，勿整篇用代码块包裹。\n"
                "4. 若某步无法完成，说明原因并给出最佳替代。\n\n"
                "【计划要点】\n"
                f"{plan_block}"
            )
            exec_messages = [
                SystemMessage(content=exec_system),
                HumanMessage(content=last_human),
            ]
            t0 = time.perf_counter()
            final = await self._chat_model.ainvoke(exec_messages)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(final)
            tel.tracer.record_llm_call(
                phase="plan_execute.execute",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._chat_model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            return {"messages": [final]}
