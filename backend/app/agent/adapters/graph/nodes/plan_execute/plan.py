from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage

from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.nodes.base_node import BaseGraphNode
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.telemetry.usage import (
    chat_model_label,
    extract_token_usage_from_message,
)
from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike


@dataclass(frozen=True, slots=True)
class PlanPromptConfig:
    system_prompt: str = (
        "你是任务规划器（Planner）。请阅读对话中的系统说明、上下文与历史（若有），理解用户真实意图。\n"
        "输出要求：\n"
        "1. 仅输出不超过 3 条的执行要点，每条一行；可用「1. 2. 3.」编号，不要标题或前缀寒暄。\n"
        "2. 使用与用户问题一致的语言（用户中文则用中文要点）。\n"
        "3. 要点应可执行、可检验：说明要做什么、依据什么信息、交付什么结果；避免空泛口号。\n"
        "4. 若信息不足，在要点中写明需要先向用户澄清的一点，并给出基于当前信息的暂定步骤。\n"
        "5. 纯文本，不要使用 Markdown 代码块或整段引用。"
    )
    max_plan_items: int = 3


class PlanNode(BaseGraphNode):
    """PlanExecute：规划节点。"""

    __slots__ = ("_prompt_config", "_telemetry")

    def __init__(
        self,
        chat_model: ChatModelLike,
        prompt_config: PlanPromptConfig | None = None,
        *,
        telemetry: AgentGraphTelemetry | None = None,
    ) -> None:
        super().__init__(chat_model)
        self._prompt_config = prompt_config or PlanPromptConfig()
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def _process(self, state: LangGraphAgentState) -> dict:
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.plan",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            plan_prompt = [
                SystemMessage(content=self._prompt_config.system_prompt),
                *state["messages"],
            ]
            t0 = time.perf_counter()
            plan_message: BaseMessage = await self._chat_model.ainvoke(plan_prompt)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(plan_message)
            tel.tracer.record_llm_call(
                phase="plan_execute.plan",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._chat_model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            plan_text = (
                plan_message.content
                if isinstance(plan_message.content, str)
                else str(plan_message.content)
            )
            return {"messages": [plan_message], "plan": plan_text}
