"""工作台编排：显式 LangGraph（首轮 plan → react_loop ↔ tools → **replan** 循环 → fan-in → aggregate）。

每轮工具返回后先经 ``workbench_replan`` 按最新事实刷新 ``state.plan``，再进入下一轮 ReAct；
子任务由 ``workbench_invoke_sub_agent`` 等驱动。与 ``PlanExecuteStrategy`` 的两段式不同。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps
from app.agent.adapters.graph.factory import get_agent_checkpointer
from app.agent.adapters.graph.nodes.workbench import (
    WorkbenchAggregateNode,
    WorkbenchFanInNode,
    WorkbenchPlanNode,
    WorkbenchReplanNode,
    WorkbenchToolAwareAgentNode,
)
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.ports import CompiledAgentGraph


class WorkbenchStrategy:
    """显式 StateGraph：首轮计划 + ReAct↔工具 + 每轮工具后修订计划 + fan-in + 汇总。"""

    def compile(self, deps: GraphCompileDeps) -> CompiledAgentGraph:
        graph = StateGraph(LangGraphAgentState)
        graph.add_node(
            "workbench_plan",
            WorkbenchPlanNode(deps.chat_model, telemetry=deps.telemetry),
        )
        graph.add_node(
            "workbench_replan",
            WorkbenchReplanNode(deps.chat_model, telemetry=deps.telemetry),
        )
        graph.add_node(
            "workbench_react_loop",
            WorkbenchToolAwareAgentNode(
                deps.chat_model,
                deps.tools,
                telemetry=deps.telemetry,
                tool_choice_policy=deps.tool_choice_policy,
            ),
        )
        graph.add_node("workbench_tools", ToolNode(list(deps.tools)))
        graph.add_node(
            "workbench_aggregate",
            WorkbenchAggregateNode(deps.chat_model, telemetry=deps.telemetry),
        )
        graph.add_node("workbench_fan_in", WorkbenchFanInNode())
        graph.add_edge(START, "workbench_plan")
        graph.add_edge("workbench_plan", "workbench_react_loop")
        graph.add_conditional_edges(
            "workbench_react_loop",
            tools_condition,
            {
                "tools": "workbench_tools",
                "__end__": "workbench_fan_in",
            },
        )
        graph.add_edge("workbench_tools", "workbench_replan")
        graph.add_edge("workbench_replan", "workbench_react_loop")
        graph.add_edge("workbench_fan_in", "workbench_aggregate")
        graph.add_edge("workbench_aggregate", END)
        return graph.compile(checkpointer=get_agent_checkpointer())
