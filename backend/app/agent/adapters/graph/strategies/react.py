from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps
from app.agent.adapters.graph.factory import get_agent_checkpointer
from app.agent.adapters.graph.nodes.react import ReactInputFilterEntryNode, ReactToolAwareModelNode
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.ports import CompiledAgentGraph


def _route_after_react_input_filter(
    state: LangGraphAgentState,
) -> Literal["__end__", "react_model"]:
    if state.get("react_input_filter_stop"):
        return "__end__"
    return "react_model"


class ReactStrategy:
    """ReAct：显式 ``StateGraph``（输入过滤 → 模型 ↔ ``ToolNode``），与 Workbench 工具环同构。"""

    def compile(self, deps: GraphCompileDeps) -> CompiledAgentGraph:
        tools_list = list(deps.tools)
        graph = StateGraph(LangGraphAgentState)
        graph.add_node("react_input_filter", ReactInputFilterEntryNode(deps.input_content_filter))
        graph.add_node(
            "react_model",
            ReactToolAwareModelNode(
                deps.chat_model,
                deps.tools,
                telemetry=deps.telemetry,
                tool_choice_policy=deps.tool_choice_policy,
            ),
        )
        graph.add_edge(START, "react_input_filter")
        graph.add_conditional_edges(
            "react_input_filter",
            _route_after_react_input_filter,
            {"__end__": END, "react_model": "react_model"},
        )
        if tools_list:
            graph.add_node("react_tools", ToolNode(tools_list))
            graph.add_conditional_edges(
                "react_model",
                tools_condition,
                {"tools": "react_tools", "__end__": END},
            )
            graph.add_edge("react_tools", "react_model")
        else:
            graph.add_edge("react_model", END)
        return graph.compile(checkpointer=get_agent_checkpointer())
