from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps
from app.agent.adapters.graph.factory import get_agent_checkpointer
from app.agent.adapters.graph.nodes.plan_execute import ExecuteNode, PlanNode
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.ports import CompiledAgentGraph


class PlanExecuteStrategy:
    """Plan → Execute 两阶段编排。"""

    def compile(self, deps: GraphCompileDeps) -> CompiledAgentGraph:
        plan_node = PlanNode(deps.chat_model, telemetry=deps.telemetry)
        execute_node = ExecuteNode(deps.chat_model, telemetry=deps.telemetry)
        graph = StateGraph(LangGraphAgentState)
        graph.add_node("plan", plan_node)
        graph.add_node("execute", execute_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", END)
        return graph.compile(checkpointer=get_agent_checkpointer())
