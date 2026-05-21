"""将 ``ModelConfigSnapshot.tool_choice`` 注入 LangChain Agent 的模型请求。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.typing import ContextT

from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.adapters.tools.tool_choice_arg import openai_tool_choice_arg
from app.agent.kernel.spec import ToolChoicePolicy


class SnapshotToolChoiceMiddleware(AgentMiddleware[LangGraphAgentState, ContextT, Any]):
    """覆盖 ``create_agent`` 内部构造的 ``ModelRequest(tool_choice=None)``。"""

    state_schema = LangGraphAgentState

    def __init__(self, policy: ToolChoicePolicy | None) -> None:
        super().__init__()
        self._policy = policy

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        tc = openai_tool_choice_arg(self._policy)
        if tc is None or not request.tools:
            return handler(request)
        return handler(request.override(tool_choice=tc))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        tc = openai_tool_choice_arg(self._policy)
        if tc is None or not request.tools:
            return await handler(request)
        return await handler(request.override(tool_choice=tc))
