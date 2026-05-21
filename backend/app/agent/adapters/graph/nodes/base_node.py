"""图节点基类：统一 ``__call__`` 前置校验与 ``_process`` 模板。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.spec import ChatModelLike


class BaseGraphNode(ABC):
    __slots__ = ("_chat_model",)

    def __init__(self, chat_model: ChatModelLike) -> None:
        self._chat_model = chat_model

    @abstractmethod
    async def _process(self, state: LangGraphAgentState) -> dict: ...

    async def __call__(self, state: LangGraphAgentState) -> dict:
        if not state.get("messages"):
            raise ValueError("节点执行前 messages 不能为空")
        return await self._process(state)
