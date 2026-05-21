"""ReAct 图首节点：等价于原 ``InputContentFilterBeforeAgentMiddleware.before_agent``。"""

from __future__ import annotations

from typing import Any

from app.agent.adapters.graph.middleware.input_content_filter import (
    InputContentFilterBeforeAgentMiddleware,
)
from app.agent.adapters.graph.middleware.input_filter_evaluate import (
    compile_banned_regexes,
    input_content_filter_evaluate,
)
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.spec import InputContentFilterConfig


class ReactInputFilterEntryNode:
    """写入 ``react_input_filter_stop`` 供条件边短路到 ``END``；否则继续 ``react_model``。"""

    __slots__ = ("_config", "_regex", "_active")

    def __init__(self, config: InputContentFilterConfig) -> None:
        self._config = config
        self._active = InputContentFilterBeforeAgentMiddleware.is_active(config)
        self._regex = compile_banned_regexes(config) if self._active else []

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        cleared: dict[str, Any] = {"react_input_filter_stop": False}
        if not self._active:
            return cleared
        out = input_content_filter_evaluate(state, self._config, self._regex)
        if out is None:
            return cleared
        msgs = out.get("messages") or []
        if out.get("jump_to") == "end":
            return {"messages": list(msgs), "react_input_filter_stop": True}
        return {"messages": list(msgs), "react_input_filter_stop": False}
