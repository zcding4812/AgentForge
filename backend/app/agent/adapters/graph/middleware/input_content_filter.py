"""基于 LangChain 官方 ``AgentMiddleware`` 的末条用户输入内容过滤（``before_agent`` 钩子）。

遵循 LangChain 文档「Custom middleware — Node-style hooks」：
``before_agent`` 在每次 invoke 的 Agent 主循环**开始前**执行一次，适合在模型/工具前做校验与短路；
对需要提前结束的情形配合 ``@hook_config(can_jump_to=[\"end\"])`` 与 ``jump_to: end``。

参考: https://docs.langchain.com/oss/python/langchain/middleware/custom
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import hook_config
from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.spec import InputContentFilterConfig

from .input_filter_evaluate import compile_banned_regexes, input_content_filter_evaluate


class InputContentFilterBeforeAgentMiddleware(AgentMiddleware[LangGraphAgentState, ContextT, Any]):
    """`before_agent`：在 ``create_agent``（如 Simple Chat）主流程前检查末条 human；ReAct 改用 ``ReactInputFilterEntryNode``。"""

    state_schema = LangGraphAgentState

    def __init__(self, config: InputContentFilterConfig) -> None:
        super().__init__()
        self._config = config
        self._regex = compile_banned_regexes(config)

    @staticmethod
    def is_active(c: InputContentFilterConfig) -> bool:
        # 与控制台「启用」一致；无具体规则时 evaluate 会放行
        return bool(c.enabled)

    def apply_to_state(self, state: LangGraphAgentState) -> dict[str, Any] | None:  # noqa: D102
        return input_content_filter_evaluate(state, self._config, self._regex)

    @hook_config(can_jump_to=["end"])
    def before_agent(  # type: ignore[no-untyped-def]  # noqa: D102
        self, state: LangGraphAgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return self.apply_to_state(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(  # type: ignore[no-untyped-def]  # noqa: D102
        self, state: LangGraphAgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return self.apply_to_state(state)
