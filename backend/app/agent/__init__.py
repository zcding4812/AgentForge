"""Agent 模块：能力内核（kernel）/ 用例（application）/ 出站适配器（adapters）。

包根 ``__all__`` 为对外常用符号；子模块不再各自定义 ``__all__``，请按模块路径显式导入。
"""

# 适配器工厂（对外扩展入口）
from app.agent.adapters.graph import LangGraphAgentFactory
from app.agent.adapters.models import DefaultChatModelFactory

# 应用层（对外核心入口）
from app.agent.application.facade import AgentExecutionError, AgentRunFacade

# 内核核心类型（上层常用，无需导入深层路径）
from app.agent.kernel import (
    AgentKind,
    FinalAgentOutput,
    ModelConfigSnapshot,
    OutputControl,
    PromptSlots,
)

__all__ = [
    # 应用层
    "AgentExecutionError",
    "AgentRunFacade",
    # 内核核心类型
    "AgentKind",
    "ModelConfigSnapshot",
    "FinalAgentOutput",
    "OutputControl",
    "PromptSlots",
    # 适配器工厂
    "DefaultChatModelFactory",
    "LangGraphAgentFactory",
]
