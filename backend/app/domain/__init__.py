"""
领域层（``app.domain``）
======================

**职责**：与 HTTP、数据库会话、外部 SDK **无关** 的业务规则、不变量与纯数据转换。
**调用方**：主要为 ``app.services.*``；由服务层编排仓储后调用领域 **类实例方法**（或兼容用模块级函数）。

**结构**（按子域分包）::

    domain/
        agent/              # Agent 资源、记忆参数、``invoke_memory.py``（invoke 前记忆段）
        conversation/       # 会话消息 / 分页 / 滚动摘要触发（``__init__`` 统一再导出）

**依赖约束**（与仓库 ``AGENTS.md`` / fastapi-framework §4.2 一致）：

- **禁止** 依赖 ``fastapi``、``starlette`` 请求对象。
- **避免** 依赖 ``app.repositories``；持久化查询留在服务层与仓储。
- 允许依赖 ``app.agent.kernel``（值对象 / 槽位类型）、``app.models``（仅作 **行数据形状** 输入时；后续可收紧为 Protocol）。

推荐类入口::

    from app.domain.conversation import ConversationHistoryCoordinator, HistoryTurnAssembler
    from app.domain.agent import AgentMemorySettingsParser, AgentMemorySettings

设计说明见 ``backend/docs/domain-layer.md``。
"""

from app.domain.agent import (
    DEFAULT_ALLOW_CLIENT_CHAT_HISTORY,
    DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    AgentMemorySettings,
    AgentMemorySettingsParser,
    parse_agent_memory_settings,
)
from app.domain.conversation import (
    AssistantToolMetadataParser,
    ConversationHistoryCoordinator,
    ConversationMessageLike,
    HistoryTurnAssembler,
    HistoryTurnDeduper,
    conversation_messages_to_history_turns,
    dedupe_last_turn_if_same_as_current,
    history_turn_slots_from_json_str,
    history_turn_slots_to_json_str,
)

__all__ = [
    "DEFAULT_ALLOW_CLIENT_CHAT_HISTORY",
    "DEFAULT_MAX_HISTORY_ROUNDS_CAP",
    "AgentMemorySettings",
    "AgentMemorySettingsParser",
    "parse_agent_memory_settings",
    "AssistantToolMetadataParser",
    "ConversationHistoryCoordinator",
    "ConversationMessageLike",
    "HistoryTurnAssembler",
    "HistoryTurnDeduper",
    "conversation_messages_to_history_turns",
    "dedupe_last_turn_if_same_as_current",
    "history_turn_slots_from_json_str",
    "history_turn_slots_to_json_str",
]
