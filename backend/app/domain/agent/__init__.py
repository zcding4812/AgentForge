"""Agent 业务子域：与 ``agent_entity``、编排调用相关的纯规则（无 FastAPI / 无仓储 I/O）。"""

from app.domain.agent.invoke_memory import (
    ConversationInvokeMemoryPort,
    HistoryMergeKind,
    HistoryMergeStrategy,
    MemoryPrepareContext,
    PassthroughHistoryMergeStrategy,
    apply_max_history_rounds_cap,
    resolve_history_merge_kind,
)
from app.domain.agent.knowledge_binding import (
    AgentKnowledgeBindingParser,
    AgentKnowledgeBindingSettings,
    parse_agent_knowledge_binding,
)
from app.domain.agent.memory_settings import (
    DEFAULT_ALLOW_CLIENT_CHAT_HISTORY,
    DEFAULT_COMPRESS_BATCH_ROUNDS,
    DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    DEFAULT_MIN_TAIL_RAW_ROUNDS,
    DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD,
    DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT,
    DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS,
    DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST,
    DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST,
    AgentMemorySettings,
    AgentMemorySettingsParser,
    parse_agent_memory_settings,
)

__all__ = [
    "AgentKnowledgeBindingParser",
    "AgentKnowledgeBindingSettings",
    "DEFAULT_ALLOW_CLIENT_CHAT_HISTORY",
    "DEFAULT_COMPRESS_BATCH_ROUNDS",
    "DEFAULT_MAX_HISTORY_ROUNDS_CAP",
    "DEFAULT_MIN_TAIL_RAW_ROUNDS",
    "DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD",
    "DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT",
    "DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS",
    "DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST",
    "DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST",
    "AgentMemorySettings",
    "AgentMemorySettingsParser",
    "ConversationInvokeMemoryPort",
    "HistoryMergeKind",
    "HistoryMergeStrategy",
    "MemoryPrepareContext",
    "PassthroughHistoryMergeStrategy",
    "apply_max_history_rounds_cap",
    "parse_agent_knowledge_binding",
    "parse_agent_memory_settings",
    "resolve_history_merge_kind",
]
