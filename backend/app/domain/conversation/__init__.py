"""对话子域：持久化消息语义 → 内核槽位、分页与滚动摘要触发等。

实现仍分模块文件（``history``、``pagination`` …），此处统一 **对外再导出**，便于
``from app.domain.conversation import ...`` 单点入口。
"""

from collections.abc import Sequence

from app.agent.kernel.spec import HistoryTurnSlot
from app.domain.conversation.content_tokens import estimate_user_content_tokens
from app.domain.conversation.exceptions import (
    ConversationDomainError,
    ReferencedAgentNotFoundError,
    SessionAgentMismatchError,
    SessionNotFoundError,
    SessionNotWritableError,
)
from app.domain.conversation.history import (
    AssistantToolMetadataParser,
    ConversationHistoryCoordinator,
    ConversationMessageLike,
    HistoryTurnAssembler,
    HistoryTurnDeduper,
    history_turn_slots_from_json_str,
    history_turn_slots_to_json_str,
)
from app.domain.conversation.pagination import (
    normalize_conversation_list_pagination,
    normalize_conversation_messages_pagination,
)
from app.domain.conversation.rolling_summary_trigger import (
    RollingSummaryTriggerMetrics,
    decide_rolling_summary_trigger,
)


def conversation_messages_to_history_turns(
    messages: Sequence[ConversationMessageLike],
) -> list[HistoryTurnSlot]:
    """兼容旧调用点；等价于 ``HistoryTurnAssembler().assemble(messages)``。"""
    return HistoryTurnAssembler().assemble(messages)


def dedupe_last_turn_if_same_as_current(
    turns: list[HistoryTurnSlot],
    user_message: str,
) -> list[HistoryTurnSlot]:
    """兼容旧调用点；等价于 ``HistoryTurnDeduper().dedupe_last_if_same_as_current(...)``。"""
    return HistoryTurnDeduper().dedupe_last_if_same_as_current(turns, user_message)


__all__ = [
    "AssistantToolMetadataParser",
    "ConversationDomainError",
    "ConversationHistoryCoordinator",
    "ConversationMessageLike",
    "HistoryTurnAssembler",
    "HistoryTurnDeduper",
    "ReferencedAgentNotFoundError",
    "RollingSummaryTriggerMetrics",
    "SessionAgentMismatchError",
    "SessionNotFoundError",
    "SessionNotWritableError",
    "conversation_messages_to_history_turns",
    "dedupe_last_turn_if_same_as_current",
    "decide_rolling_summary_trigger",
    "estimate_user_content_tokens",
    "history_turn_slots_from_json_str",
    "history_turn_slots_to_json_str",
    "normalize_conversation_list_pagination",
    "normalize_conversation_messages_pagination",
]
