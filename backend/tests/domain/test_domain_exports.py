"""根包 ``app.domain`` 再导出与文档约定一致。"""

from app.domain import (
    AgentMemorySettings,
    AgentMemorySettingsParser,
    AssistantToolMetadataParser,
    ConversationHistoryCoordinator,
    ConversationMessageLike,
    HistoryTurnAssembler,
    HistoryTurnDeduper,
    conversation_messages_to_history_turns,
    dedupe_last_turn_if_same_as_current,
    history_turn_slots_from_json_str,
    history_turn_slots_to_json_str,
    parse_agent_memory_settings,
)


def test_root_imports() -> None:
    assert callable(parse_agent_memory_settings)
    assert callable(conversation_messages_to_history_turns)
    assert callable(dedupe_last_turn_if_same_as_current)
    assert callable(history_turn_slots_from_json_str)
    assert callable(history_turn_slots_to_json_str)
    assert AgentMemorySettings.__name__ == "AgentMemorySettings"
    assert ConversationMessageLike.__name__ == "ConversationMessageLike"
    assert AssistantToolMetadataParser.__name__ == "AssistantToolMetadataParser"
    assert HistoryTurnAssembler.__name__ == "HistoryTurnAssembler"
    assert HistoryTurnDeduper.__name__ == "HistoryTurnDeduper"
    assert ConversationHistoryCoordinator.__name__ == "ConversationHistoryCoordinator"
    assert AgentMemorySettingsParser.__name__ == "AgentMemorySettingsParser"
