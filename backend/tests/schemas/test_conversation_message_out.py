"""``ConversationMessageOut`` 与 ORM ``message_metadata`` 对齐。"""

from datetime import datetime

from app.schemas.conversation import ConversationMessageExecutionOut, ConversationMessageOut


def test_model_validate_maps_message_metadata_to_metadata() -> None:
    class Row:
        id = 1
        session_id = "abc"
        agent_id = 9
        role = "user"
        content = "hi"
        content_type = "text"
        created_at = datetime(2020, 1, 1, 0, 0, 0)
        message_metadata = {"k": 1}
        turn_index = 1
        reply_message_id = None
        tokens = None

    out = ConversationMessageOut.model_validate(Row())
    assert out.agent_id == 9
    assert out.metadata == {"k": 1}


def test_conversation_message_execution_out_roundtrip() -> None:
    ex = ConversationMessageExecutionOut(
        message_id=42,
        session_id="s1",
        agent_id=7,
        role="assistant",
        process_trace=[{"seq": 0, "text": "x", "phase": "final", "kind": "model"}],
        tool_history=[{"seq": 0, "phase": "call", "name": "t"}],
        thinking_text="…",
    )
    d = ex.model_dump(mode="json")
    assert d["message_id"] == 42
    assert len(d["process_trace"]) == 1
