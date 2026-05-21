from types import SimpleNamespace

from app.agent.kernel.spec import HistoryTurnSlot, ToolCallSlot, ToolResultSlot
from app.domain.conversation import (
    conversation_messages_to_history_turns,
    dedupe_last_turn_if_same_as_current,
)
from app.models.conversation_mod import AgentConversationMessage


def _msg(
    *,
    role: str,
    content: str,
    metadata: dict | None = None,
    id_: int = 1,
) -> AgentConversationMessage:
    return AgentConversationMessage(
        id=id_,
        session_id="s1",
        agent_id=1,
        role=role,
        content=content,
        content_type="text",
        message_metadata=metadata,
        turn_index=0,
    )


def test_protocol_accepts_simple_namespace() -> None:
    """领域函数只依赖 ``ConversationMessageLike``，不必使用 ORM 实体。"""
    rows = [
        SimpleNamespace(role="user", content="hi", message_metadata=None),
        SimpleNamespace(role="assistant", content="yo", message_metadata=None),
    ]
    turns = conversation_messages_to_history_turns(rows)
    assert len(turns) == 1
    assert turns[0] == HistoryTurnSlot(user="hi", assistant="yo")


def test_pair_user_assistant() -> None:
    rows = [
        _msg(role="user", content="hi", id_=1),
        _msg(role="assistant", content="hello", id_=2),
    ]
    turns = conversation_messages_to_history_turns(rows)
    assert len(turns) == 1
    assert turns[0] == HistoryTurnSlot(user="hi", assistant="hello")


def test_drop_trailing_orphan_user() -> None:
    rows = [
        _msg(role="user", content="hi", id_=1),
        _msg(role="assistant", content="hello", id_=2),
        _msg(role="user", content="pending", id_=3),
    ]
    turns = conversation_messages_to_history_turns(rows)
    assert len(turns) == 1
    assert turns[0].user == "hi"


def test_drop_middle_orphan_user() -> None:
    rows = [
        _msg(role="user", content="a", id_=1),
        _msg(role="user", content="b", id_=2),
        _msg(role="assistant", content="rep", id_=3),
    ]
    turns = conversation_messages_to_history_turns(rows)
    assert len(turns) == 1
    assert turns[0].user == "b"
    assert turns[0].assistant == "rep"


def test_assistant_tool_metadata() -> None:
    rows = [
        _msg(role="user", content="q", id_=1),
        _msg(
            role="assistant",
            content="",
            id_=2,
            metadata={
                "tool_calls": [{"id": "c1", "name": "search", "arguments": "{}"}],
                "tool_results": [{"tool_call_id": "c1", "name": "search", "content": "ok"}],
            },
        ),
    ]
    turns = conversation_messages_to_history_turns(rows)
    assert len(turns) == 1
    t = turns[0]
    assert t.user == "q"
    assert t.assistant == ""
    assert t.tool_calls == (ToolCallSlot(id="c1", name="search", arguments="{}"),)
    assert t.tool_results == (ToolResultSlot(tool_call_id="c1", name="search", content="ok"),)


def test_dedupe_last_turn_same_as_current() -> None:
    turns = [
        HistoryTurnSlot(user="same", assistant="a"),
        HistoryTurnSlot(user="x", assistant="y"),
    ]
    out = dedupe_last_turn_if_same_as_current(turns, "x")
    assert len(out) == 1
    assert out[0].user == "same"


def test_dedupe_noop() -> None:
    turns = [HistoryTurnSlot(user="a", assistant="b")]
    out = dedupe_last_turn_if_same_as_current(turns, "different")
    assert len(out) == 1
