from app.agent.kernel.spec import HistoryTurnSlot, ToolCallSlot, ToolResultSlot
from app.domain.conversation import (
    history_turn_slots_from_json_str,
    history_turn_slots_to_json_str,
)


def test_roundtrip_plain_turn() -> None:
    turns = [HistoryTurnSlot(user="u", assistant="a")]
    s = history_turn_slots_to_json_str(turns)
    out = history_turn_slots_from_json_str(s)
    assert out == turns


def test_roundtrip_with_tools() -> None:
    turns = [
        HistoryTurnSlot(
            user="q",
            assistant="",
            tool_calls=(ToolCallSlot(id="c1", name="search", arguments="{}"),),
            tool_results=(ToolResultSlot(tool_call_id="c1", name="search", content="ok"),),
        ),
    ]
    s = history_turn_slots_to_json_str(turns)
    out = history_turn_slots_from_json_str(s)
    assert out == turns
