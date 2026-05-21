from app.agent.kernel.spec import ToolCallSlot, ToolResultSlot
from app.domain.conversation import AssistantToolMetadataParser


def test_tool_calls_empty_meta() -> None:
    p = AssistantToolMetadataParser()
    assert p.extract_tool_calls(None) == ()
    assert p.extract_tool_calls({}) == ()


def test_tool_calls_skips_invalid_rows() -> None:
    p = AssistantToolMetadataParser()
    out = p.extract_tool_calls(
        {"tool_calls": [{"id": "", "name": "x"}, {"id": "1", "name": "t", "arguments": "{}"}]},
    )
    assert out == (ToolCallSlot(id="1", name="t", arguments="{}"),)


def test_tool_results_roundtrip() -> None:
    p = AssistantToolMetadataParser()
    tr = p.extract_tool_results(
        {
            "tool_results": [
                {"tool_call_id": "c1", "name": "n", "content": "body"},
            ],
        },
    )
    assert tr == (ToolResultSlot(tool_call_id="c1", name="n", content="body"),)
