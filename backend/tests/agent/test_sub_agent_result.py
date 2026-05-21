from app.agent.adapters.tools.workbench.sub_agent_invoke import SubAgentPayloadCompactor


def test_compact_truncates_long_assistant_text() -> None:
    long_text = "x" * 50_000
    out = SubAgentPayloadCompactor.compact(
        assistant_text=long_text,
        structured=None,
        tokens=None,
        sub_agent_id=1,
        sub_agent_kind="react",
        max_chars=1000,
    )
    assert len(out["assistant_text"]) < len(long_text)
    assert out.get("result_truncated") is True


def test_compact_omits_huge_structured() -> None:
    huge = {"a": "z" * 30_000}
    out = SubAgentPayloadCompactor.compact(
        assistant_text="ok",
        structured=huge,
        tokens=1,
        sub_agent_id=2,
        sub_agent_kind="simple_chat",
        max_chars=8000,
    )
    assert out["structured"] is None
    assert out.get("structured_omitted") is True


def test_compact_attaches_nested_tool_history() -> None:
    out = SubAgentPayloadCompactor.compact(
        assistant_text="ok",
        structured=None,
        tokens=None,
        sub_agent_id=3,
        sub_agent_kind="react",
        sub_tool_history=[
            {"phase": "call", "name": "child_tool", "arguments": {"a": 1}, "tool_call_id": "t1"},
            {"phase": "result", "name": "child_tool", "content": "{}", "tool_call_id": "t1"},
        ],
        sub_process_trace=[{"seq": 0, "text": "子步", "phase": "step"}],
    )
    assert out.get("sub_tool_history") is not None
    assert len(out["sub_tool_history"]) == 2
    assert out.get("sub_process_trace") is not None
