from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.adapters.graph.nodes.workbench.react_loop import (
    _collect_fan_in_summary,
    _expand_parallel_tool_calls,
)


def test_expand_parallel_tool_call_into_sub_agent_calls() -> None:
    msg = AIMessage(
        content="ok",
        tool_calls=[
            {
                "name": "workbench_invoke_sub_agents_parallel",
                "id": "call_parallel_1",
                "type": "tool_call",
                "args": {
                    "tasks_json": [
                        {"agent_id": 11, "user_message": "任务A"},
                        {"agent_id": 12, "user_message": "任务B", "parent_messages": "[]"},
                    ]
                },
            }
        ],
    )
    out = _expand_parallel_tool_calls(msg)
    assert len(out.tool_calls) == 2
    assert out.tool_calls[0]["name"] == "workbench_invoke_sub_agent"
    assert out.tool_calls[0]["args"] == {"agent_id": 11, "user_message": "任务A"}
    assert out.tool_calls[1]["name"] == "workbench_invoke_sub_agent"
    assert out.tool_calls[1]["args"] == {
        "agent_id": 12,
        "user_message": "任务B",
        "parent_messages": "[]",
    }


def test_invalid_parallel_args_fallback_to_original_tool() -> None:
    msg = AIMessage(
        content="ok",
        tool_calls=[
            {
                "name": "workbench_invoke_sub_agents_parallel",
                "id": "call_parallel_bad",
                "type": "tool_call",
                "args": {"tasks_json": "{invalid-json"},
            }
        ],
    )
    out = _expand_parallel_tool_calls(msg)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "workbench_invoke_sub_agents_parallel"


def test_collect_fan_in_summary_from_sub_agent_tool_messages() -> None:
    msgs = [
        ToolMessage(
            content='{"code":"OK","message":"sub_agent_completed","data":{"sub_agent_id":21}}',
            tool_call_id="tc1",
            name="workbench_invoke_sub_agent",
        ),
        ToolMessage(
            content=(
                '{"code":"OK","message":"parallel_sub_agent_completed","data":{"items":'
                '[{"agent_id":22,"code":"OK","message":"ok"},{"agent_id":23,"code":"INVOKE_FAILED","message":"x"}]}}'
            ),
            tool_call_id="tc2",
            name="workbench_invoke_sub_agents_parallel",
        ),
    ]
    out = _collect_fan_in_summary(msgs)
    assert out["total_calls"] == 3
    assert out["success_calls"] == 2
    assert out["failed_calls"] == 1
    assert out["unique_child_agent_ids"] == [21, 22, 23]
    assert all(
        isinstance(x["child_agent_id"], int) or x["child_agent_id"] is None for x in out["items"]
    )
    assert out["child_stats"] == [
        {
            "child_agent_id": 21,
            "total_calls": 1,
            "success_calls": 1,
            "failed_calls": 0,
            "last_code": "OK",
        },
        {
            "child_agent_id": 22,
            "total_calls": 1,
            "success_calls": 1,
            "failed_calls": 0,
            "last_code": "OK",
        },
        {
            "child_agent_id": 23,
            "total_calls": 1,
            "success_calls": 0,
            "failed_calls": 1,
            "last_code": "INVOKE_FAILED",
        },
    ]
