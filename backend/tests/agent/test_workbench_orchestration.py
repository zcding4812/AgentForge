"""工作台编排边：从末态 messages 解析 ``workbench_invoke_sub_agent``。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.kernel.workbench_orchestration import (
    extract_workbench_orchestration_edges,
)


def test_extract_empty_messages() -> None:
    assert extract_workbench_orchestration_edges([], parent_agent_id=1) == ()


def test_extract_single_invoke() -> None:
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "workbench_invoke_sub_agent",
                    "id": "call_1",
                    "args": {"agent_id": 42, "user_message": "do it"},
                }
            ],
        ),
        ToolMessage(content="{}", tool_call_id="call_1", name="workbench_invoke_sub_agent"),
        AIMessage(content="done"),
    ]
    edges = extract_workbench_orchestration_edges(msgs, parent_agent_id=7)
    assert len(edges) == 1
    assert edges[0].order == 0
    assert edges[0].parent_agent_id == 7
    assert edges[0].child_agent_id == 42


def test_extract_parallel_tool_expands_edges() -> None:
    tasks = [
        {"agent_id": 10, "user_message": "a"},
        {"agent_id": 20, "user_message": "b"},
    ]
    msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "workbench_invoke_sub_agents_parallel",
                    "id": "call_p",
                    "args": {"tasks_json": json.dumps(tasks)},
                }
            ],
        ),
    ]
    edges = extract_workbench_orchestration_edges(msgs, parent_agent_id=1)
    assert len(edges) == 2
    assert edges[0].order == 0 and edges[0].child_agent_id == 10
    assert edges[1].order == 1 and edges[1].child_agent_id == 20


def test_extract_ignores_other_tools() -> None:
    msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "other_tool", "id": "c1", "args": {}},
                {
                    "name": "workbench_invoke_sub_agent",
                    "id": "c2",
                    "args": {"agent_id": 3, "user_message": "x"},
                },
            ],
        ),
    ]
    edges = extract_workbench_orchestration_edges(msgs, parent_agent_id=1)
    assert len(edges) == 1
    assert edges[0].child_agent_id == 3
