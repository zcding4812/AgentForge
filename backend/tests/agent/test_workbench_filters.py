from app.agent.adapters.tools.workbench.tool_support import (
    WORKBENCH_BUILTIN_TOOL_FILTER,
    tool_names_from_agent_config_json,
)


def test_workbench_builtin_tool_filter_strips_prefix() -> None:
    assert WORKBENCH_BUILTIN_TOOL_FILTER.filter_names(
        ["server_time", "workbench_list_agents", "workbench_invoke_sub_agent", "custom"]
    ) == ["server_time", "custom"]


def test_tool_names_from_agent_config_json() -> None:
    assert tool_names_from_agent_config_json({"tool_names": ["a", "workbench_x", "b"]}) == [
        "a",
        "b",
    ]
    assert tool_names_from_agent_config_json(None) == []
