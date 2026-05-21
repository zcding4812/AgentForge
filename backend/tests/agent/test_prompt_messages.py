from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.adapters.message_builder import DEFAULT_SYSTEM_PROMPT, build_agent_messages
from app.agent.kernel.spec import HistoryTurnSlot, PromptSlots, ToolCallSlot, ToolResultSlot


def test_build_messages_user_only() -> None:
    msgs = build_agent_messages("hello", None)
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == DEFAULT_SYSTEM_PROMPT
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "hello"


def test_build_messages_full_slots() -> None:
    slots = PromptSlots(
        system_prompt="你是助手\n\n## 能力\n优先简洁\n\n## 约束\n用中文答",
        context="文档：foo",
    )
    msgs = build_agent_messages("问题？", slots)
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert "你是助手" in msgs[0].content
    assert "优先简洁" in msgs[0].content
    assert "用中文答" in msgs[0].content
    assert isinstance(msgs[1], HumanMessage)
    assert "参考资料" in msgs[1].content
    assert "用户问题" in msgs[1].content
    assert "问题？" in msgs[1].content


def test_build_messages_default_system_when_no_custom_system_prompt() -> None:
    slots = PromptSlots(context="仅有参考")
    msgs = build_agent_messages("hi", slots)
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == DEFAULT_SYSTEM_PROMPT
    assert "参考资料" in msgs[1].content


def test_build_messages_workbench_omits_default_system_without_flow_supplement() -> None:
    slots = PromptSlots(
        context="仅有参考",
        omit_platform_default_system=True,
    )
    msgs = build_agent_messages("hi", slots)
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert "参考资料" in msgs[0].content


def test_build_messages_workbench_flow_supplement_only_when_omit_default() -> None:
    slots = PromptSlots(
        system_prompt="## 流程补充\n委派时优先单一路径。",
        omit_platform_default_system=True,
        context="ctx",
    )
    msgs = build_agent_messages("问题", slots)
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert "流程补充" in msgs[0].content
    assert isinstance(msgs[1], HumanMessage)


def test_build_messages_rolling_summary_after_system_before_history() -> None:
    slots = PromptSlots(
        system_prompt="角色",
        rolling_summary="摘要：上一轮结论",
        history_turns=(HistoryTurnSlot("u1", "a1"),),
        max_history_rounds=10,
    )
    msgs = build_agent_messages("last", slots)
    assert len(msgs) == 5
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "角色"
    assert isinstance(msgs[1], SystemMessage)
    assert msgs[1].content.startswith("【较早对话摘要】")
    assert "上一轮结论" in msgs[1].content
    assert msgs[2].content == "u1"
    assert msgs[3].content == "a1"
    assert msgs[4].content == "last"


def test_build_messages_workbench_catalog_after_summary_before_history() -> None:
    slots = PromptSlots(
        system_prompt="角色",
        rolling_summary="摘要：上一轮结论",
        workbench_workspace_agent_catalog="- `agent_id=2` **子** · `react`",
        history_turns=(HistoryTurnSlot("u1", "a1"),),
        max_history_rounds=10,
    )
    msgs = build_agent_messages("last", slots)
    assert len(msgs) == 6
    assert msgs[1].content.startswith("【较早对话摘要】")
    assert msgs[2].content.startswith("【当前命名空间 Agent 一览】")
    assert "agent_id=2" in msgs[2].content
    assert msgs[3].content == "u1"
    assert msgs[4].content == "a1"
    assert msgs[5].content == "last"


def test_build_messages_history_rounds() -> None:
    slots = PromptSlots(
        system_prompt="角色",
        history_turns=(
            HistoryTurnSlot("u1", "a1"),
            HistoryTurnSlot("u2", "a2"),
        ),
        max_history_rounds=10,
    )
    msgs = build_agent_messages("last", slots)
    assert len(msgs) == 6
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[1].content == "u1" and isinstance(msgs[1], HumanMessage)
    assert msgs[2].content == "a1" and isinstance(msgs[2], AIMessage)
    assert msgs[3].content == "u2"
    assert msgs[4].content == "a2"
    assert msgs[5].content == "last" and isinstance(msgs[5], HumanMessage)


def test_build_messages_history_sliding_window() -> None:
    slots = PromptSlots(
        system_prompt="s",
        history_turns=(
            HistoryTurnSlot("u1", "a1"),
            HistoryTurnSlot("u2", "a2"),
            HistoryTurnSlot("u3", "a3"),
        ),
        max_history_rounds=2,
    )
    msgs = build_agent_messages("last", slots)
    assert msgs[1].content == "u2"
    assert msgs[2].content == "a2"
    assert msgs[3].content == "u3"
    assert msgs[4].content == "a3"


def test_build_messages_tool_calls_after_assistant_in_history() -> None:
    slots = PromptSlots(
        system_prompt="角色",
        history_turns=(
            HistoryTurnSlot(
                user="查北京天气",
                assistant="",
                tool_calls=(
                    ToolCallSlot(id="call_w", name="get_weather", arguments='{"city":"北京"}'),
                ),
                tool_results=(
                    ToolResultSlot(tool_call_id="call_w", name="get_weather", content="晴，25°C"),
                ),
            ),
        ),
    )
    msgs = build_agent_messages("谢谢", slots)
    assert len(msgs) == 5
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[1].content == "查北京天气"
    assert isinstance(msgs[2], AIMessage) and msgs[2].tool_calls
    assert isinstance(msgs[3], ToolMessage)
    assert msgs[3].content == "晴，25°C"
    assert msgs[4].content == "谢谢"
