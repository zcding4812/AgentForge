from __future__ import annotations

from app.agent.adapters.tools.tool_choice_arg import openai_tool_choice_arg
from app.agent.kernel.spec import ToolChoicePolicy


def test_openai_tool_choice_arg_none_policy() -> None:
    assert openai_tool_choice_arg(None) is None


def test_openai_tool_choice_arg_auto_omits() -> None:
    assert openai_tool_choice_arg(ToolChoicePolicy(mode="auto")) is None


def test_openai_tool_choice_arg_modes() -> None:
    assert openai_tool_choice_arg(ToolChoicePolicy(mode="none")) == "none"
    assert openai_tool_choice_arg(ToolChoicePolicy(mode="required")) == "required"


def test_openai_tool_choice_arg_specific() -> None:
    assert openai_tool_choice_arg(ToolChoicePolicy(mode="specific", forced_tool_name="search")) == {
        "type": "function",
        "function": {"name": "search"},
    }


def test_openai_tool_choice_arg_specific_blank_name() -> None:
    assert openai_tool_choice_arg(ToolChoicePolicy(mode="specific", forced_tool_name="  ")) is None
