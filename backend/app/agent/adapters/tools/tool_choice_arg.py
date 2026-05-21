"""将 ``ToolChoicePolicy`` 映射为 OpenAI Chat Completions 风格的 ``tool_choice`` 参数。"""

from __future__ import annotations

from typing import Any

from app.agent.kernel.spec import ToolChoiceMode, ToolChoicePolicy


def openai_tool_choice_arg(policy: ToolChoicePolicy | None) -> str | dict[str, Any] | None:
    """返回 ``bind_tools(..., tool_choice=...)`` / ``ModelRequest.override`` 可用的值；无需覆盖则返回 ``None``。"""
    if policy is None:
        return None
    mode: ToolChoiceMode = policy.mode
    if mode == "auto":
        return None
    if mode == "none":
        return "none"
    if mode == "required":
        return "required"
    if mode == "specific":
        name = (policy.forced_tool_name or "").strip()
        if not name:
            return None
        return {"type": "function", "function": {"name": name}}
    return None
