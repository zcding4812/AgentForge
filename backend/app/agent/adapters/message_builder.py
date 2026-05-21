"""Agent 初始消息构建：四类顺序、LangChain 消息列表。"""

from __future__ import annotations

import json
from typing import Any, Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.adapters.memory import apply_max_history_token_budget
from app.agent.kernel.default_prompts import SYSTEM_DEFAULT_PROMPT
from app.agent.kernel.spec import HistoryTurnSlot, PromptSlots, ToolCallSlot

DEFAULT_MAX_HISTORY_ROUNDS: Final[int] = 10

# 与 ``kernel.default_prompts.SYSTEM_DEFAULT_PROMPT``、``id=system`` 同源（供历史导入名不变）
DEFAULT_SYSTEM_PROMPT: Final[str] = SYSTEM_DEFAULT_PROMPT


class MessageBuilder:
    """按规范构建初始 ``messages`` 列表（System → 历史含工具 → 当前用户 Human）。"""

    def __init__(
        self,
        default_system_prompt: str | None = None,
        default_max_history: int = DEFAULT_MAX_HISTORY_ROUNDS,
    ) -> None:
        self._default_system_prompt: Final[str] = default_system_prompt or DEFAULT_SYSTEM_PROMPT
        self._default_max_history: Final[int] = default_max_history

    def build_initial_messages(
        self,
        user_message: str,
        slots: PromptSlots | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        self._append_system_prompt(messages, slots)
        self._append_rolling_summary(messages, slots)
        self._append_workbench_workspace_catalog(messages, slots)
        self._append_history(messages, slots)
        self._append_current_user(messages, user_message, slots)
        return apply_max_history_token_budget(
            messages,
            max_history_tokens=slots.max_history_tokens if slots else None,
        )

    def _append_system_prompt(
        self,
        messages: list[BaseMessage],
        slots: PromptSlots | None,
    ) -> None:
        custom = self._compose_system_prompt(slots)
        if slots is not None and slots.omit_platform_default_system:
            if custom:
                messages.append(SystemMessage(content=custom))
            return
        messages.append(
            SystemMessage(content=custom if custom else self._default_system_prompt),
        )

    def _append_history(
        self,
        messages: list[BaseMessage],
        slots: PromptSlots | None,
    ) -> None:
        if not slots:
            return
        max_turns = (
            slots.max_history_rounds if slots.max_history_rounds > 0 else self._default_max_history
        )
        for turn in self._take_last_turns(slots.history_turns, max_turns):
            self._append_single_history_turn(messages, turn)

    def _append_current_user(
        self,
        messages: list[BaseMessage],
        user_message: str,
        slots: PromptSlots | None,
    ) -> None:
        ctx = self._strip(slots.context) if slots else None
        content = self._compose_current_user_content(user_message, ctx)
        messages.append(HumanMessage(content=content))

    @staticmethod
    def _strip(text: str | None) -> str | None:
        if text is None:
            return None
        s = text.strip()
        return s if s else None

    def _compose_system_prompt(self, slots: PromptSlots | None) -> str | None:
        return self._strip(slots.system_prompt) if slots else None

    def _append_rolling_summary(
        self,
        messages: list[BaseMessage],
        slots: PromptSlots | None,
    ) -> None:
        if not slots:
            return
        raw = self._strip(slots.rolling_summary)
        if not raw:
            return
        messages.append(
            SystemMessage(
                content="【较早对话摘要】\n" + raw,
            ),
        )

    def _append_workbench_workspace_catalog(
        self,
        messages: list[BaseMessage],
        slots: PromptSlots | None,
    ) -> None:
        if not slots:
            return
        raw = self._strip(slots.workbench_workspace_agent_catalog)
        if not raw:
            return
        messages.append(SystemMessage(content="【当前命名空间 Agent 一览】\n" + raw))

    def _compose_current_user_content(self, user_message: str, context: str | None) -> str:
        um = user_message.strip()
        ctx = self._strip(context)
        if ctx:
            return f"参考资料：\n{ctx}\n\n用户问题：\n{um}"
        return um

    @staticmethod
    def _take_last_turns(
        turns: tuple[HistoryTurnSlot, ...],
        max_turns: int,
    ) -> tuple[HistoryTurnSlot, ...]:
        if max_turns <= 0 or not turns:
            return ()
        if len(turns) <= max_turns:
            return turns
        return turns[-max_turns:]

    def _append_single_history_turn(
        self,
        messages: list[BaseMessage],
        turn: HistoryTurnSlot,
    ) -> None:
        user_text = self._strip(turn.user)
        if not user_text:
            return
        messages.append(HumanMessage(content=user_text))
        ai_text = self._strip(turn.assistant) or ""
        if turn.tool_calls:
            messages.append(
                AIMessage(
                    content=ai_text,
                    tool_calls=self._tool_calls_to_langchain(turn.tool_calls),
                ),
            )
        else:
            messages.append(AIMessage(content=ai_text))
        for tr in turn.tool_results:
            messages.append(
                ToolMessage(
                    content=tr.content,
                    tool_call_id=tr.tool_call_id,
                    name=tr.name,
                ),
            )

    @staticmethod
    def _tool_calls_to_langchain(
        tool_calls: tuple[ToolCallSlot, ...],
    ) -> list[dict[str, Any]]:
        lc: list[dict[str, Any]] = []
        for tc in tool_calls:
            raw = (tc.arguments or "").strip()
            try:
                args = json.loads(raw) if raw else {}
                args = args if isinstance(args, dict) else {}
            except json.JSONDecodeError:
                args = {}
            lc.append({"name": tc.name, "args": args, "id": tc.id})
        return lc


def build_initial_messages(user_message: str, slots: PromptSlots | None) -> list[BaseMessage]:
    return MessageBuilder().build_initial_messages(user_message, slots)


def build_agent_messages(user_message: str, slots: PromptSlots | None = None) -> list[BaseMessage]:
    """与 ``build_initial_messages`` 同义，保留旧调用点。"""
    return MessageBuilder().build_initial_messages(user_message, slots)
