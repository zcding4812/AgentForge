"""使用 LangChain ``trim_messages`` 对「窗内历史」段做 token 预算裁剪（不改变轮次逻辑，可与 ``max_history_rounds`` 叠加）。"""

from __future__ import annotations

import logging

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

logger = logging.getLogger(__name__)


def _split_prefix_history_and_current(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage], BaseMessage | None]:
    """拆分消息为三段：

    - 前缀：开头连续的 ``SystemMessage``
    - 中间：对话历史（Human / AI / Tool）
    - 当前：最后一条用户提问（应为 ``HumanMessage``）

    若末条不是 ``HumanMessage``，则将末条并入中间段并返回 ``current=None``（由上层决定是否原样返回）。
    """
    if not messages:
        return [], [], None

    last_msg = messages[-1]
    rest_msgs = messages[:-1]

    prefix: list[BaseMessage] = []
    split_idx = 0
    while split_idx < len(rest_msgs) and isinstance(rest_msgs[split_idx], SystemMessage):
        prefix.append(rest_msgs[split_idx])
        split_idx += 1

    history = rest_msgs[split_idx:]

    if isinstance(last_msg, HumanMessage):
        return prefix, history, last_msg

    logger.warning(
        "消息裁剪：末条应为 HumanMessage，实际为 %s",
        type(last_msg).__name__,
    )
    return prefix, history + [last_msg], None


def _trim_history_middle_by_tokens(
    middle: list[BaseMessage],
    *,
    max_tokens: int,
    token_counter: str = "approximate",
) -> list[BaseMessage]:
    """仅裁剪中间对话历史，``strategy=last`` 保留离当前轮最近的内容。"""
    if max_tokens <= 0 or not middle:
        return middle

    return trim_messages(
        middle,
        max_tokens=max_tokens,
        token_counter=token_counter,  # type: ignore[arg-type]
        strategy="last",
        allow_partial=True,
        include_system=False,
    )


def apply_max_history_token_budget(
    messages: list[BaseMessage],
    *,
    max_history_tokens: int | None,
    token_counter: str = "approximate",
) -> list[BaseMessage]:
    """应用 token 预算：始终保留系统前缀与末条用户 Human；只裁剪中间历史。

    ``max_history_tokens`` 为 ``None`` 或 ``<=0`` 时不修改。可与 ``max_history_rounds`` 叠加。
    """
    if max_history_tokens is None or max_history_tokens <= 0:
        return messages

    prefix, history, current = _split_prefix_history_and_current(messages)
    if current is None:
        return messages

    trimmed_history = _trim_history_middle_by_tokens(
        middle=history,
        max_tokens=max_history_tokens,
        token_counter=token_counter,
    )
    return [*prefix, *trimmed_history, current]
