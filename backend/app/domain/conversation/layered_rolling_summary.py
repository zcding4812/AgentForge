"""分层滚动摘要：尾窗截取与链式压缩里程碑（纯函数，无 I/O）。"""

from __future__ import annotations

from app.agent.kernel.spec import HistoryTurnSlot


def layered_compression_milestone_reached(
    *,
    summary_anchor_turn_index: int,
    max_user_turn_index: int,
    min_tail_raw_rounds: int,
    compress_batch_rounds: int,
) -> bool:
    """下一批链式压缩是否「已到点」：``max_u >= anchor + B + W``。"""
    a = int(summary_anchor_turn_index)
    u = int(max_user_turn_index)
    w = max(1, int(min_tail_raw_rounds))
    b = max(1, int(compress_batch_rounds))
    return u >= a + b + w


def slice_turns_for_layered_tail(
    turns: list[HistoryTurnSlot],
    rolling_summary: str | None,
    *,
    min_tail_raw_rounds: int,
) -> list[HistoryTurnSlot]:
    """存在非空滚动摘要时，仅保留时间轴上最后 ``W`` 轮原文，避免与摘要重复。"""
    raw = (rolling_summary or "").strip()
    if not raw or not turns:
        return turns
    w = max(1, int(min_tail_raw_rounds))
    if len(turns) <= w:
        return turns
    return turns[-w:]
