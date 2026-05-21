"""滚动摘要调度：纯函数策略（与 I/O 无关），与 ``AgentMemorySettings`` 对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.agent.memory_settings import AgentMemorySettings
from app.domain.conversation.layered_rolling_summary import layered_compression_milestone_reached

RollingSummaryReason = Literal["threshold", "urgent"]


@dataclass(frozen=True, slots=True)
class RollingSummaryTriggerMetrics:
    """供触发策略使用：基于最近一批消息的 token 估算与轮次（由仓储填充）。"""

    total_message_tokens_estimated: int
    max_user_turn_index: int
    tokens_after_anchor: int


def decide_rolling_summary_trigger(
    *,
    memory: AgentMemorySettings,
    context_window_tokens: int,
    history_total_tokens: int,
    summary_job_status: int,
    summary_version: int,
    summary_anchor_turn_index: int,
    max_user_turn_index: int,
    total_message_tokens: int,
    tokens_after_anchor: int,
    idle_statuses: tuple[int, ...],
) -> RollingSummaryReason | None:
    """
    返回待入队的 ``reason``；不满足时返回 ``None``。

    - ``idle_statuses``：视为可接受新摘要任务的状态（通常为 idle）。
    - 增量门槛：尚无成功摘要（``summary_version==0`` 且锚点为 0）时，用全量轮次/全量消息 token；
      否则用 ``max_user_turn_index - anchor`` 与 ``tokens_after_anchor``。
    """
    if summary_job_status not in idle_statuses:
        return None

    cw = max(int(context_window_tokens), 1)
    total = max(int(history_total_tokens), 0)
    ratio = total / cw

    urgent = float(memory.summary_context_ratio_urgent)
    threshold = float(memory.summary_context_ratio_threshold)

    if ratio >= urgent:
        return "urgent"

    if layered_compression_milestone_reached(
        summary_anchor_turn_index=int(summary_anchor_turn_index),
        max_user_turn_index=int(max_user_turn_index),
        min_tail_raw_rounds=int(memory.min_tail_raw_rounds),
        compress_batch_rounds=int(memory.compress_batch_rounds),
    ):
        return "threshold"

    if ratio < threshold:
        return None

    if summary_version == 0 and summary_anchor_turn_index == 0:
        new_rounds = int(max_user_turn_index)
        new_tokens = int(total_message_tokens)
    else:
        new_rounds = max(0, int(max_user_turn_index) - int(summary_anchor_turn_index))
        new_tokens = max(0, int(tokens_after_anchor))

    min_r = int(memory.summary_min_rounds_since_last)
    min_t = int(memory.summary_min_tokens_since_last)
    incremental_ok = new_rounds >= min_r or new_tokens >= min_t
    if incremental_ok:
        return "threshold"
    return None
