"""滚动摘要触发纯函数。"""

from app.domain.agent import AgentMemorySettingsParser
from app.domain.conversation.rolling_summary_trigger import decide_rolling_summary_trigger


def _defaults(**overrides: object):
    base = dict(
        memory=AgentMemorySettingsParser().parse(None),
        context_window_tokens=10_000,
        history_total_tokens=3000,
        summary_job_status=0,
        summary_version=0,
        summary_anchor_turn_index=0,
        max_user_turn_index=25,
        total_message_tokens=3000,
        tokens_after_anchor=500,
        idle_statuses=(0,),
    )
    base.update(overrides)
    return base


def test_not_idle_returns_none() -> None:
    assert decide_rolling_summary_trigger(**_defaults(summary_job_status=2)) is None


def test_urgent_ratio() -> None:
    # 0.75 > default urgent 0.7
    r = decide_rolling_summary_trigger(
        **_defaults(history_total_tokens=7500, context_window_tokens=10_000),
    )
    assert r == "urgent"


def test_threshold_requires_incremental() -> None:
    # ratio 0.4 达常规线，但首摘「增量」不足：5 轮且仅 500 token（<20 轮且 <1000 token）
    assert (
        decide_rolling_summary_trigger(
            **_defaults(
                history_total_tokens=4000,
                max_user_turn_index=5,
                total_message_tokens=500,
                tokens_after_anchor=100,
            ),
        )
        is None
    )


def test_threshold_when_rounds_enough() -> None:
    assert (
        decide_rolling_summary_trigger(
            **_defaults(
                history_total_tokens=4500,
                max_user_turn_index=25,
                total_message_tokens=4500,
                tokens_after_anchor=100,
            ),
        )
        == "threshold"
    )


def test_below_ratio_none() -> None:
    # 比例极低且轮次不足分层里程碑（默认 9）：不触发
    assert (
        decide_rolling_summary_trigger(
            **_defaults(history_total_tokens=1000, max_user_turn_index=5),
        )
        is None
    )


def test_layered_milestone_triggers_without_token_ratio() -> None:
    """分层里程碑 max_u >= anchor+B+W 时，不依赖 Token 比例即可入队。"""
    assert (
        decide_rolling_summary_trigger(
            **_defaults(
                history_total_tokens=1000,
                max_user_turn_index=10,
            ),
        )
        == "threshold"
    )


def test_incremental_tokens_after_anchor() -> None:
    assert (
        decide_rolling_summary_trigger(
            **_defaults(
                history_total_tokens=4500,
                max_user_turn_index=5,
                total_message_tokens=4500,
                summary_version=1,
                summary_anchor_turn_index=10,
                tokens_after_anchor=1200,
            ),
        )
        == "threshold"
    )
