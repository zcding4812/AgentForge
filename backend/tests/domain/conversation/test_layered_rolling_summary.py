"""分层滚动摘要纯函数。"""

from app.agent.kernel.spec import HistoryTurnSlot
from app.domain.conversation.layered_rolling_summary import (
    layered_compression_milestone_reached,
    slice_turns_for_layered_tail,
)


def test_milestone_first_batch() -> None:
    assert (
        layered_compression_milestone_reached(
            summary_anchor_turn_index=0,
            max_user_turn_index=8,
            min_tail_raw_rounds=6,
            compress_batch_rounds=3,
        )
        is False
    )
    assert (
        layered_compression_milestone_reached(
            summary_anchor_turn_index=0,
            max_user_turn_index=9,
            min_tail_raw_rounds=6,
            compress_batch_rounds=3,
        )
        is True
    )


def test_milestone_second_batch() -> None:
    assert (
        layered_compression_milestone_reached(
            summary_anchor_turn_index=3,
            max_user_turn_index=11,
            min_tail_raw_rounds=6,
            compress_batch_rounds=3,
        )
        is False
    )
    assert (
        layered_compression_milestone_reached(
            summary_anchor_turn_index=3,
            max_user_turn_index=12,
            min_tail_raw_rounds=6,
            compress_batch_rounds=3,
        )
        is True
    )


def test_slice_tail_only_when_summary_present() -> None:
    turns = [HistoryTurnSlot(user=f"u{i}", assistant=f"a{i}") for i in range(10)]
    assert slice_turns_for_layered_tail(turns, None, min_tail_raw_rounds=6) == turns
    assert slice_turns_for_layered_tail(turns, "   ", min_tail_raw_rounds=6) == turns
    out = slice_turns_for_layered_tail(turns, "有摘要", min_tail_raw_rounds=6)
    assert len(out) == 6
    assert out[0].user == "u4"
