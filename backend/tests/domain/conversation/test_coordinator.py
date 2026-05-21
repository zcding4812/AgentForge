from types import SimpleNamespace

from app.agent.kernel.spec import HistoryTurnSlot
from app.domain.conversation import ConversationHistoryCoordinator


def test_prepare_for_model_invoke_dedupes() -> None:
    rows = [
        SimpleNamespace(role="user", content="dup", message_metadata=None),
        SimpleNamespace(role="assistant", content="ok", message_metadata=None),
    ]
    c = ConversationHistoryCoordinator()
    turns = c.prepare_turns_for_model_invoke(rows, "dup")
    assert turns == []


def test_assemble_turns_only_skips_dedupe() -> None:
    rows = [
        SimpleNamespace(role="user", content="dup", message_metadata=None),
        SimpleNamespace(role="assistant", content="ok", message_metadata=None),
    ]
    c = ConversationHistoryCoordinator()
    turns = c.assemble_turns_only(rows)
    assert turns == [HistoryTurnSlot(user="dup", assistant="ok")]
