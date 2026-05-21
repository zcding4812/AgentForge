"""invoke 记忆段：策略种类解析与轮次裁剪（无 I/O）。"""

from app.agent.kernel import PromptSlots
from app.domain.agent import (
    AgentMemorySettingsParser,
    HistoryMergeKind,
    MemoryPrepareContext,
    apply_max_history_rounds_cap,
    resolve_history_merge_kind,
)
from app.schemas.agent import AgentInvokeRequest, PromptEngineeringBody


def _req(**kwargs: object) -> AgentInvokeRequest:
    return AgentInvokeRequest.model_validate(
        {
            "agent_kind": "simple_chat",
            "user_message": "hi",
            **kwargs,
        },
    )


def test_resolve_passthrough_without_session() -> None:
    mem = AgentMemorySettingsParser().parse(None)
    ctx = MemoryPrepareContext(
        body=_req(),
        effective_sid=None,
        memory_settings=mem,
        has_db=True,
        has_conversation_service=True,
    )
    assert resolve_history_merge_kind(ctx) == HistoryMergeKind.PASSTHROUGH


def test_resolve_passthrough_when_client_history_preferred() -> None:
    mem = AgentMemorySettingsParser().parse({"memory": {"allow_client_chat_history": True}})
    body = _req(
        prompts=PromptEngineeringBody(
            chat_history=[{"user": "a", "assistant": "b"}],
        ),
    )
    ctx = MemoryPrepareContext(
        body=body,
        effective_sid="sid-1",
        memory_settings=mem,
        has_db=True,
        has_conversation_service=True,
    )
    assert resolve_history_merge_kind(ctx) == HistoryMergeKind.PASSTHROUGH


def test_resolve_server_db_when_defaults_and_session() -> None:
    mem = AgentMemorySettingsParser().parse(None)
    body = _req(agent_id=1)
    ctx = MemoryPrepareContext(
        body=body,
        effective_sid="sid-1",
        memory_settings=mem,
        has_db=True,
        has_conversation_service=True,
    )
    assert resolve_history_merge_kind(ctx) == HistoryMergeKind.SERVER_DB


def test_apply_rounds_cap_clamps() -> None:
    mem = AgentMemorySettingsParser().parse(None)
    cap = mem.max_history_rounds_cap
    slots = PromptSlots(
        system_prompt="s",
        context=None,
        rolling_summary=None,
        history_turns=(),
        max_history_rounds=cap + 5,
    )
    out = apply_max_history_rounds_cap(slots, mem)
    assert out is not None
    assert out.max_history_rounds == cap
