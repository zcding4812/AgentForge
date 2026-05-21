"""服务端 DB 历史合并策略（Mock 会话端口）。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.domain.agent import AgentMemorySettingsParser, MemoryPrepareContext
from app.schemas.agent import AgentInvokeRequest, PromptEngineeringBody
from app.services.agent_invoke.invoke_memory import ServerDbHistoryMergeStrategy


def _req(**kwargs: object) -> AgentInvokeRequest:
    return AgentInvokeRequest.model_validate(
        {
            "agent_kind": "simple_chat",
            "user_message": "hi",
            **kwargs,
        },
    )


def test_server_db_merge_returns_base_when_no_conversation() -> None:
    strat = ServerDbHistoryMergeStrategy()
    mem = AgentMemorySettingsParser().parse(None)
    ctx = MemoryPrepareContext(
        body=_req(),
        effective_sid="sid-1",
        memory_settings=mem,
        has_db=True,
        has_conversation_service=False,
    )

    out = asyncio.run(strat.merge(None, ctx, None))
    assert out is None


def test_server_db_merge_uses_conversation() -> None:
    conv = MagicMock()
    conv.prepare_history_turns_for_model_invoke = AsyncMock(return_value=[])
    conv.load_rolling_summary = AsyncMock(return_value=None)
    conv.maybe_schedule_rolling_summary = AsyncMock(return_value=None)
    strat = ServerDbHistoryMergeStrategy()
    mem = AgentMemorySettingsParser().parse(None)
    body = _req(agent_id=1, prompts=PromptEngineeringBody(system_prompt="sys"))
    ctx = MemoryPrepareContext(
        body=body,
        effective_sid="sid-1",
        memory_settings=mem,
        has_db=True,
        has_conversation_service=True,
    )

    asyncio.run(strat.merge(None, ctx, conv))
    conv.prepare_history_turns_for_model_invoke.assert_called_once()
    conv.maybe_schedule_rolling_summary.assert_called_once()
