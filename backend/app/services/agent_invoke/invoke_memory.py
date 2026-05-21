"""prepare_invoke 记忆段：设置加载、DB 合并策略、观测装饰与模板管线。

与 ``app/domain/agent/invoke_memory`` 配合：领域规则在 domain，此处为适配与编排。

含 :class:`InvokePromptBuilder`：从 ``AgentInvokeRequest`` 构造 ``PromptSlots``（含与 DB 历史合并）。
字符串字段已在 Pydantic 模型中规范化，此处不再 ``strip()``。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import redis.asyncio as redis

from app.agent.kernel import (
    HistoryTurnSlot,
    PromptSlots,
    ToolCallSlot,
    ToolResultSlot,
)
from app.domain.agent import AgentMemorySettings, AgentMemorySettingsParser
from app.domain.agent.invoke_memory import (
    ConversationInvokeMemoryPort,
    HistoryMergeKind,
    MemoryPrepareContext,
    PassthroughHistoryMergeStrategy,
    apply_max_history_rounds_cap,
    resolve_history_merge_kind,
)
from app.domain.agent.knowledge_binding import (
    AgentKnowledgeBindingParser,
    AgentKnowledgeBindingSettings,
)
from app.domain.conversation.layered_rolling_summary import slice_turns_for_layered_tail
from app.infrastructure.cache import get_agent_memory_cache
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.repositories.agent_repo import AgentRepository
from app.schemas.agent import AgentInvokeRequest

if TYPE_CHECKING:
    from app.domain.agent.invoke_memory import HistoryMergeStrategy

logger = logging.getLogger(__name__)


class InvokePromptBuilder:
    """提示词槽位：请求体 ``prompts`` → ``PromptSlots``（无状态，纯静态逻辑）。"""

    @staticmethod
    def slots_from_body(body: AgentInvokeRequest) -> PromptSlots | None:
        p = body.prompts
        if p is None:
            return None
        system_prompt = p.system_prompt
        context = p.context
        turns: list[HistoryTurnSlot] = []
        for turn in p.chat_history or []:
            u = turn.user
            if not u:
                continue
            a = turn.assistant or ""
            tcs = tuple(
                ToolCallSlot(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments or "{}",
                )
                for tc in (turn.tool_calls or [])
            )
            trs = tuple(
                ToolResultSlot(
                    tool_call_id=tr.tool_call_id,
                    name=tr.name,
                    content=tr.content if tr.content is not None else "",
                )
                for tr in (turn.tool_results or [])
            )
            if not a and not tcs:
                continue
            turns.append(HistoryTurnSlot(user=u, assistant=a, tool_calls=tcs, tool_results=trs))
        if not any((system_prompt, context)) and not turns:
            return None
        return PromptSlots(
            system_prompt=system_prompt,
            context=context,
            rolling_summary=None,
            history_turns=tuple(turns),
            max_history_rounds=p.max_history_rounds,
            max_history_tokens=p.max_history_tokens,
        )

    @staticmethod
    def compose_with_db_history(
        body: AgentInvokeRequest,
        prev_slots: PromptSlots | None,
        db_turns: tuple[HistoryTurnSlot, ...],
        *,
        max_rounds_cap: int,
        rolling_summary: str | None = None,
    ) -> PromptSlots | None:
        """以 DB 轮次为 history_turns；system/context/max 来自请求体（并与 ``prev_slots`` 合并缺省）。

        ``rolling_summary`` 来自 ``conversation_session.summary``（服务端 SSOT），非客户端字段。
        """
        p = body.prompts
        system_prompt = p.system_prompt if p else None
        context = p.context if p else None
        max_r = p.max_history_rounds if p else 10
        max_r = max(1, min(max_r, max_rounds_cap))
        max_tok = p.max_history_tokens if p else None
        if prev_slots:
            if system_prompt is None:
                system_prompt = prev_slots.system_prompt
            if context is None:
                context = prev_slots.context
            if max_tok is None:
                max_tok = prev_slots.max_history_tokens
        rs = (rolling_summary or "").strip() or None
        if not db_turns and not any((system_prompt, context)) and not rs:
            return None
        return PromptSlots(
            system_prompt=system_prompt,
            context=context,
            rolling_summary=rs,
            history_turns=db_turns,
            max_history_rounds=max_r,
            max_history_tokens=max_tok,
        )


class AgentMemorySettingsLoader:
    """从 ``agent_entity.config_json.memory`` 解析；可选 Redis 读穿缓存。"""

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager | None,
        redis_client: redis.Redis | None,
    ) -> None:
        self._db = db_manager
        self._redis = redis_client

    async def load_for_invoke(
        self,
        body: AgentInvokeRequest,
    ) -> tuple[AgentMemorySettings, AgentKnowledgeBindingSettings]:
        mem_parser = AgentMemorySettingsParser()
        kb_parser = AgentKnowledgeBindingParser()
        if body.agent_id is None or self._db is None:
            return mem_parser.parse(None), kb_parser.parse(None)
        aid = body.agent_id
        mem_cache = get_agent_memory_cache()
        if self._redis is not None:
            hit = await mem_cache.get(self._redis, aid)
            if hit is not None:
                return hit
        row = await AgentRepository.get_by_id(aid, db_manager=self._db)
        if row is None:
            return mem_parser.parse(None), kb_parser.parse(None)
        settings = mem_parser.parse(row.config_json)
        knowledge = kb_parser.parse(row.config_json)
        if self._redis is not None:
            await mem_cache.set(self._redis, aid, settings, knowledge)
        return settings, knowledge


class ServerDbHistoryMergeStrategy:
    """从 DB 组装轮次 + 滚动摘要，并评估异步摘要调度。"""

    async def merge(
        self,
        base_slots: PromptSlots | None,
        ctx: MemoryPrepareContext,
        conversation: ConversationInvokeMemoryPort | None,
    ) -> PromptSlots | None:
        if conversation is None or ctx.effective_sid is None:
            logger.warning(
                "ServerDbHistoryMergeStrategy: missing conversation or session_id",
                extra={"event": "agent.memory.strategy.server_db.invalid_context"},
            )
            return base_slots
        sid = ctx.effective_sid
        mem = ctx.memory_settings
        db_turns = await conversation.prepare_history_turns_for_model_invoke(
            sid,
            ctx.body.user_message,
            max_history_rounds_cap=mem.max_history_rounds_cap,
        )
        rs_snap = await conversation.load_rolling_summary(sid)
        rolling_summary = rs_snap.text if rs_snap else None
        db_turns = slice_turns_for_layered_tail(
            db_turns,
            rolling_summary,
            min_tail_raw_rounds=mem.min_tail_raw_rounds,
        )
        slots = InvokePromptBuilder.compose_with_db_history(
            ctx.body,
            base_slots,
            tuple(db_turns),
            max_rounds_cap=mem.max_history_rounds_cap,
            rolling_summary=rolling_summary,
        )
        await conversation.maybe_schedule_rolling_summary(sid, mem)
        return slots


class ObservedHistoryMergeStrategy:
    """Decorator：包装任意 :class:`HistoryMergeStrategy`，记录 merge 耗时。"""

    def __init__(self, inner: HistoryMergeStrategy, *, log: logging.Logger | None = None) -> None:
        self._inner = inner
        self._log = log or logger

    async def merge(
        self,
        base_slots: PromptSlots | None,
        ctx: MemoryPrepareContext,
        conversation,
    ) -> PromptSlots | None:
        name = type(self._inner).__name__
        t0 = time.perf_counter()
        try:
            out = await self._inner.merge(base_slots, ctx, conversation)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._log.info(
                "agent.memory.history_merge.done",
                extra={
                    "event": "agent.memory.history_merge.done",
                    "strategy": name,
                    "duration_ms": round(dt_ms, 2),
                    "has_effective_session": ctx.effective_sid is not None,
                },
            )
            return out
        except Exception:
            self._log.exception(
                "agent.memory.history_merge.failed",
                extra={
                    "event": "agent.memory.history_merge.failed",
                    "strategy": name,
                    "has_effective_session": ctx.effective_sid is not None,
                },
            )
            raise


def with_observation(
    strategy: HistoryMergeStrategy,
    *,
    enabled: bool = True,
    log: logging.Logger | None = None,
) -> HistoryMergeStrategy:
    if not enabled:
        return strategy
    return ObservedHistoryMergeStrategy(strategy, log=log)


class PrepareInvokeMemoryPipeline:
    """模板方法：``slots_from_body`` → ``HistoryMergeStrategy`` → ``max_history_rounds`` cap。"""

    def __init__(self, *, observe_strategy: bool = True) -> None:
        self._observe_strategy = observe_strategy

    async def build_prompt_slots(
        self,
        *,
        body: AgentInvokeRequest,
        effective_sid: str | None,
        memory_settings: AgentMemorySettings,
        conversation,
        has_db: bool,
    ) -> PromptSlots | None:
        ctx = MemoryPrepareContext(
            body=body,
            effective_sid=effective_sid,
            memory_settings=memory_settings,
            has_db=has_db,
            has_conversation_service=conversation is not None,
        )
        base_slots = InvokePromptBuilder.slots_from_body(body)
        kind = resolve_history_merge_kind(ctx)
        raw = (
            PassthroughHistoryMergeStrategy()
            if kind == HistoryMergeKind.PASSTHROUGH
            else ServerDbHistoryMergeStrategy()
        )
        strategy = with_observation(raw, enabled=self._observe_strategy)
        merged = await strategy.merge(base_slots, ctx, conversation)
        return apply_max_history_rounds_cap(merged, memory_settings)
