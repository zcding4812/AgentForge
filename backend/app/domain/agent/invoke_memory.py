"""Agent invoke 前记忆段：上下文、策略种类、端口、透传与轮次裁剪（无 FastAPI / 无仓储 I/O）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.agent.kernel import HistoryTurnSlot, PromptSlots
from app.agent.kernel.ports import RollingSummarySnapshot
from app.domain.agent.memory_settings import AgentMemorySettings
from app.schemas.agent import AgentInvokeRequest


@dataclass(frozen=True, slots=True)
class MemoryPrepareContext:
    """供 :class:`HistoryMergeStrategy` 与管线使用；不含用户正文日志。"""

    body: AgentInvokeRequest
    effective_sid: str | None
    memory_settings: AgentMemorySettings
    has_db: bool
    has_conversation_service: bool

    @property
    def prefer_client_chat_history(self) -> bool:
        client_hist = self.body.prompts.chat_history if self.body.prompts else []
        return self.memory_settings.allow_client_chat_history and bool(client_hist)


@runtime_checkable
class ConversationInvokeMemoryPort(Protocol):
    """组装 DB 历史、滚动摘要与摘要调度所需的最小异步面（由 :class:`ConversationService` 满足）。"""

    async def prepare_history_turns_for_model_invoke(
        self,
        session_id: str,
        current_user_message: str,
        *,
        max_history_rounds_cap: int,
    ) -> list[HistoryTurnSlot]: ...

    async def load_rolling_summary(self, session_id: str) -> RollingSummarySnapshot | None: ...

    async def maybe_schedule_rolling_summary(
        self,
        session_id: str,
        memory: AgentMemorySettings,
    ) -> None: ...


@runtime_checkable
class HistoryMergeStrategy(Protocol):
    async def merge(
        self,
        base_slots: PromptSlots | None,
        ctx: MemoryPrepareContext,
        conversation: ConversationInvokeMemoryPort | None,
    ) -> PromptSlots | None: ...


class HistoryMergeKind(StrEnum):
    PASSTHROUGH = "passthrough"
    SERVER_DB = "server_db"


def resolve_history_merge_kind(ctx: MemoryPrepareContext) -> HistoryMergeKind:
    """按上下文选择策略种类（无 I/O）。"""
    if ctx.body.workbench_force_client_history:
        return HistoryMergeKind.PASSTHROUGH
    if not ctx.effective_sid or not ctx.has_db or not ctx.has_conversation_service:
        return HistoryMergeKind.PASSTHROUGH
    if ctx.prefer_client_chat_history:
        return HistoryMergeKind.PASSTHROUGH
    return HistoryMergeKind.SERVER_DB


class PassthroughHistoryMergeStrategy:
    """不合并 DB：无会话、无库、或客户端 chat_history 优先。"""

    async def merge(
        self,
        base_slots: PromptSlots | None,
        ctx: MemoryPrepareContext,
        conversation: ConversationInvokeMemoryPort | None,
    ) -> PromptSlots | None:
        return base_slots


def apply_max_history_rounds_cap(
    slots: PromptSlots | None,
    memory_settings: AgentMemorySettings,
) -> PromptSlots | None:
    if slots is None:
        return None
    cap = memory_settings.max_history_rounds_cap
    if slots.max_history_rounds > cap:
        return replace(slots, max_history_rounds=cap)
    return slots
