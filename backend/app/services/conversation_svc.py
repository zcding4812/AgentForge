from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Literal

import redis.asyncio as redis
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.agent.adapters.models import RegistryDispatchingChatModelFactory
from app.agent.kernel.ports import MemoryPort, RollingSummarySnapshot
from app.agent.kernel.spec import HistoryTurnSlot, InferenceHyperparameters
from app.core.constants.conversation import (
    CONVERSATION_DETAIL_MESSAGES_DEFAULT_PAGE_SIZE,
    CONVERSATION_LIST_DEFAULT_PAGE_SIZE,
    CONVERSATION_SESSION_ACTIVE,
    SUMMARY_JOB_STATUS_IDLE,
)
from app.core.logger import get_logger
from app.domain.agent import AgentMemorySettingsParser
from app.domain.agent.memory_settings import DEFAULT_MAX_HISTORY_ROUNDS_CAP, AgentMemorySettings
from app.domain.conversation import (
    ConversationDomainError,
    HistoryTurnAssembler,
    HistoryTurnDeduper,
    ReferencedAgentNotFoundError,
    SessionAgentMismatchError,
    SessionNotFoundError,
    SessionNotWritableError,
    decide_rolling_summary_trigger,
    estimate_user_content_tokens,
    normalize_conversation_list_pagination,
    normalize_conversation_messages_pagination,
)
from app.domain.conversation.layered_rolling_summary import layered_compression_milestone_reached
from app.infrastructure.cache import get_history_turns_cache
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.infrastructure.jobs import (
    RollingSummaryJob,
    RollingSummaryWorker,
    enqueue_rolling_summary,
)
from app.models.conversation_mod import AgentConversationMessage, AgentConversationSession
from app.repositories.agent_repo import AgentRepository
from app.repositories.conversation_repo import ConversationRepository
from app.schemas.conversation import (
    ConversationDetailData,
    ConversationMessageAppendBody,
    ConversationMessageExecutionOut,
    ConversationMessageOut,
    ConversationSessionCreateBody,
    ConversationSessionListData,
    ConversationSessionOut,
    ConversationSessionPatchBody,
)
from app.schemas.response import PageMeta
from app.services.agent_invoke.snapshot_builder import InvokeSnapshotBuilder


def _message_out_list_row(
    row: AgentConversationMessage,
    *,
    include_message_metadata: bool,
) -> ConversationMessageOut:
    """列表分页：可选不附带 metadata（列已 defer 时禁止访问 ``message_metadata`` 以免隐式加载）。"""
    if include_message_metadata:
        return ConversationMessageOut.model_validate(row)
    return ConversationMessageOut(
        id=row.id,
        session_id=row.session_id,
        agent_id=row.agent_id,
        role=row.role,  # type: ignore[arg-type]
        content=row.content,
        content_type=row.content_type,
        created_at=row.created_at,
        metadata=None,
        turn_index=row.turn_index,
        reply_message_id=row.reply_message_id,
        tokens=row.tokens,
    )


def _session_allows_append(sess: AgentConversationSession | None) -> bool:
    """会话存在且为活跃状态时可追加消息。"""
    return sess is not None and sess.status == CONVERSATION_SESSION_ACTIVE


logger = get_logger(__name__)

ManualRollingSummaryResult = Literal["queued", "duplicate", "queue_unavailable"]


class ConversationService(MemoryPort):
    """Agent 对话会话与消息的持久化编排（单租户）。

    依赖在构造时注入：``db_manager`` 必选；``redis`` 可选，用于已组装轮次读穿缓存与失效。

    实现 :class:`~app.agent.kernel.ports.MemoryPort`：滚动摘要加载与进程内队列调度（lifespan 消费者）。
    """

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager,
        redis: redis.Redis | None = None,
    ) -> None:
        self._db = db_manager
        self._redis = redis

    async def _fetch_session_row(self, session_id: str) -> AgentConversationSession | None:
        """按 ``session_id`` 读取会话行（不含已软删，与仓储默认一致）。"""
        return await ConversationRepository.get_session_row(session_id, db_manager=self._db)

    async def load_rolling_summary(self, session_id: str) -> RollingSummarySnapshot | None:
        """:class:`MemoryPort`：读取已落库的滚动摘要快照。"""
        row = await self._fetch_session_row(session_id)
        if row is None:
            return None
        text = (row.summary or "").strip() or None
        return RollingSummarySnapshot(
            text=text,
            version=row.summary_version,
            job_status=row.summary_job_status,
        )

    async def schedule_summarize(
        self,
        session_id: str,
        *,
        reason: Literal["threshold", "urgent", "manual"],
    ) -> None:
        """:class:`MemoryPort`：将摘要任务写入进程内队列（lifespan 后台协程消费）。

        入队前将 ``summary_job_status`` 从 idle 置为 pending，避免同会话短时间重复入队；
        若进程内队列未注册则回滚 pending 并记日志（不含用户原文）。
        """
        marked = await ConversationRepository.try_mark_summary_job_pending(
            session_id, db_manager=self._db
        )
        if not marked:
            logger.info(
                "rolling_summary schedule skipped",
                extra={
                    "event": "rolling_summary.schedule_skipped",
                    "session_id": session_id,
                    "trigger_reason": reason,
                    "skip": "not_idle_or_race",
                },
            )
            return
        ok = await enqueue_rolling_summary(session_id, reason)
        if not ok:
            await ConversationRepository.revert_summary_job_pending_to_idle(
                session_id, db_manager=self._db
            )
            logger.warning(
                "rolling_summary enqueue failed after pending mark",
                extra={
                    "event": "rolling_summary.enqueue_failed",
                    "session_id": session_id,
                    "trigger_reason": reason,
                },
            )
            return
        logger.info(
            "rolling_summary enqueued",
            extra={
                "event": "rolling_summary.enqueued",
                "session_id": session_id,
                "trigger_reason": reason,
            },
        )

    async def maybe_schedule_rolling_summary(
        self,
        session_id: str,
        memory: AgentMemorySettings,
    ) -> None:
        """在 ``prepare_invoke`` 读库组装后调用：按 ``memory`` 中比例与增量门槛决定是否入队（非阻塞）。"""
        row = await self._fetch_session_row(session_id)
        if row is None:
            return
        metrics = await ConversationRepository.rolling_summary_trigger_metrics(
            session_id,
            summary_anchor_turn_index=int(row.summary_anchor_turn_index or 0),
            db_manager=self._db,
        )
        summary_tok = estimate_user_content_tokens(row.summary or "")
        history_total = metrics.total_message_tokens_estimated + summary_tok

        reason = decide_rolling_summary_trigger(
            memory=memory,
            context_window_tokens=memory.summary_context_window_tokens,
            history_total_tokens=history_total,
            summary_job_status=int(row.summary_job_status),
            summary_version=int(row.summary_version),
            summary_anchor_turn_index=int(row.summary_anchor_turn_index or 0),
            max_user_turn_index=metrics.max_user_turn_index,
            total_message_tokens=metrics.total_message_tokens_estimated,
            tokens_after_anchor=metrics.tokens_after_anchor,
            idle_statuses=(SUMMARY_JOB_STATUS_IDLE,),
        )
        if reason is None:
            return
        cw = max(int(memory.summary_context_window_tokens), 1)
        ratio = history_total / cw
        logger.info(
            "rolling_summary trigger matched",
            extra={
                "event": "rolling_summary.trigger_matched",
                "session_id": session_id,
                "trigger_reason": reason,
                "history_total_tokens": history_total,
                "context_window_tokens": memory.summary_context_window_tokens,
                "ratio": round(ratio, 6),
                "max_user_turn_index": metrics.max_user_turn_index,
                "summary_anchor_turn_index": int(row.summary_anchor_turn_index or 0),
                "summary_version": int(row.summary_version),
                "job_status": int(row.summary_job_status),
            },
        )
        await self.schedule_summarize(session_id, reason=reason)

    async def request_manual_rolling_summary(self, session_id: str) -> ManualRollingSummaryResult:
        """手动入队：``queued`` / ``duplicate``（已有 pending/running）/ ``queue_unavailable``。"""
        if await self._fetch_session_row(session_id) is None:
            raise LookupError("会话不存在或已删除")
        marked = await ConversationRepository.try_mark_summary_job_pending(
            session_id, db_manager=self._db
        )
        if not marked:
            logger.info(
                "rolling_summary manual skipped",
                extra={
                    "event": "rolling_summary.manual_skipped",
                    "session_id": session_id,
                    "skip": "not_idle",
                },
            )
            return "duplicate"
        ok = await enqueue_rolling_summary(session_id, "manual")
        if not ok:
            await ConversationRepository.revert_summary_job_pending_to_idle(
                session_id, db_manager=self._db
            )
            logger.warning(
                "rolling_summary manual enqueue failed",
                extra={
                    "event": "rolling_summary.enqueue_failed",
                    "session_id": session_id,
                    "trigger_reason": "manual",
                },
            )
            return "queue_unavailable"
        logger.info(
            "rolling_summary manual enqueued",
            extra={
                "event": "rolling_summary.enqueued",
                "session_id": session_id,
                "trigger_reason": "manual",
            },
        )
        return "queued"

    async def execute_rolling_summary_job(self, session_id: str, reason: str) -> None:
        """后台消费者调用：抢占任务、生成摘要（优先 LLM）、落库并失效缓存。"""
        claimed = await ConversationRepository.try_claim_summary_job(
            session_id, db_manager=self._db
        )
        if not claimed:
            logger.info(
                "rolling_summary worker claim skipped",
                extra={
                    "event": "rolling_summary.worker_claim_skipped",
                    "session_id": session_id,
                    "worker_reason": reason,
                },
            )
            return
        try:
            text, anchor = await self._produce_rolling_summary_text(session_id, reason)
            applied = await ConversationRepository.apply_rolling_summary(
                session_id,
                text,
                summary_anchor_turn_index=anchor,
                db_manager=self._db,
            )
            if not applied:
                await ConversationRepository.reset_summary_job_to_idle(
                    session_id, db_manager=self._db
                )
                logger.warning(
                    "rolling_summary apply mismatch",
                    extra={
                        "event": "rolling_summary.apply_skipped",
                        "session_id": session_id,
                        "worker_reason": reason,
                    },
                )
                return
            await self._invalidate_history_cache(session_id)
        except Exception:
            logger.exception(
                "rolling_summary job failed",
                extra={
                    "event": "rolling_summary.job_failed",
                    "session_id": session_id,
                    "worker_reason": reason,
                },
            )
            await ConversationRepository.reset_summary_job_to_idle(session_id, db_manager=self._db)
            raise

    @staticmethod
    def _lc_message_text(msg: BaseMessage) -> str:
        raw = msg.content
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            parts: list[str] = []
            for block in raw:
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
        return str(raw).strip()

    @staticmethod
    def _rows_for_layered_batch(
        rows: list[AgentConversationMessage],
        anchor: int,
        batch: int,
    ) -> list[AgentConversationMessage]:
        """选取 ``turn_index`` 在 ``(anchor, anchor+batch]`` 内的消息行（user/assistant）。"""
        lo = int(anchor) + 1
        hi = int(anchor) + int(batch)
        return [m for m in rows if lo <= int(m.turn_index or 0) <= hi]

    def _format_messages_for_summary(self, rows: list[AgentConversationMessage]) -> str:
        lines: list[str] = []
        total = 0
        max_chars = 48_000
        for m in rows[-100:]:
            role = (m.role or "").strip()
            c = (m.content or "").strip()
            if not c:
                continue
            line = f"{role}: {c}"
            if total + len(line) > max_chars:
                lines.append(f"…（前文省略，共约 {len(rows)} 条消息）")
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines) if lines else "（无文本消息）"

    async def _rolling_summary_fallback_text(
        self,
        rows: list[AgentConversationMessage],
        reason: str,
    ) -> str:
        if not rows:
            return f"【滚动摘要·占位】会话暂无消息（reason={reason}）"
        lines: list[str] = []
        for m in rows[-12:]:
            role = (m.role or "").strip()
            c = (m.content or "").strip().replace("\n", " ")[:240]
            lines.append(f"- {role}: {c}")
        body = "\n".join(lines)
        return "【滚动摘要·占位】\n" + body

    async def _produce_rolling_summary_text(self, session_id: str, reason: str) -> tuple[str, int]:
        """链式分层摘要：在里程碑处仅合并「上一摘要 + 接下来 B 轮」；否则回退为全量摘录 + 锚点=max_u。"""
        rows = await ConversationRepository.list_messages_for_session(
            session_id,
            limit_last=120,
            db_manager=self._db,
        )
        sess = await self._fetch_session_row(session_id)
        max_u = await ConversationRepository.max_user_turn_index(session_id, db_manager=self._db)
        if sess is None:
            fb = await self._rolling_summary_fallback_text(rows, reason)
            return fb, max_u

        agent = await AgentRepository.get_by_id(int(sess.agent_id), db_manager=self._db)
        cfg = agent.config_json if agent is not None else None
        mem = AgentMemorySettingsParser().parse(cfg if isinstance(cfg, dict) else None)
        w, b = int(mem.min_tail_raw_rounds), int(mem.compress_batch_rounds)
        anchor = int(sess.summary_anchor_turn_index or 0)
        use_layered = layered_compression_milestone_reached(
            summary_anchor_turn_index=anchor,
            max_user_turn_index=max_u,
            min_tail_raw_rounds=w,
            compress_batch_rounds=b,
        )
        batch_rows = self._rows_for_layered_batch(rows, anchor, b) if use_layered else []
        if use_layered and batch_rows:
            transcript = self._format_messages_for_summary(batch_rows)
            new_anchor = anchor + b
        else:
            transcript = self._format_messages_for_summary(rows)
            new_anchor = max_u

        sys_mid = mem.summarization_sys_model_id
        if sys_mid is None and agent is not None:
            sys_mid = agent.sys_model_id
        if sys_mid is None:
            logger.info(
                "rolling summary: no sys_model, using fallback text",
                extra={"session_id": session_id},
            )
            fb = await self._rolling_summary_fallback_text(rows, reason)
            return fb, new_anchor

        try:
            snap_b = InvokeSnapshotBuilder(self._db)
            base = await snap_b.from_sys_model(int(sys_mid))
            snapshot = replace(
                base,
                hyperparameters=InferenceHyperparameters(
                    temperature=0.2,
                    max_tokens=2048,
                    top_p=0.9,
                ),
            )
            llm = RegistryDispatchingChatModelFactory().build(snapshot)
            prev = (sess.summary or "").strip()
            system = (
                "你是对话摘要助手。在「已有摘要」基础上合并「对话摘录」，输出一段简洁的中文摘要，"
                "保留关键事实与未决事项；不要复述本说明；不要编造。"
            )
            user = (
                f"【已有摘要】\n{prev or '（无）'}\n\n"
                f"【对话摘录】\n{transcript}\n\n"
                f"【触发原因】{reason}\n请只输出更新后的摘要正文，不要加标题或前缀。"
            )
            out = await llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)],
            )
            text = self._lc_message_text(out)
            if not text:
                fb = await self._rolling_summary_fallback_text(rows, reason)
                return fb, new_anchor
            return text, new_anchor
        except Exception:
            logger.exception(
                "rolling summary LLM failed, using fallback",
                extra={"session_id": session_id, "sys_model_id": sys_mid},
            )
            fb = await self._rolling_summary_fallback_text(rows, reason)
            return fb, new_anchor

    async def _invalidate_history_cache(self, session_id: str) -> None:
        """本会话下已组装轮次缓存全部失效（legacy key + cap 变体）。"""
        if self._redis is None:
            return
        await get_history_turns_cache().invalidate(self._redis, session_id)

    async def _assembled_turns_from_db(
        self,
        session_id: str,
        *,
        max_messages: int,
    ) -> list[HistoryTurnSlot]:
        rows = await ConversationRepository.list_messages_for_session(
            session_id,
            limit_last=max_messages,
            db_manager=self._db,
        )
        return HistoryTurnAssembler().assemble(rows)

    async def get_assembled_turns_cached(
        self,
        session_id: str,
        *,
        max_history_rounds_cap: int = DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    ) -> list[HistoryTurnSlot]:
        """读穿缓存：命中则直接返回；否则从 DB 组装，可选写回 Redis。

        ``max_history_rounds_cap`` 决定从 DB 拉取的最近消息行数上界（与 Agent 记忆配置一致），避免长会话全表扫描。
        未配置 Redis 时行为等同于每次直连 DB 组装。
        """
        _cap = max(1, min(max_history_rounds_cap, 200))
        row_limit = min(4 * _cap + 64, 2500)
        cache = get_history_turns_cache()

        if self._redis is not None:
            hit = await cache.get(self._redis, session_id, max_history_rounds_cap)
            if hit is not None:
                return hit

        turns = await self._assembled_turns_from_db(session_id, max_messages=row_limit)

        if self._redis is not None:
            await cache.set(self._redis, session_id, turns, max_history_rounds_cap)

        return turns

    async def create_session(self, body: ConversationSessionCreateBody) -> ConversationSessionOut:
        """新建会话；校验 Agent 存在后写入仓储。"""
        session_id = uuid.uuid4().hex
        title = body.title or "新对话"
        agent = await AgentRepository.get_by_id(body.agent_id, db_manager=self._db)
        if agent is None:
            raise ReferencedAgentNotFoundError()

        row = await ConversationRepository.create_session(
            session_id=session_id,
            agent_id=body.agent_id,
            title=title[:255],
            db_manager=self._db,
        )
        return ConversationSessionOut.model_validate(row)

    async def get_session_detail(
        self,
        session_id: str,
        *,
        include_deleted: bool = False,
        page: int = 1,
        page_size: int = CONVERSATION_DETAIL_MESSAGES_DEFAULT_PAGE_SIZE,
        messages_agent_id: int | None = None,
        include_message_metadata: bool = True,
    ) -> ConversationDetailData | None:
        """会话详情：元数据 + 消息分页（第 1 页为最近 ``page_size`` 条，页内时间正序）。

        ``messages_agent_id``：仅返回并统计 ``conversation_message.agent_id`` 匹配的消息；
        工作台与会话共享落库时用于历史 UI 不混入子 Agent 行。
        ``include_message_metadata``：为 False 时不读取各消息 ``metadata`` JSON 列且响应中恒为 null；
        执行过程等请使用单条 ``get_message_execution``。
        """
        sess = await ConversationRepository.get_session_row(
            session_id,
            include_deleted=include_deleted,
            db_manager=self._db,
        )
        if sess is None:
            return None
        page, page_size = normalize_conversation_messages_pagination(page, page_size)
        total = await ConversationRepository.count_messages_for_session(
            session_id,
            message_agent_id=messages_agent_id,
            db_manager=self._db,
        )
        defer_meta = not include_message_metadata
        messages = await ConversationRepository.list_messages_page_for_session(
            session_id,
            page=page,
            page_size=page_size,
            message_agent_id=messages_agent_id,
            defer_message_metadata=defer_meta,
            db_manager=self._db,
        )
        return ConversationDetailData(
            session=ConversationSessionOut.model_validate(sess),
            messages=[
                _message_out_list_row(m, include_message_metadata=include_message_metadata)
                for m in messages
            ],
            messages_meta=PageMeta(page=page, page_size=page_size, total=total),
        )

    async def get_message_execution(
        self,
        session_id: str,
        message_id: int,
    ) -> ConversationMessageExecutionOut:
        """返回指定助手消息 metadata 中的 ``process_trace`` / ``tool_history`` 等执行快照。"""
        if self._db is None:
            raise ConversationDomainError("数据库不可用", status_code=503)
        row = await ConversationRepository.get_message_in_session(
            session_id,
            message_id,
            db_manager=self._db,
        )
        if row is None:
            raise ConversationDomainError("消息不存在或与会话不符", status_code=404)
        if row.role != "assistant":
            raise ConversationDomainError("仅助手消息可查询执行过程", status_code=400)
        meta = row.message_metadata if isinstance(row.message_metadata, dict) else {}
        pt = meta.get("process_trace")
        th = meta.get("tool_history")
        tt = meta.get("thinking_text")
        if pt is not None and not isinstance(pt, list):
            pt = None
        if th is not None:
            if not isinstance(th, list):
                th = None
            else:
                th = [x for x in th if isinstance(x, dict)]
        thinking: str | None = None
        if isinstance(tt, str) and tt.strip():
            thinking = tt.strip()
        return ConversationMessageExecutionOut(
            message_id=row.id,
            session_id=row.session_id,
            agent_id=row.agent_id,
            role="assistant",
            process_trace=pt,
            tool_history=th,
            thinking_text=thinking,
        )

    async def list_sessions(
        self,
        *,
        page: int = 1,
        page_size: int = CONVERSATION_LIST_DEFAULT_PAGE_SIZE,
        agent_id: int | None = None,
    ) -> ConversationSessionListData:
        """分页列出会话；``page`` / ``page_size`` 经 :func:`~app.domain.conversation.normalize_conversation_list_pagination` 归一化。"""
        page, page_size = normalize_conversation_list_pagination(page, page_size)
        items, total = await ConversationRepository.list_sessions_page(
            page=page,
            page_size=page_size,
            agent_id=agent_id,
            db_manager=self._db,
        )
        return ConversationSessionListData(
            items=[ConversationSessionOut.model_validate(r) for r in items],
            meta=PageMeta(page=page, page_size=page_size, total=total),
        )

    async def require_active_session_for_agent(
        self,
        session_id: str,
        *,
        agent_id: int,
    ) -> None:
        """供 Agent 调用前校验：会话须存在、未软删、活跃，且归属 ``agent_id``。

        Raises:
            SessionNotFoundError: 会话不存在或已软删。
            SessionNotWritableError: 会话已关闭。
            SessionAgentMismatchError: ``agent_id`` 与会话绑定不一致。
        """
        sess = await self._fetch_session_row(session_id)
        if sess is None:
            raise SessionNotFoundError()
        if sess.status != CONVERSATION_SESSION_ACTIVE:
            raise SessionNotWritableError()
        if sess.agent_id != agent_id:
            raise SessionAgentMismatchError()

    async def load_history_turn_slots_for_session(
        self,
        session_id: str,
        *,
        max_history_rounds_cap: int = DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    ) -> list[HistoryTurnSlot]:
        """只读：归并为 ``HistoryTurnSlot``（不含与当前句去重）；命中 Redis 则跳过 DB。"""
        return await self.get_assembled_turns_cached(
            session_id,
            max_history_rounds_cap=max_history_rounds_cap,
        )

    async def prepare_history_turns_for_model_invoke(
        self,
        session_id: str,
        current_user_message: str,
        *,
        max_history_rounds_cap: int = DEFAULT_MAX_HISTORY_ROUNDS_CAP,
    ) -> list[HistoryTurnSlot]:
        """Agent 本轮：缓存的已组装轮次 + 与 ``current_user_message`` 去重。"""
        assembled = await self.get_assembled_turns_cached(
            session_id,
            max_history_rounds_cap=max_history_rounds_cap,
        )
        return HistoryTurnDeduper().dedupe_last_if_same_as_current(
            assembled,
            current_user_message,
        )

    async def append_message(
        self,
        session_id: str,
        body: ConversationMessageAppendBody,
    ) -> ConversationMessageOut | None:
        """追加消息；会话不可用或已关闭时返回 ``None``（与 HTTP 400 语义配合）。"""
        sess = await self._fetch_session_row(session_id)
        if not _session_allows_append(sess):
            return None

        aid = body.agent_id if body.agent_id is not None else int(sess.agent_id)
        row = await ConversationRepository.append_message(
            session_id=session_id,
            agent_id=aid,
            role=body.role,
            content=body.content,
            content_type=body.content_type,
            metadata=body.metadata,
            tokens=body.tokens,
            db_manager=self._db,
        )
        await self._invalidate_history_cache(session_id)
        return ConversationMessageOut.model_validate(row)

    async def patch_session(
        self,
        session_id: str,
        body: ConversationSessionPatchBody,
    ) -> ConversationSessionOut | None:
        """更新标题或状态；无有效变更字段时仅查询当前会话并返回（不写库、不刷缓存）。"""
        if body.title is None and body.status is None:
            sess = await self._fetch_session_row(session_id)
            return ConversationSessionOut.model_validate(sess) if sess else None

        status = body.status

        ok = await ConversationRepository.update_session(
            session_id=session_id,
            title=body.title[:255] if body.title is not None else None,
            status=status,
            db_manager=self._db,
        )
        if not ok:
            return None
        await self._invalidate_history_cache(session_id)
        sess = await self._fetch_session_row(session_id)
        return ConversationSessionOut.model_validate(sess) if sess else None

    async def soft_delete_session(self, session_id: str) -> bool:
        """软删除；成功时失效该会话轮次缓存。"""
        ok = await ConversationRepository.soft_delete_session(
            session_id=session_id,
            db_manager=self._db,
        )
        if ok:
            await self._invalidate_history_cache(session_id)
        return ok

    async def create_session_for_invoke(
        self,
        *,
        agent_id: int,
        user_message: str,
    ) -> ConversationSessionOut:
        """新对话：用首条用户消息摘要作为标题并绑定 ``agent_id``。"""
        if user_message:
            title = user_message[:200] + ("…" if len(user_message) > 200 else "")
        else:
            title = "新对话"
        body = ConversationSessionCreateBody(agent_id=agent_id, title=title)
        return await self.create_session(body)


def create_rolling_summary_worker(
    db_manager: SQLAlchemyDatabaseManager,
    redis_client: redis.Redis | None,
    *,
    max_queue_size: int = 512,
) -> RollingSummaryWorker:
    """进程内滚动摘要消费者：绑定 :class:`ConversationService` 与 ``execute_rolling_summary_job``（由 lifespan 启动）。"""
    svc = ConversationService(db_manager, redis=redis_client)

    async def on_job(job: RollingSummaryJob) -> None:
        await svc.execute_rolling_summary_job(job["session_id"], job["reason"])

    return RollingSummaryWorker(
        on_job=on_job,
        max_queue_size=max_queue_size,
        consumer_task_name="rolling-summary-consumer",
    )
