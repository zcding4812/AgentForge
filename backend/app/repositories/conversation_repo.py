"""``conversation_session`` / ``conversation_message`` 表访问。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Date, case, cast, desc, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.constants.conversation import (
    CONVERSATION_SESSION_ACTIVE,
    LIST_MESSAGES_DEFAULT_LIMIT_LAST,
    SUMMARY_JOB_STATUS_IDLE,
    SUMMARY_JOB_STATUS_PENDING,
    SUMMARY_JOB_STATUS_RUNNING,
)
from app.domain.conversation import (
    RollingSummaryTriggerMetrics,
    estimate_user_content_tokens,
    normalize_conversation_list_pagination,
)
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.conversation_mod import AgentConversationMessage, AgentConversationSession
from app.repositories.base_repo import BaseRepository


class ConversationRepository(BaseRepository[AgentConversationSession]):
    model = AgentConversationSession

    @classmethod
    async def create_session(
        cls,
        *,
        session_id: str,
        agent_id: int,
        title: str,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> AgentConversationSession:
        return await cls.create(
            db_manager=db_manager,
            session_id=session_id,
            agent_id=agent_id,
            title=title,
            status=CONVERSATION_SESSION_ACTIVE,
            deleted_at=None,
        )

    @classmethod
    async def get_session_row(
        cls,
        session_id: str,
        *,
        include_deleted: bool = False,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> AgentConversationSession | None:
        async def _read(session: AsyncSession) -> AgentConversationSession | None:
            q = select(cls.model).where(cls.model.session_id == session_id)
            if not include_deleted:
                q = q.where(cls.model.deleted_at.is_(None))
            return (await session.scalars(q)).first()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_sessions_page(
        cls,
        *,
        page: int = 1,
        page_size: int = 20,
        agent_id: int | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> tuple[list[AgentConversationSession], int]:
        normalized_page, normalized_page_size = normalize_conversation_list_pagination(
            page, page_size
        )

        async def _read(session: AsyncSession) -> tuple[list[AgentConversationSession], int]:
            filt = cls.model.deleted_at.is_(None)
            if agent_id is not None:
                # 会话行 ``agent_id`` 为「创建者」；同 session 下子 Agent 通过 workbench 共享会话落库时，
                # 消息行 ``conversation_message.agent_id`` 为子 Agent，须一并出现在该 Agent 的历史列表中。
                has_participant_message = exists(
                    select(AgentConversationMessage.id).where(
                        AgentConversationMessage.session_id == cls.model.session_id,
                        AgentConversationMessage.agent_id == agent_id,
                    )
                )
                filt = filt & (or_(cls.model.agent_id == agent_id, has_participant_message))
            count_q = select(func.count()).select_from(cls.model).where(filt)
            total = int(await session.scalar(count_q) or 0)
            offset = (normalized_page - 1) * normalized_page_size
            data_q = (
                select(cls.model)
                .where(filt)
                .order_by(cls.model.updated_at.desc())
                .offset(offset)
                .limit(normalized_page_size)
            )
            items = list((await session.scalars(data_q)).all())
            return items, total

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def append_message(
        cls,
        *,
        session_id: str,
        agent_id: int,
        role: str,
        content: str,
        content_type: str = "text",
        metadata: dict[str, Any] | None = None,
        tokens: int | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> AgentConversationMessage:
        now = datetime.now()

        async def _write(session: AsyncSession) -> AgentConversationMessage:
            rl = (role or "").strip().lower()
            turn_idx: int = 0
            reply_id: int | None = None
            # 时间轴上最近一条 user（须与当前行同一 agent_id）：同会话多 Agent 并行写入时，若按「全会话
            # 全局 last user」取 reply_message_id，子 A 的 assistant 会错挂到子 B 刚插入的 user，出现
            # 「同一提问 id 被回复两次」与轮次错乱。
            last_user = (
                await session.scalars(
                    select(AgentConversationMessage)
                    .where(
                        AgentConversationMessage.session_id == session_id,
                        AgentConversationMessage.agent_id == agent_id,
                        func.lower(AgentConversationMessage.role) == "user",
                    )
                    .order_by(AgentConversationMessage.id.desc())
                    .limit(1)
                )
            ).first()
            if rl == "user":
                turn_idx = int(last_user.turn_index) + 1 if last_user is not None else 1
            elif rl == "assistant" and last_user is not None:
                turn_idx = int(last_user.turn_index)
                reply_id = last_user.id
            # 无 user 的 assistant / 非 user-assistant 角色：保持 turn_index=0、reply_id=None

            tok: int | None = None
            if rl == "user":
                tok = tokens if tokens is not None else estimate_user_content_tokens(content)
            elif rl == "assistant":
                tok = tokens
            msg = AgentConversationMessage(
                session_id=session_id,
                agent_id=agent_id,
                role=role,
                content=content,
                content_type=content_type,
                message_metadata=metadata,
                created_at=now,
                turn_index=turn_idx,
                reply_message_id=reply_id,
                tokens=tok,
            )
            session.add(msg)
            await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                )
                .values(updated_at=now)
            )
            await session.flush()
            await session.refresh(msg)
            return msg

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def get_message_in_session(
        cls,
        session_id: str,
        message_id: int,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> AgentConversationMessage | None:
        """按主键与会话 ID 取单行（用于按消息拉取执行过程等）。"""

        async def _read(session: AsyncSession) -> AgentConversationMessage | None:
            q = select(AgentConversationMessage).where(
                AgentConversationMessage.id == message_id,
                AgentConversationMessage.session_id == session_id,
            )
            return (await session.scalars(q)).first()

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def count_messages_for_session(
        cls,
        session_id: str,
        *,
        message_agent_id: int | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> int:
        """会话内消息总行数。

        ``message_agent_id`` 非空时仅统计 ``conversation_message.agent_id`` 匹配的行
        （工作台共享会话下用于 UI 只展示编排 Agent 侧消息）。
        """

        async def _read(session: AsyncSession) -> int:
            filters = [AgentConversationMessage.session_id == session_id]
            if message_agent_id is not None:
                filters.append(AgentConversationMessage.agent_id == message_agent_id)
            q = select(func.count()).select_from(AgentConversationMessage).where(*filters)
            return int((await session.execute(q)).scalar_one())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_messages_page_for_session(
        cls,
        session_id: str,
        *,
        page: int,
        page_size: int,
        message_agent_id: int | None = None,
        defer_message_metadata: bool = False,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[AgentConversationMessage]:
        """消息分页：第 1 页为时间轴上最近 ``page_size`` 条（页内按时间正序）；第 2 页为更早的一批。

        ``message_agent_id`` 非空时仅返回该 ``agent_id`` 的消息行（与 :meth:`count_messages_for_session` 一致）。
        ``defer_message_metadata``：为 True 时不加载 ``metadata`` JSON 列（列表页减负；单条 execution 再取）。
        """

        p = max(1, page)
        ps = max(1, page_size)
        offset = (p - 1) * ps

        async def _read(session: AsyncSession) -> list[AgentConversationMessage]:
            filters = [AgentConversationMessage.session_id == session_id]
            if message_agent_id is not None:
                filters.append(AgentConversationMessage.agent_id == message_agent_id)
            q = (
                select(AgentConversationMessage)
                .where(*filters)
                .order_by(
                    desc(AgentConversationMessage.created_at),
                    desc(AgentConversationMessage.id),
                )
                .offset(offset)
                .limit(ps)
            )
            if defer_message_metadata:
                q = q.options(defer(AgentConversationMessage.message_metadata))
            rows = list((await session.scalars(q)).all())
            rows.reverse()
            return rows

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_messages_for_session(
        cls,
        session_id: str,
        *,
        limit_last: int = LIST_MESSAGES_DEFAULT_LIMIT_LAST,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[AgentConversationMessage]:
        """列出会话消息：取时间轴上最近 ``limit_last`` 条，按时间正序返回。"""

        async def _read(session: AsyncSession) -> list[AgentConversationMessage]:
            q = (
                select(AgentConversationMessage)
                .where(AgentConversationMessage.session_id == session_id)
                .order_by(
                    desc(AgentConversationMessage.created_at),
                    desc(AgentConversationMessage.id),
                )
                .limit(limit_last)
            )
            rows = list((await session.scalars(q)).all())
            rows.reverse()
            return rows

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def rolling_summary_trigger_metrics(
        cls,
        session_id: str,
        *,
        summary_anchor_turn_index: int,
        limit_last: int = LIST_MESSAGES_DEFAULT_LIMIT_LAST,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> RollingSummaryTriggerMetrics:
        """最近 ``limit_last`` 条消息的 token 估算；用于摘要触发比例与增量门槛。"""

        rows = await cls.list_messages_for_session(
            session_id,
            limit_last=limit_last,
            db_manager=db_manager,
        )
        total_msg = 0
        after_anchor = 0
        max_u = 0
        anchor = int(summary_anchor_turn_index)
        for m in rows:
            raw_t = m.tokens
            est = int(raw_t) if raw_t is not None else estimate_user_content_tokens(m.content or "")
            est = max(0, est)
            total_msg += est
            rl = (m.role or "").strip().lower()
            if rl == "user":
                max_u = max(max_u, int(m.turn_index or 0))
            if int(m.turn_index or 0) > anchor:
                after_anchor += est
        return RollingSummaryTriggerMetrics(
            total_message_tokens_estimated=total_msg,
            max_user_turn_index=max_u,
            tokens_after_anchor=after_anchor,
        )

    @classmethod
    async def max_user_turn_index(
        cls,
        session_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> int:
        """当前会话 user 行的最大 ``turn_index``。"""

        async def _read(session: AsyncSession) -> int:
            q = select(func.max(AgentConversationMessage.turn_index)).where(
                AgentConversationMessage.session_id == session_id,
                func.lower(AgentConversationMessage.role) == "user",
            )
            v = await session.scalar(q)
            return int(v or 0)

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def update_session(
        cls,
        *,
        session_id: str,
        title: str | None = None,
        status: int | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        async def _write(session: AsyncSession) -> bool:
            values: dict[str, Any] = {"updated_at": datetime.now()}
            if title is not None:
                values["title"] = title
            if status is not None:
                values["status"] = status
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                )
                .values(**values)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def try_mark_summary_job_pending(
        cls,
        session_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """仅当当前为 idle 时置为 pending，用于入队前去重。"""

        async def _write(session: AsyncSession) -> bool:
            now = datetime.now()
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                    AgentConversationSession.summary_job_status == SUMMARY_JOB_STATUS_IDLE,
                )
                .values(summary_job_status=SUMMARY_JOB_STATUS_PENDING, updated_at=now)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def revert_summary_job_pending_to_idle(
        cls,
        session_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """入队失败等场景：仅 pending → idle。"""

        async def _write(session: AsyncSession) -> bool:
            now = datetime.now()
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                    AgentConversationSession.summary_job_status == SUMMARY_JOB_STATUS_PENDING,
                )
                .values(summary_job_status=SUMMARY_JOB_STATUS_IDLE, updated_at=now)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def try_claim_summary_job(
        cls,
        session_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """将 ``summary_job_status`` 从 idle/pending 置为 running；未命中行则 ``False``（已在跑或会话不存在）。"""

        async def _write(session: AsyncSession) -> bool:
            now = datetime.now()
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                    AgentConversationSession.summary_job_status.in_(
                        [SUMMARY_JOB_STATUS_IDLE, SUMMARY_JOB_STATUS_PENDING]
                    ),
                )
                .values(summary_job_status=SUMMARY_JOB_STATUS_RUNNING, updated_at=now)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def apply_rolling_summary(
        cls,
        session_id: str,
        summary_text: str,
        *,
        summary_anchor_turn_index: int,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """在 ``running`` 状态下写入摘要并递增 ``summary_version``，然后置 idle。"""

        async def _write(session: AsyncSession) -> bool:
            now = datetime.now()
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                    AgentConversationSession.summary_job_status == SUMMARY_JOB_STATUS_RUNNING,
                )
                .values(
                    summary=summary_text,
                    summary_version=AgentConversationSession.summary_version + 1,
                    summary_job_status=SUMMARY_JOB_STATUS_IDLE,
                    summary_anchor_turn_index=summary_anchor_turn_index,
                    updated_at=now,
                )
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def reset_summary_job_to_idle(
        cls,
        session_id: str,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        """摘要失败时将 ``running`` 恢复为 idle，不修改 ``summary`` / ``summary_version``。"""

        async def _write(session: AsyncSession) -> bool:
            now = datetime.now()
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                    AgentConversationSession.summary_job_status == SUMMARY_JOB_STATUS_RUNNING,
                )
                .values(summary_job_status=SUMMARY_JOB_STATUS_IDLE, updated_at=now)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def sum_tokens_by_role_by_day(
        cls,
        *,
        since_day: date,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[tuple[date, int, int]]:
        """按自然日聚合 ``user`` / ``assistant`` 且 ``tokens`` 非空的用量（运维看板堆叠图）。"""

        cutoff = datetime.combine(since_day, datetime.min.time())
        role_l = func.lower(AgentConversationMessage.role)

        async def _read(session: AsyncSession) -> list[tuple[date, int, int]]:
            day_col = cast(AgentConversationMessage.created_at, Date)
            user_sum = func.coalesce(
                func.sum(case((role_l == "user", AgentConversationMessage.tokens), else_=0)),
                0,
            )
            asst_sum = func.coalesce(
                func.sum(case((role_l == "assistant", AgentConversationMessage.tokens), else_=0)),
                0,
            )
            stmt = (
                select(day_col, user_sum, asst_sum)
                .where(
                    AgentConversationMessage.tokens.isnot(None),
                    AgentConversationMessage.created_at >= cutoff,
                )
                .group_by(day_col)
                .order_by(day_col.asc())
            )
            rows = (await session.execute(stmt)).all()
            out: list[tuple[date, int, int]] = []
            for row in rows:
                d_raw = row[0]
                if isinstance(d_raw, datetime):
                    d_norm = d_raw.date()
                elif isinstance(d_raw, date):
                    d_norm = d_raw
                else:
                    continue
                out.append((d_norm, int(row[1] or 0), int(row[2] or 0)))
            return out

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def soft_delete_session(
        cls,
        *,
        session_id: str,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> bool:
        now = datetime.now()

        async def _write(session: AsyncSession) -> bool:
            res = await session.execute(
                update(AgentConversationSession)
                .where(
                    AgentConversationSession.session_id == session_id,
                    AgentConversationSession.deleted_at.is_(None),
                )
                .values(deleted_at=now, updated_at=now)
            )
            return res.rowcount > 0

        return await cls.run_write(_write, db_manager)
