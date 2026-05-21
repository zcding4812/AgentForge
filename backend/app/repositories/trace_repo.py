"""``trace_run`` / ``trace_span`` / ``trace_span_log`` 读写；供链路服务与 ``tracing/sinks`` 落库。"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    TRACE_LIST_DEFAULT_PAGE_SIZE,
    TraceRunStatus,
    TraceSpanStatus,
)
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.trace_mod import TraceRun, TraceSpan, TraceSpanLog
from app.repositories.base_repo import BaseRepository

logger = logging.getLogger(__name__)


class TraceRepository(BaseRepository[TraceRun]):
    model = TraceRun

    @classmethod
    async def create_trace_run(
        cls,
        *,
        trace_id: str,
        trace_name: str,
        http_path: str,
        http_method: str = "GET",
        service_name: str = "fastapi-backend",
        session_id: str | None = None,
        agent_task_id: str | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> TraceRun | None:
        if await cls.exists_by(trace_id=trace_id, db_manager=db_manager):
            logger.info("追踪记录已存在 | trace_id: %s", trace_id)
            return None

        async def _write(session: AsyncSession) -> TraceRun:
            instance = cls.model(
                trace_id=trace_id,
                trace_name=trace_name,
                http_path=http_path,
                http_method=http_method.upper()[:16],
                service_name=service_name,
                session_id=session_id,
                agent_task_id=agent_task_id,
                status=TraceRunStatus.RUNNING.value,
                started_at=datetime.now(),
            )
            session.add(instance)
            await session.flush()
            return instance

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def create_trace_run_with_root_span(
        cls,
        *,
        trace_id: str,
        trace_name: str,
        http_path: str,
        http_method: str = "GET",
        service_name: str = "fastapi-backend",
        session_id: str | None = None,
        agent_task_id: str | None = None,
        span_id: str,
        span_name: str,
        span_type: str = "fastapi",
        component: str | None = "fastapi",
        depth: int = 0,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> None:
        """一次事务写入 trace_run + 根 span，避免 HTTP 入口两次 ``run_write`` 往返叠加延迟。"""

        async def _write(session: AsyncSession) -> None:
            existing = await session.scalar(select(cls.model).where(cls.model.trace_id == trace_id))
            if existing:
                logger.info("追踪记录已存在 | trace_id: %s", trace_id)
                return

            tr = cls.model(
                trace_id=trace_id,
                trace_name=trace_name,
                http_path=http_path,
                http_method=http_method.upper()[:16],
                service_name=service_name,
                session_id=session_id,
                agent_task_id=agent_task_id,
                status=TraceRunStatus.RUNNING.value,
                started_at=datetime.now(),
            )
            session.add(tr)
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                span_name=span_name,
                span_type=span_type,
                component=component,
                depth=depth,
                tags=None,
                started_at=datetime.now(),
                status=TraceSpanStatus.IN_PROGRESS,
            )
            session.add(span)

        await cls.run_write(_write, db_manager)

    @classmethod
    async def count_trace_runs(
        cls,
        *,
        trace_id: str | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> int:
        async def _read(session: AsyncSession) -> int:
            q = select(func.count()).select_from(cls.model)
            if trace_id:
                q = q.where(cls.model.trace_id == trace_id)
            return int(await session.scalar(q) or 0)

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def list_trace_runs(
        cls,
        *,
        trace_id: str | None = None,
        offset: int = 0,
        limit: int = TRACE_LIST_DEFAULT_PAGE_SIZE,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[TraceRun]:
        async def _read(session: AsyncSession) -> list[TraceRun]:
            q = select(cls.model)
            if trace_id:
                q = q.where(cls.model.trace_id == trace_id)
            q = q.order_by(cls.model.started_at.desc())
            if offset > 0:
                q = q.offset(offset)
            if limit > 0:
                q = q.limit(limit)
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def aggregate_trace_run_stats(
        cls,
        *,
        trace_id: str | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        async def _read(session: AsyncSession) -> tuple[int, int, int, int, int, int]:
            agg = select(
                func.count().label("total"),
                func.coalesce(
                    func.sum(
                        case((cls.model.status == TraceRunStatus.SUCCESS.value, 1), else_=0),
                    ),
                    0,
                ).label("success"),
                func.coalesce(
                    func.sum(
                        case((cls.model.status == TraceRunStatus.FAILED.value, 1), else_=0),
                    ),
                    0,
                ).label("failed"),
                func.coalesce(
                    func.sum(
                        case((cls.model.status == TraceRunStatus.RUNNING.value, 1), else_=0),
                    ),
                    0,
                ).label("running"),
                func.avg(cls.model.duration_ms).label("avg_ms"),
            ).select_from(cls.model)
            if trace_id:
                agg = agg.where(cls.model.trace_id == trace_id)
            row = (await session.execute(agg)).one()
            total = int(row.total or 0)
            success = int(row.success or 0)
            failed = int(row.failed or 0)
            running = int(row.running or 0)
            avg_ms = int(round(row.avg_ms)) if row.avg_ms is not None else 0

            dq = select(cls.model.duration_ms).where(cls.model.duration_ms.isnot(None))
            if trace_id:
                dq = dq.where(cls.model.trace_id == trace_id)
            dq = dq.order_by(cls.model.duration_ms.asc())
            durations = [int(x) for x in (await session.scalars(dq)).all() if x is not None]
            p95_ms = durations[int((len(durations) - 1) * 0.95)] if durations else 0

            return total, success, failed, running, avg_ms, p95_ms

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def finish_trace_run(
        cls,
        *,
        trace_id: str,
        status: str,
        duration_ms: int | None = None,
        ended_at: datetime | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> None:
        async def _write(session: AsyncSession) -> None:
            tr = await session.scalar(select(cls.model).where(cls.model.trace_id == trace_id))
            if not tr:
                return

            tr.status = status
            tr.ended_at = ended_at or datetime.now()
            if duration_ms is not None:
                tr.duration_ms = duration_ms

        await cls.run_write(_write, db_manager)

    @classmethod
    async def create_span(
        cls,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 0,
        tags: dict | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> TraceSpan:
        async def _write(session: AsyncSession) -> TraceSpan:
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_name=span_name,
                span_type=span_type,
                component=component,
                depth=depth,
                tags=tags,
                started_at=datetime.now(),
                status=TraceSpanStatus.IN_PROGRESS,
            )
            session.add(span)
            await session.flush()
            return span

        return await cls.run_write(_write, db_manager)

    @classmethod
    async def finish_span(
        cls,
        *,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> None:
        async def _write(session: AsyncSession) -> None:
            sp = await session.scalar(select(TraceSpan).where(TraceSpan.span_id == span_id))
            if not sp:
                return

            # duration_ms 来自 perf_counter，与「finish 时刻的 now()」混用反推 started_at 时，
            # 各 span 的 finish 先后不一，会出现子条从 0ms 画起、父条从中间画起等父子时间轴错位。
            # 结束时间一律相对 create_span 写入的 started_at 延伸，保持与 duration 同一口径。
            sp.status = status
            sp.duration_ms = duration_ms
            sp.ended_at = sp.started_at + timedelta(milliseconds=duration_ms)
            sp.error_message = error_message
            sp.is_slow = duration_ms > 1000

        await cls.run_write(_write, db_manager)

    @classmethod
    async def list_trace_spans(
        cls,
        *,
        trace_id: str,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[TraceSpan]:
        async def _read(session: AsyncSession) -> list[TraceSpan]:
            q = (
                select(TraceSpan)
                .where(TraceSpan.trace_id == trace_id)
                .order_by(TraceSpan.started_at, TraceSpan.id)
            )
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def append_span_log(
        cls,
        *,
        trace_id: str,
        span_id: str,
        log_level: str,
        event_name: str,
        message: str,
        payload: dict | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> None:
        async def _write(session: AsyncSession) -> None:
            max_seq = await session.scalar(
                select(func.coalesce(func.max(TraceSpanLog.seq_no), 0)).where(
                    TraceSpanLog.trace_id == trace_id,
                    TraceSpanLog.span_id == span_id,
                )
            )
            next_seq = int(max_seq or 0) + 1
            log = TraceSpanLog(
                trace_id=trace_id,
                span_id=span_id,
                log_level=log_level,
                event_name=event_name,
                message=message,
                payload=payload,
                occurred_at=datetime.now(),
                seq_no=next_seq,
            )
            session.add(log)

        await cls.run_write(_write, db_manager)

    @classmethod
    async def list_span_logs_by_trace(
        cls,
        *,
        trace_id: str,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> list[TraceSpanLog]:
        async def _read(session: AsyncSession) -> list[TraceSpanLog]:
            q = (
                select(TraceSpanLog)
                .where(TraceSpanLog.trace_id == trace_id)
                .order_by(TraceSpanLog.occurred_at, TraceSpanLog.seq_no)
            )
            return list((await session.scalars(q)).all())

        return await cls.run_read(_read, db_manager)
