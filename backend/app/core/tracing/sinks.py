from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.context import RequestContext
from app.core.logger import get_logger
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.repositories.trace_repo import TraceRepository

logger = get_logger(__name__)


class TraceStartSink(ABC):
    @abstractmethod
    def on_trace_start(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        trace_name: str,
        method: str,
        path: str,
    ) -> None: ...

    async def on_trace_start_async(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        trace_name: str,
        method: str,
        path: str,
    ) -> None:
        self.on_trace_start(
            trace_id=trace_id,
            request_id=request_id,
            span_id=span_id,
            trace_name=trace_name,
            method=method,
            path=path,
        )


class TraceFinishSink(ABC):
    @abstractmethod
    def on_trace_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None: ...

    async def on_trace_finish_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        self.on_trace_finish(
            trace_id=trace_id,
            span_id=span_id,
            status=status,
            duration_ms=duration_ms,
            status_code=status_code,
        )


class SpanStartSink(ABC):
    @abstractmethod
    def on_span_start(
        self,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None,
        component: str | None,
        depth: int,
        tags: dict | None,
    ) -> None: ...

    async def on_span_start_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None,
        component: str | None,
        depth: int,
        tags: dict | None,
    ) -> None:
        self.on_span_start(
            trace_id=trace_id,
            span_id=span_id,
            span_name=span_name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            component=component,
            depth=depth,
            tags=tags,
        )


class SpanFinishSink(ABC):
    @abstractmethod
    def on_span_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None,
    ) -> None: ...

    async def on_span_finish_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        self.on_span_finish(
            trace_id=trace_id,
            span_id=span_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )


class SpanLogSink(ABC):
    @abstractmethod
    def on_span_log(
        self,
        *,
        trace_id: str,
        span_id: str,
        event_name: str,
        message: str,
        payload: dict | None,
        log_level: str,
    ) -> None: ...

    async def on_span_log_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        event_name: str,
        message: str,
        payload: dict | None,
        log_level: str,
    ) -> None:
        self.on_span_log(
            trace_id=trace_id,
            span_id=span_id,
            event_name=event_name,
            message=message,
            payload=payload,
            log_level=log_level,
        )


class BaseTraceSink(TraceStartSink, TraceFinishSink, SpanStartSink, SpanFinishSink, SpanLogSink):
    """完整 Trace Sink（组合各生命周期接口）。"""


_db_retry_async = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(SQLAlchemyError),
    reraise=True,
)


def _resolve_sink_db_manager() -> SQLAlchemyDatabaseManager | None:
    req = RequestContext.get_request()
    if req is None:
        return None
    return getattr(req.app.state, "db_manager", None)


def _schedule_trace_db_on_main_loop(coro) -> None:
    """
    同步 Trace 回调里调度异步仓储：必须提交到 **应用主事件循环**。
    禁止在线程内 ``asyncio.run``，否则与全局 ``AsyncEngine`` 连接池跨 loop，会触发
    "Future attached to a different loop"。
    """
    req = RequestContext.get_request()
    if req is None:
        return
    loop = getattr(req.app.state, "asyncio_loop", None)
    if loop is None or not loop.is_running():
        logger.warning("Trace DB sink skipped: asyncio_loop 不可用或未运行")
        return
    fut = asyncio.run_coroutine_threadsafe(coro, loop)

    def _done(f) -> None:
        try:
            f.result()
        except Exception:
            logger.exception("Trace DB sink 异步任务失败")

    fut.add_done_callback(_done)


class DbTraceSink(BaseTraceSink):
    def on_trace_start(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        trace_name: str,
        method: str,
        path: str,
    ) -> None:
        dm = _resolve_sink_db_manager()
        if dm is None:
            return

        async def _go() -> None:
            await TraceRepository.create_trace_run_with_root_span(
                trace_id=trace_id,
                trace_name=trace_name,
                http_path=path,
                http_method=method,
                span_id=span_id,
                span_name=trace_name,
                span_type="fastapi",
                component="fastapi",
                depth=0,
                db_manager=dm,
            )

        _schedule_trace_db_on_main_loop(_go())

    def on_trace_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        dm = _resolve_sink_db_manager()
        if dm is None:
            return

        async def _go() -> None:
            await TraceRepository.finish_span(
                span_id=span_id,
                status=status,
                duration_ms=duration_ms,
                db_manager=dm,
            )
            await TraceRepository.finish_trace_run(
                trace_id=trace_id,
                status=status,
                duration_ms=duration_ms,
                db_manager=dm,
            )

        _schedule_trace_db_on_main_loop(_go())

    def on_span_start(
        self,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None,
        component: str | None,
        depth: int,
        tags: dict | None,
    ) -> None:
        dm = _resolve_sink_db_manager()
        if dm is None:
            return

        async def _go() -> None:
            await TraceRepository.create_span(
                trace_id=trace_id,
                span_id=span_id,
                span_name=span_name,
                span_type=span_type,
                parent_span_id=parent_span_id,
                component=component,
                depth=depth,
                tags=tags,
                db_manager=dm,
            )

        _schedule_trace_db_on_main_loop(_go())

    def on_span_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        dm = _resolve_sink_db_manager()
        if dm is None:
            return

        async def _go() -> None:
            await TraceRepository.finish_span(
                span_id=span_id,
                status=status,
                duration_ms=duration_ms,
                error_message=error_message,
                db_manager=dm,
            )

        _schedule_trace_db_on_main_loop(_go())

    def on_span_log(
        self,
        *,
        trace_id: str,
        span_id: str,
        event_name: str,
        message: str,
        payload: dict | None,
        log_level: str,
    ) -> None:
        dm = _resolve_sink_db_manager()
        if dm is None:
            return

        async def _go() -> None:
            await TraceRepository.append_span_log(
                trace_id=trace_id,
                span_id=span_id,
                log_level=log_level,
                event_name=event_name,
                message=message,
                payload=payload,
                db_manager=dm,
            )

        _schedule_trace_db_on_main_loop(_go())

    @_db_retry_async
    async def on_trace_start_async(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        trace_name: str,
        method: str,
        path: str,
    ) -> None:
        await TraceRepository.create_trace_run_with_root_span(
            trace_id=trace_id,
            trace_name=trace_name,
            http_path=path,
            http_method=method,
            span_id=span_id,
            span_name=trace_name,
            span_type="fastapi",
            component="fastapi",
            depth=0,
        )

    @_db_retry_async
    async def on_trace_finish_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        await TraceRepository.finish_span(
            span_id=span_id,
            status=status,
            duration_ms=duration_ms,
        )
        await TraceRepository.finish_trace_run(
            trace_id=trace_id,
            status=status,
            duration_ms=duration_ms,
        )

    @_db_retry_async
    async def on_span_start_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None,
        component: str | None,
        depth: int,
        tags: dict | None,
    ) -> None:
        await TraceRepository.create_span(
            trace_id=trace_id,
            span_id=span_id,
            span_name=span_name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            component=component,
            depth=depth,
            tags=tags,
        )

    @_db_retry_async
    async def on_span_finish_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        await TraceRepository.finish_span(
            span_id=span_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    @_db_retry_async
    async def on_span_log_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        event_name: str,
        message: str,
        payload: dict | None,
        log_level: str,
    ) -> None:
        await TraceRepository.append_span_log(
            trace_id=trace_id,
            span_id=span_id,
            log_level=log_level,
            event_name=event_name,
            message=message,
            payload=payload,
        )


class LoggerTraceSink(BaseTraceSink):
    def on_trace_start(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        trace_name: str,
        method: str,
        path: str,
    ) -> None:
        logger.debug("[trace] %s %s %s", method, path, trace_id)

    def on_trace_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        logger.debug("[trace] %sms http=%s %s", duration_ms, status_code, trace_id)

    def on_span_start(
        self,
        *,
        trace_id: str,
        span_id: str,
        span_name: str,
        span_type: str,
        parent_span_id: str | None,
        component: str | None,
        depth: int,
        tags: dict | None,
    ) -> None:
        logger.debug("[span] + %s", span_name)

    def on_span_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None,
    ) -> None:
        if error_message:
            logger.debug("[span] ~ %sms %s %s", duration_ms, status, error_message)
        else:
            logger.debug("[span] ~ %sms %s", duration_ms, status)

    def on_span_log(
        self,
        *,
        trace_id: str,
        span_id: str,
        event_name: str,
        message: str,
        payload: dict | None,
        log_level: str,
    ) -> None:
        logger.debug("[span] %s | %s", event_name, message)
