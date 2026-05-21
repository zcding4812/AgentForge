from __future__ import annotations

import asyncio
from time import perf_counter

from fastapi import Request

from app.core.constants import DEFAULT_REQUEST_ID
from app.core.constants.tracing import TraceRunStatus, TraceSpanStatus
from app.core.context import RequestTraceContext, TraceRuntimeContext
from app.core.logger import get_logger
from app.core.tracing.context import FastApiTraceContextProvider, TraceContextProvider
from app.core.tracing.sinks import BaseTraceSink, DbTraceSink, LoggerTraceSink

logger = get_logger(__name__)


def normalize_trace_run_status(raw: str | None) -> TraceRunStatus:
    if not raw:
        return TraceRunStatus.FAILED
    try:
        return TraceRunStatus(raw.strip().lower())
    except ValueError:
        return TraceRunStatus.FAILED


def normalize_span_status(raw: str | None) -> TraceSpanStatus:
    if not raw:
        return TraceSpanStatus.UNKNOWN

    r = raw.strip().lower()
    if r in ("failed", "timeout"):
        return TraceSpanStatus.ERROR
    if r == "success":
        return TraceSpanStatus.OK
    if r == "running":
        return TraceSpanStatus.IN_PROGRESS
    try:
        return TraceSpanStatus(r)
    except ValueError:
        return TraceSpanStatus.UNKNOWN


def _should_log_agent_span_lifecycle(span_type: str, component: str | None) -> bool:
    """Agent / 图节点等长链路 span 写入 span_logs，供监控页展示。"""
    if span_type == "agent":
        return True
    if component in ("agent-graph", "agent-service"):
        return True
    return False


class SpanScope:
    def __init__(
        self,
        *,
        tracer: Tracer,
        trace_id: str,
        span_id: str,
        timeout_ms: int,
    ) -> None:
        self._tracer = tracer
        self._trace_id = trace_id
        self._span_id = span_id
        self._start_ts = perf_counter()
        self._span_token = None
        self._timeout_ms = timeout_ms

    def __enter__(self) -> str:
        self._span_token = TraceRuntimeContext.set_current_span_id(self._span_id)
        return self._span_id

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if self._span_token is not None:
            TraceRuntimeContext.reset_current_span_id(self._span_token)
        duration_ms = max(1, round((perf_counter() - self._start_ts) * 1000))
        is_timeout = duration_ms > self._timeout_ms
        if is_timeout:
            status = "timeout"
            error_message = f"Span timeout (>{self._timeout_ms}ms)"
        elif exc is None:
            status = "success"
            error_message = None
        else:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"

        self._tracer.finish_span(
            trace_id=self._trace_id,
            span_id=self._span_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        return False


class AsyncSpanScope:
    def __init__(
        self,
        *,
        tracer: Tracer,
        trace_id: str,
        span_name: str,
        span_type: str = "custom",
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 1,
        tags: dict | None = None,
        timeout_ms: int = 3000,
    ) -> None:
        self._tracer = tracer
        self._trace_id = trace_id
        self._span_name = span_name
        self._span_type = span_type
        self._parent_span_id = parent_span_id
        self._component = component
        self._depth = depth
        self._tags = tags
        self._span_id: str | None = None
        self._start_ts = 0.0
        self._span_token = None
        self._timeout_ms = timeout_ms

    async def __aenter__(self) -> str:
        # 计时起点须在「本 span」的 start_span_async 完成之后：本 span 的 duration 才不包含自身 create 落库。
        # 嵌套子 span 的 start/finish 仍处在父 span 的 await 链上，父在 __aexit__ 取样 perf 时已包含子的整段 __aexit__
        # （含子 finish_span_async 落库）；若甘特图仍错位，多为 started_at（墙钟）与 perf 混用，见 trace_svc 展示修正。
        self._span_id = await self._tracer.start_span_async(
            trace_id=self._trace_id,
            span_name=self._span_name,
            span_type=self._span_type,
            parent_span_id=self._parent_span_id,
            component=self._component,
            depth=self._depth,
            tags=self._tags,
        )
        self._start_ts = perf_counter()
        self._span_token = TraceRuntimeContext.set_current_span_id(self._span_id)
        if _should_log_agent_span_lifecycle(self._span_type, self._component):
            payload: dict[str, object] = {
                "span_name": self._span_name,
                "span_type": self._span_type,
                "parent_span_id": self._parent_span_id,
                "depth": self._depth,
                "component": self._component,
            }
            if self._tags:
                payload["tags"] = self._tags
            await self._tracer.log_event_async(
                event_name="span.lifecycle.start",
                message=f"{self._span_name} 开始",
                trace_id=self._trace_id,
                span_id=self._span_id,
                payload=payload,
            )
        return self._span_id

    async def __aexit__(self, exc_type, exc, _tb) -> bool:
        if self._span_id is None:
            return False
        if self._span_token is not None:
            TraceRuntimeContext.reset_current_span_id(self._span_token)
        duration_ms = max(1, round((perf_counter() - self._start_ts) * 1000))
        is_timeout = duration_ms > self._timeout_ms
        if is_timeout:
            status = "timeout"
            error_message = f"Span timeout (>{self._timeout_ms}ms)"
        elif exc is None:
            status = "success"
            error_message = None
        else:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"

        if _should_log_agent_span_lifecycle(self._span_type, self._component):
            await self._tracer.log_event_async(
                event_name="span.lifecycle.end",
                message=f"{self._span_name} 结束 · {status} · {duration_ms}ms",
                trace_id=self._trace_id,
                span_id=self._span_id,
                payload={
                    "status": status,
                    "duration_ms": duration_ms,
                    "error_message": error_message,
                },
            )
        await self._tracer.finish_span_async(
            trace_id=self._trace_id,
            span_id=self._span_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        return False


class Tracer:
    def __init__(
        self,
        sinks: list[BaseTraceSink],
        context_provider: TraceContextProvider,
        *,
        span_timeout_ms: int = 3000,
    ) -> None:
        self._sinks = sinks
        self._context_provider = context_provider
        self._span_timeout_ms = span_timeout_ms

    def start_http_trace(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        method: str,
        path: str,
    ) -> None:
        p = path.rstrip("/")
        trace_name = (p.rsplit("/", maxsplit=1)[-1] if p else path) or path
        for sink in self._sinks:
            self._safe_emit(
                sink.on_trace_start,
                trace_id=trace_id,
                request_id=request_id,
                span_id=span_id,
                trace_name=trace_name,
                method=method,
                path=path,
            )

    async def start_http_trace_async(
        self,
        *,
        trace_id: str,
        request_id: str,
        span_id: str,
        method: str,
        path: str,
    ) -> None:
        p = path.rstrip("/")
        trace_name = (p.rsplit("/", maxsplit=1)[-1] if p else path) or path
        await asyncio.gather(
            *[
                self._safe_emit_async(
                    sink.on_trace_start_async,
                    trace_id=trace_id,
                    request_id=request_id,
                    span_id=span_id,
                    trace_name=trace_name,
                    method=method,
                    path=path,
                )
                for sink in self._sinks
            ]
        )

    def finish_http_trace(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        for sink in self._sinks:
            self._safe_emit(
                sink.on_trace_finish,
                trace_id=trace_id,
                span_id=span_id,
                status=status,
                duration_ms=duration_ms,
                status_code=status_code,
            )

    async def finish_http_trace_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        status_code: int,
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_emit_async(
                    sink.on_trace_finish_async,
                    trace_id=trace_id,
                    span_id=span_id,
                    status=status,
                    duration_ms=duration_ms,
                    status_code=status_code,
                )
                for sink in self._sinks
            ]
        )

    def start_span(
        self,
        *,
        trace_id: str,
        span_name: str,
        span_type: str = "custom",
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 1,
        tags: dict | None = None,
    ) -> str:
        span_id = RequestTraceContext.generate_span_id()
        for sink in self._sinks:
            self._safe_emit(
                sink.on_span_start,
                trace_id=trace_id,
                span_id=span_id,
                span_name=span_name,
                span_type=span_type,
                parent_span_id=parent_span_id,
                component=component,
                depth=depth,
                tags=tags,
            )
        return span_id

    async def start_span_async(
        self,
        *,
        trace_id: str,
        span_name: str,
        span_type: str = "custom",
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 1,
        tags: dict | None = None,
    ) -> str:
        span_id = RequestTraceContext.generate_span_id()
        await asyncio.gather(
            *[
                self._safe_emit_async(
                    sink.on_span_start_async,
                    trace_id=trace_id,
                    span_id=span_id,
                    span_name=span_name,
                    span_type=span_type,
                    parent_span_id=parent_span_id,
                    component=component,
                    depth=depth,
                    tags=tags,
                )
                for sink in self._sinks
            ]
        )
        return span_id

    def finish_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None = None,
    ) -> None:
        for sink in self._sinks:
            self._safe_emit(
                sink.on_span_finish,
                trace_id=trace_id,
                span_id=span_id,
                status=status,
                duration_ms=duration_ms,
                error_message=error_message,
            )

    async def finish_span_async(
        self,
        *,
        trace_id: str,
        span_id: str,
        status: str,
        duration_ms: int,
        error_message: str | None = None,
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_emit_async(
                    sink.on_span_finish_async,
                    trace_id=trace_id,
                    span_id=span_id,
                    status=status,
                    duration_ms=duration_ms,
                    error_message=error_message,
                )
                for sink in self._sinks
            ]
        )

    def log_event(
        self,
        *,
        event_name: str,
        message: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        payload: dict | None = None,
        log_level: str = "INFO",
        request: Request | None = None,
    ) -> None:
        tid = trace_id or self._context_provider.get_trace_id(request)
        sid = span_id or self._context_provider.get_span_id(request)
        if (
            tid is None
            or sid is None
            or tid == DEFAULT_REQUEST_ID
            or not RequestTraceContext.is_w3c_trace_id(tid)
            or not RequestTraceContext.is_w3c_span_id(sid)
        ):
            return
        for sink in self._sinks:
            self._safe_emit(
                sink.on_span_log,
                trace_id=tid,
                span_id=sid,
                event_name=event_name,
                message=message,
                payload=payload,
                log_level=log_level,
            )

    async def log_event_async(
        self,
        *,
        event_name: str,
        message: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        payload: dict | None = None,
        log_level: str = "INFO",
        request: Request | None = None,
    ) -> None:
        tid = trace_id or self._context_provider.get_trace_id(request)
        sid = span_id or self._context_provider.get_span_id(request)
        if (
            tid is None
            or sid is None
            or tid == DEFAULT_REQUEST_ID
            or not RequestTraceContext.is_w3c_trace_id(tid)
            or not RequestTraceContext.is_w3c_span_id(sid)
        ):
            return
        await asyncio.gather(
            *[
                self._safe_emit_async(
                    sink.on_span_log_async,
                    trace_id=tid,
                    span_id=sid,
                    event_name=event_name,
                    message=message,
                    payload=payload,
                    log_level=log_level,
                )
                for sink in self._sinks
            ]
        )

    def span(
        self,
        *,
        trace_id: str,
        span_name: str,
        span_type: str = "custom",
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 1,
        tags: dict | None = None,
        timeout_ms: int | None = None,
    ) -> SpanScope:
        span_id = self.start_span(
            trace_id=trace_id,
            span_name=span_name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            component=component,
            depth=depth,
            tags=tags,
        )
        return SpanScope(
            tracer=self,
            trace_id=trace_id,
            span_id=span_id,
            timeout_ms=timeout_ms if timeout_ms is not None else self._span_timeout_ms,
        )

    def span_async(
        self,
        *,
        trace_id: str,
        span_name: str,
        span_type: str = "custom",
        parent_span_id: str | None = None,
        component: str | None = None,
        depth: int = 1,
        tags: dict | None = None,
        timeout_ms: int | None = None,
    ) -> AsyncSpanScope:
        return AsyncSpanScope(
            tracer=self,
            trace_id=trace_id,
            span_name=span_name,
            span_type=span_type,
            parent_span_id=parent_span_id,
            component=component,
            depth=depth,
            tags=tags,
            timeout_ms=timeout_ms if timeout_ms is not None else self._span_timeout_ms,
        )

    @staticmethod
    def _safe_emit(handler, **kwargs) -> None:
        token = TraceRuntimeContext.disable_db_span()
        try:
            handler(**kwargs)
        except Exception:
            logger.exception(
                "Trace sink emit failed | handler=%s", getattr(handler, "__name__", str(handler))
            )
        finally:
            TraceRuntimeContext.reset_db_span(token)

    @staticmethod
    async def _safe_emit_async(handler, **kwargs) -> None:
        token = TraceRuntimeContext.disable_db_span()
        try:
            await handler(**kwargs)
        except Exception:
            logger.exception(
                "Trace sink async emit failed | handler=%s",
                getattr(handler, "__name__", str(handler)),
            )
        finally:
            TraceRuntimeContext.reset_db_span(token)


def build_tracer(
    context_provider: TraceContextProvider | None = None,
) -> Tracer:
    """装配全局 Tracer（DB + 日志 Sink，span 超时 3000ms）。"""
    provider = context_provider or FastApiTraceContextProvider()
    sinks: list[BaseTraceSink] = [DbTraceSink(), LoggerTraceSink()]
    return Tracer(
        sinks=sinks,
        context_provider=provider,
        span_timeout_ms=3000,
    )


tracer = build_tracer()
