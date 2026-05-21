from collections.abc import Awaitable, Callable
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.constants import TRACEPARENT_HEADER, X_REQUEST_ID_HEADER, TraceRunStatus
from app.core.constants.tracing import TRACE_MONITORED_HTTP_PREFIXES, TRACE_MONITORED_HTTP_ROUTES
from app.core.context import RequestContext, RequestTraceContext
from app.core.logger import get_logger
from app.core.tracing import tracer
from app.core.tracing.wrappers import (
    is_streaming_response,
    short_request_id_for_log,
    wrap_streaming_body_for_http_trace,
)

# 极简日志器
logger = get_logger(__name__)


class HttpMiddleware(BaseHTTPMiddleware):
    @staticmethod
    def _should_trace(request: Request) -> bool:
        """精确路径或前缀白名单（常量见 ``app.core.constants.tracing``）。"""
        m = request.method.upper()
        path = request.url.path
        if path in TRACE_MONITORED_HTTP_ROUTES.get(m, frozenset()):
            return True
        return any(path.startswith(p) for p in TRACE_MONITORED_HTTP_PREFIXES.get(m, ()))

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # 初始化核心变量
        start_time = perf_counter()
        request_id = RequestTraceContext.setup_from_headers(request) or "unknown"
        should_trace = self._should_trace(request)
        request.state.tracing_http_active = should_trace
        token = RequestContext.set_request(request)

        try:
            # 启动链路追踪（仅核心逻辑，无冗余日志）
            if should_trace:
                trace_id = RequestTraceContext.get_trace_id(request) or "unknown"
                span_id = RequestTraceContext.get_span_id(request) or "unknown"
                await tracer.start_http_trace_async(
                    trace_id=trace_id,
                    request_id=request_id,
                    span_id=span_id,
                    method=request.method,
                    path=request.url.path,
                )

            # 执行业务逻辑
            response = await call_next(request)

            # 注入响应头
            response.headers[X_REQUEST_ID_HEADER] = request_id
            response.headers[TRACEPARENT_HEADER] = RequestTraceContext.build_traceparent(
                trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
                span_id=RequestTraceContext.get_span_id(request) or "unknown",
            )

            # 流式正文：检测 + 包装见 ``tracing.wrappers``
            if should_trace and is_streaming_response(response):
                response.body_iterator = wrap_streaming_body_for_http_trace(
                    response.body_iterator,
                    request=request,
                    response=response,
                    start_time=start_time,
                    request_id=request_id,
                )
                return response

            duration = round((perf_counter() - start_time) * 1000, 2)
            logger.info(
                "%s | %s %s · %s · %.2fms",
                short_request_id_for_log(request_id),
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )

            if should_trace:
                await tracer.finish_http_trace_async(
                    trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
                    span_id=RequestTraceContext.get_span_id(request) or "unknown",
                    status=(
                        TraceRunStatus.SUCCESS.value
                        if response.status_code < 500
                        else TraceRunStatus.FAILED.value
                    ),
                    duration_ms=max(1, int(round(duration))),
                    status_code=response.status_code,
                )

            return response

        except Exception as e:
            # 【简化日志】异常日志仅保留关键信息
            duration = round((perf_counter() - start_time) * 1000, 2)
            logger.error(
                "%s | %s %s · 500 · %.2fms · %s",
                short_request_id_for_log(request_id),
                request.method,
                request.url.path,
                duration,
                e,
                exc_info=True,
            )

            # 异常场景完成追踪
            if should_trace:
                await tracer.finish_http_trace_async(
                    trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
                    span_id=RequestTraceContext.get_span_id(request) or "unknown",
                    status=TraceRunStatus.FAILED.value,
                    duration_ms=duration,
                    status_code=500,
                )
            raise

        finally:
            # 重置请求上下文
            RequestContext.reset_request(token)
