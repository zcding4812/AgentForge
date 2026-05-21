"""HTTP 追踪与流式正文：检测流式响应、包装 ``body_iterator``，在流结束后收尾 trace。

供中间件使用：``is_streaming_response`` 决策是否推迟结束 HTTP trace；
``wrap_streaming_body_for_http_trace`` 在迭代完成后写入 ``trace_runs.duration_ms`` 并恢复
``RequestContext``。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter

from starlette.requests import Request
from starlette.responses import Response

from app.core.constants import TraceRunStatus
from app.core.context import RequestContext, RequestTraceContext
from app.core.logger import get_logger
from app.core.tracing.engine import tracer

logger = get_logger(__name__)


def short_request_id_for_log(request_id: str, *, max_len: int = 10) -> str:
    """将 ``X-Request-Id`` 截断为日志友好长度。"""
    rid = request_id.strip() or "unknown"
    return rid if len(rid) <= max_len else rid[:max_len]


def is_streaming_response(response: Response) -> bool:
    """
    判断响应是否带异步流式正文。

    Starlette 中带 ``body_iterator`` 的响应在 ``call_next`` 返回时正文往往尚未写出，
    HTTP trace 的结束与耗时须在迭代结束后再记。用 ``body_iterator`` 存在性判断，避免仅依赖
    ``isinstance(..., StreamingResponse)`` 漏掉子类或代理响应。
    """
    return getattr(response, "body_iterator", None) is not None


def wrap_streaming_body_for_http_trace(
    body: AsyncIterator[bytes],
    *,
    request: Request,
    response: Response,
    start_time: float,
    request_id: str,
) -> AsyncIterator[bytes]:
    """
    包装 ``body_iterator``：在全部 chunk 发出（或异常/断开）后结束 HTTP trace。

    使用 ``async for`` + ``finally`` 而非手写 ``__anext__``，以便统一处理
    ``GeneratorExit``、客户端断开等路径，且只 finalize 一次。
    """

    async def _gen() -> AsyncIterator[bytes]:
        err: BaseException | None = None
        try:
            async for chunk in body:
                yield chunk
        except BaseException as exc:
            err = exc
            raise
        finally:
            ctx_tok = RequestContext.set_request(request)
            try:
                duration_ms = max(1, int(round((perf_counter() - start_time) * 1000)))
                ok = err is None and response.status_code < 500
                await tracer.finish_http_trace_async(
                    trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
                    span_id=RequestTraceContext.get_span_id(request) or "unknown",
                    status=(TraceRunStatus.SUCCESS.value if ok else TraceRunStatus.FAILED.value),
                    duration_ms=duration_ms,
                    status_code=response.status_code if err is None else 500,
                )
                logger.info(
                    "%s | %s %s · %s · %.2fms",
                    short_request_id_for_log(request_id),
                    request.method,
                    request.url.path,
                    response.status_code,
                    float(duration_ms),
                )
            finally:
                RequestContext.reset_request(ctx_tok)

    return _gen()
