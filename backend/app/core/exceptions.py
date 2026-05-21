from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler as fastapi_http_exception_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as fastapi_request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse, Response

from app.agent.kernel.exceptions import AgentExecutionError
from app.core.constants import TRACEPARENT_HEADER, X_REQUEST_ID_HEADER
from app.core.context import RequestTraceContext
from app.core.logger import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> Response:
        request_id = RequestTraceContext.get_request_id(request) or "unknown"
        logger.warning(
            "参数校验失败 | 请求ID: %s | 路径: %s | 错误: %s",
            request_id,
            request.url.path,
            exc.errors(),
        )
        response = await fastapi_request_validation_exception_handler(request, exc)
        response.headers[X_REQUEST_ID_HEADER] = request_id
        response.headers[TRACEPARENT_HEADER] = RequestTraceContext.build_traceparent(
            trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
            span_id=RequestTraceContext.get_span_id(request) or "unknown",
        )
        return response

    @app.exception_handler(AgentExecutionError)
    async def agent_execution_error_handler(request: Request, exc: AgentExecutionError) -> Response:
        request_id = RequestTraceContext.get_request_id(request) or "unknown"
        trace_id = RequestTraceContext.get_trace_id(request) or "unknown"
        span_id = RequestTraceContext.get_span_id(request) or "unknown"
        client = f"{request.client.host}:{request.client.port}" if request.client else "unknown"
        path_qs = str(request.url)
        logger.warning(
            "Agent 执行失败 | trace_id=%s | span_id=%s | client=%s | %s %s \nexc_type=%s | %s",
            trace_id,
            span_id,
            client,
            request.method,
            path_qs,
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)},
            headers={
                X_REQUEST_ID_HEADER: request_id,
                TRACEPARENT_HEADER: RequestTraceContext.build_traceparent(
                    trace_id=trace_id,
                    span_id=span_id,
                ),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
        request_id = RequestTraceContext.get_request_id(request) or "unknown"
        logger.warning(
            "HTTP异常 | 请求ID: %s | 路径: %s | 状态码: %s | 详情: %s",
            request_id,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        response = await fastapi_http_exception_handler(request, exc)
        response.headers[X_REQUEST_ID_HEADER] = request_id
        response.headers[TRACEPARENT_HEADER] = RequestTraceContext.build_traceparent(
            trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
            span_id=RequestTraceContext.get_span_id(request) or "unknown",
        )
        return response

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception) -> Response:
        request_id = RequestTraceContext.get_request_id(request) or "unknown"
        logger.exception("未捕获异常 | 请求ID: %s | 路径: %s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
            headers={
                X_REQUEST_ID_HEADER: request_id,
                TRACEPARENT_HEADER: RequestTraceContext.build_traceparent(
                    trace_id=RequestTraceContext.get_trace_id(request) or "unknown",
                    span_id=RequestTraceContext.get_span_id(request) or "unknown",
                ),
            },
        )
