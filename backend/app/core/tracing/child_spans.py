"""在已开启 HTTP trace 的请求内创建子 Span（不依赖 ``@trace_span`` 装饰器）。

用于流式响应等「路由协程提前返回、业务仍在异步生成器内执行」的场景：须在返回
``StreamingResponse`` 之前 ``capture_span_parent``，再在生成器内用捕获的 parent 建子 span。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from starlette.requests import Request

from app.core.constants.tracing import TRACE_SPAN_AGENT_WORK_TIMEOUT_MS
from app.core.context import RequestTraceContext, TraceRuntimeContext


@dataclass(frozen=True, slots=True)
class SpanParentCapture:
    """单次请求内用于挂子 span 的父级信息（在 RequestContext 仍可用时捕获）。"""

    trace_id: str
    parent_span_id: str | None


def capture_span_parent(request: Request | None) -> SpanParentCapture | None:
    """从当前请求解析 trace_id 与父 span_id；未开启追踪或缺少上下文时返回 None。"""
    if request is None:
        return None
    if not getattr(request.state, "tracing_http_active", False):
        return None
    trace_id = RequestTraceContext.get_trace_id(request)
    if not trace_id or not RequestTraceContext.is_w3c_trace_id(trace_id):
        return None
    parent_span_id = TraceRuntimeContext.get_current_span_id() or RequestTraceContext.get_span_id(
        request
    )
    return SpanParentCapture(trace_id=trace_id, parent_span_id=parent_span_id)


@asynccontextmanager
async def child_span_async(
    span_name: str,
    parent: SpanParentCapture | None,
    *,
    span_type: str = "custom",
    component: str | None = None,
    depth: int = 2,
    timeout_ms: int | None = None,
) -> AsyncIterator[None]:
    f"""在 ``parent`` 下挂子 span；``parent`` 为 None 时直接透传（不记 span）。

    ``timeout_ms``：未传时使用全局 Tracer 默认（通常 3000ms）。Agent/LLM 长任务请传入
    ``TRACE_SPAN_AGENT_WORK_TIMEOUT_MS``（当前 {TRACE_SPAN_AGENT_WORK_TIMEOUT_MS}ms），避免整段推理被判为 timeout。
    """
    if parent is None:
        yield
        return
    # 延迟导入 ``tracer``：避免 ``tracing/__init__`` 与 ``child_spans`` 在包初始化阶段循环引用。
    from app.core.tracing.engine import tracer

    async with tracer.span_async(
        trace_id=parent.trace_id,
        span_name=span_name,
        span_type=span_type,
        parent_span_id=parent.parent_span_id,
        component=component,
        depth=depth,
        timeout_ms=timeout_ms,
    ):
        yield
