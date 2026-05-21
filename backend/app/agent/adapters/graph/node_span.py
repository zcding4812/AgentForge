"""LangGraph 节点与 HTTP Trace 子 Span 对齐（适配器层；内核不依赖本模块）。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.core.constants.tracing import TRACE_SPAN_AGENT_WORK_TIMEOUT_MS
from app.core.context import RequestTraceContext
from app.core.tracing.child_spans import SpanParentCapture, child_span_async


@asynccontextmanager
async def agent_graph_child_span(
    span_name: str,
    *,
    trace_id: str | None,
    trace_parent_span_id: str | None,
) -> AsyncGenerator[None, None]:
    """在存在 ``trace_id`` 时挂到 ``trace_parent_span_id`` 下的子 span；否则无操作。"""
    if not trace_id or not RequestTraceContext.is_w3c_trace_id(trace_id):
        yield
        return
    cap = SpanParentCapture(trace_id, trace_parent_span_id)
    async with child_span_async(
        span_name,
        cap,
        span_type="agent",
        component="agent-graph",
        depth=3,
        timeout_ms=TRACE_SPAN_AGENT_WORK_TIMEOUT_MS,
    ):
        yield
