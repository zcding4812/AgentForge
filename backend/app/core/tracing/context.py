from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import Request

from app.core.constants import DEFAULT_REQUEST_ID
from app.core.context import RequestContext, RequestTraceContext, TraceRuntimeContext


class TraceContextProvider(ABC):
    """追踪上下文提供者：抽象获取 trace_id / span_id / request 的逻辑。"""

    @abstractmethod
    def get_trace_id(self, request: Request | None = None) -> str | None: ...

    @abstractmethod
    def get_span_id(self, request: Request | None = None) -> str | None: ...

    @abstractmethod
    def get_current_request(self) -> Request | None: ...


class FastApiTraceContextProvider(TraceContextProvider):
    """FastAPI 请求链路：复用 RequestTraceContext / RequestContext。"""

    def get_trace_id(self, request: Request | None = None) -> str | None:
        req = request or self.get_current_request()
        if not req:
            return None
        tid = RequestTraceContext.get_trace_id(req)
        return tid if tid != DEFAULT_REQUEST_ID else None

    def get_span_id(self, request: Request | None = None) -> str | None:
        # 先读 ContextVar：SSE 流式正文迭代时中间件可能已 reset RequestContext，但子 span 仍挂在运行时上。
        cur = TraceRuntimeContext.get_current_span_id()
        if cur:
            return cur
        req = request or self.get_current_request()
        if not req:
            return None
        return RequestTraceContext.get_span_id(req)

    def get_current_request(self) -> Request | None:
        return RequestContext.get_request()
