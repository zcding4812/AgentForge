from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from fastapi import Request

from app.core.constants import DEFAULT_REQUEST_ID
from app.core.tracing.context import FastApiTraceContextProvider, TraceContextProvider
from app.core.tracing.engine import tracer


def trace_span(
    *,
    span_name: str | None = None,
    span_type: str = "custom",
    component: str | None = None,
    trace_id_arg: str | None = None,
    depth: int = 1,
    context_provider: TraceContextProvider | None = None,
) -> Callable:
    provider = context_provider or FastApiTraceContextProvider()

    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)
        resolved_name = span_name if span_name is not None else _default_span_name(func)

        def get_span_context(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[str | None, Request | None, str | None]:
            bound = signature.bind_partial(*args, **kwargs)
            trace_id, request = _get_trace_context(bound.arguments, trace_id_arg, provider)
            if (
                not trace_id
                or request is None
                or not getattr(request.state, "tracing_http_active", False)
            ):
                return None, None, None
            parent_span_id = provider.get_span_id(request)
            return trace_id, request, parent_span_id

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_id, _request, parent_span_id = get_span_context(args, kwargs)
            if not trace_id:
                return await func(*args, **kwargs)

            async with tracer.span_async(
                trace_id=trace_id,
                span_name=resolved_name,
                span_type=span_type,
                parent_span_id=parent_span_id,
                component=component,
                depth=depth,
            ):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_id, _request, parent_span_id = get_span_context(args, kwargs)
            if not trace_id:
                return func(*args, **kwargs)

            with tracer.span(
                trace_id=trace_id,
                span_name=resolved_name,
                span_type=span_type,
                parent_span_id=parent_span_id,
                component=component,
                depth=depth,
            ):
                return func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


def _default_span_name(func: Callable) -> str:
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "unknown")
    return qualname.rsplit(".", maxsplit=1)[-1]


def _read_argument(
    arguments: dict[str, Any],
    key: str | None,
    context_provider: TraceContextProvider,
) -> str | None:
    if not key:
        return None
    value = arguments.get(key)
    if isinstance(value, Request):
        return context_provider.get_trace_id(value)
    if isinstance(value, dict):
        nested_trace_id = value.get("trace_id")
        if isinstance(nested_trace_id, str) and nested_trace_id.strip():
            return nested_trace_id.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _get_trace_context(
    bound_arguments: dict[str, Any],
    trace_id_arg: str | None,
    context_provider: TraceContextProvider,
) -> tuple[str | None, Request | None]:
    if trace_id_arg is not None and trace_id_arg in bound_arguments:
        value = bound_arguments[trace_id_arg]
        if isinstance(value, Request):
            tid = context_provider.get_trace_id(value)
            if not tid:
                return None, None
            return tid, value
        tid = _read_argument(bound_arguments, trace_id_arg, context_provider)
        if tid:
            req = context_provider.get_current_request()
            if req is None:
                return None, None
            return tid, req

    req = context_provider.get_current_request()
    if req is None:
        return None, None
    tid = context_provider.get_trace_id(req)
    if not tid or tid == DEFAULT_REQUEST_ID:
        return None, None
    return tid, req
