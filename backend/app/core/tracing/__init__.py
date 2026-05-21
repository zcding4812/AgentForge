"""应用链路追踪：包根仅导出常用 API。

业务代码典型用法：``tracer``、``@trace_span``、``capture_span_parent`` / ``child_span_async``。
扩展实现（自定义 Sink、``TraceContextProvider``、``Tracer`` / ``build_tracer``）请从子模块导入，例如
``from app.core.tracing.sinks import BaseTraceSink``。

核心设计
--------

**延迟导入（打破循环依赖）**

    子模块 ``child_spans`` 不在顶层导入 ``engine.tracer``，仅在
    :func:`~app.core.tracing.child_spans.child_span_async` 进入「需记子 span」分支时
    再 ``from app.core.tracing.engine import tracer``，避免
    ``import app.core.tracing`` 时出现 ``__init__ → child_spans → engine → …`` 的初始化环。

**依赖方向**

    * ``engine`` 依赖 ``sinks``、``context``；
    * ``decorators`` 依赖 ``engine``、``context``；
    * ``child_spans`` 依赖 ``context`` 与常量，运行期再解析 ``tracer``。

Note:
    ``wrappers`` 供中间件内部使用，不在此 re-export。

"""

from app.core.tracing.child_spans import (
    SpanParentCapture,
    capture_span_parent,
    child_span_async,
)
from app.core.tracing.decorators import trace_span
from app.core.tracing.engine import tracer

__all__ = [
    "SpanParentCapture",
    "capture_span_parent",
    "child_span_async",
    "trace_span",
    "tracer",
]
