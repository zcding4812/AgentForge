"""链路追踪：W3C traceparent、状态枚举、列表 API 分页与路由白名单。状态规范化见 ``app.core.tracing.engine``。"""

from __future__ import annotations

from enum import StrEnum

# W3C Trace Context
TRACEPARENT_VERSION = "00"
TRACEPARENT_FLAGS = "01"


class TraceRunStatus(StrEnum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"


class TraceSpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"


# 链路列表 HTTP 分页（与 OpenAPI Query、`TraceService` / `TraceRepository` 一致）
TRACE_LIST_DEFAULT_PAGE_SIZE = 10
TRACE_LIST_MAX_PAGE_SIZE = 100

# Agent / LLM 类 span 的时长上限（毫秒）。全局 Tracer 默认 3000ms，整段流式或推理易误判为 timeout。
TRACE_SPAN_AGENT_WORK_TIMEOUT_MS = 600_000  # 10 分钟


# 纳入 HTTP 全链路 trace 的路由白名单（方法 → 路径集合，与 HttpMiddleware 对齐）
# 探活 GET /health 不写入 trace_runs，避免监控刷屏与库表膨胀。
TRACE_MONITORED_HTTP_ROUTES: dict[str, frozenset[str]] = {
    "POST": frozenset(
        {
            "/api/agents/invoke",
            "/api/agents/invoke/stream",
        }
    ),
}

# 路径前缀白名单（与上一项互补）：含动态段、无法逐条列入 frozenset 时使用（如 default-prompts/{kind}）
TRACE_MONITORED_HTTP_PREFIXES: dict[str, tuple[str, ...]] = {}
