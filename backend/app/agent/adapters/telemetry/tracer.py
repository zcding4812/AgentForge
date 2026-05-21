"""默认 Agent 遥测：实现 ``AgentTracerPort``，写入应用日志（不依赖 ``app.core.tracing`` 的 span API）。"""

from __future__ import annotations

from app.agent.kernel.ports import AgentTracerPort
from app.agent.kernel.spec import AgentKind
from app.core.logger import get_logger


class AppAgentTracer:
    """结构化日志形式的编译 / LLM 调用记录，便于与日志采集或后续 Metrics 管道对接。"""

    __slots__ = ("_log",)

    def __init__(self, *, name: str = "app.agent.telemetry") -> None:
        self._log = get_logger(name)

    def record_graph_compile(
        self,
        *,
        agent_kind: AgentKind,
        duration_ms: float,
        success: bool = True,
        error: str | None = None,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
    ) -> None:
        self._log.info(
            "agent.graph.compile kind=%s duration_ms=%.3f success=%s error=%s "
            "trace_id=%s trace_parent_span_id=%s",
            str(agent_kind),
            duration_ms,
            success,
            error,
            trace_id,
            trace_parent_span_id,
        )

    def record_llm_call(
        self,
        *,
        phase: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        model_name: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
    ) -> None:
        self._log.info(
            "agent.llm phase=%s duration_ms=%.3f prompt_tokens=%s completion_tokens=%s "
            "total_tokens=%s model=%s request_id=%s trace_id=%s trace_parent_span_id=%s",
            phase,
            duration_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model_name,
            request_id,
            trace_id,
            trace_parent_span_id,
        )


_default_app_tracer: AppAgentTracer | None = None


def get_app_agent_tracer() -> AgentTracerPort:
    """进程内单例；供 ``AgentRunFacade`` 等内部取用。"""
    global _default_app_tracer
    if _default_app_tracer is None:
        _default_app_tracer = AppAgentTracer()
    return _default_app_tracer
