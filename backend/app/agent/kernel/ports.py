"""内核端口（Protocol）：图工厂与聊天模型工厂；实现位于 adapters。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from app.agent.kernel.spec import AgentKind, AgentState, ModelConfigSnapshot

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps

    CompiledAgentGraph: TypeAlias = CompiledStateGraph
else:
    BaseTool = object
    CompiledAgentGraph = object

TModel = TypeVar("TModel")


class AgentTracerPort(Protocol):
    """Agent 运行期遥测（编译耗时、LLM 调用耗时与 token）；实现可在 adapters 桥接日志/Metrics。

    内核不依赖 ``app``；默认使用 :data:`NO_OP_AGENT_TRACER`。
    """

    def record_graph_compile(
        self,
        *,
        agent_kind: AgentKind,
        duration_ms: float,
        success: bool = True,
        error: str | None = None,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
    ) -> None: ...

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
    ) -> None: ...


class NoOpAgentTracer:
    """无开销占位，便于在未注入实现时仍调用同一套 API。"""

    __slots__ = ()

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
        return None

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
        return None


NO_OP_AGENT_TRACER: AgentTracerPort = NoOpAgentTracer()


@dataclass(frozen=True, slots=True)
class AgentGraphTelemetry:
    """一次图编译与节点执行共用的遥测与链路字段（避免 Context / Deps / 节点间重复传参）。"""

    tracer: AgentTracerPort = NO_OP_AGENT_TRACER
    trace_id: str | None = None
    trace_parent_span_id: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """外部工具（HTTP/MCP）在一次 Agent 调用内共享的上下文：对齐设计中的 session_id / 审计字段。

    由 ``GraphBuildContext.tool_invocation`` 与 ``kernel.tool_runtime`` 中的
    ``ToolInvocationBinder``（ContextVar）同步设置，供 ``http_fetch`` / MCP 适配器读取；**内核不**依赖 LangChain Tool 具体类。
    """

    session_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None


class ChatModelFactoryPort(Protocol[TModel]):
    def build(self, snapshot: ModelConfigSnapshot) -> TModel: ...


@dataclass(frozen=True, slots=True)
class GraphBuildContext:
    model_snapshot: ModelConfigSnapshot
    tools: tuple[BaseTool | dict, ...] = ()
    #: 遥测与链路（``tracer`` / ``trace_id`` / ``request_id`` 等）单对象下传。
    telemetry: AgentGraphTelemetry = field(default_factory=AgentGraphTelemetry)
    #: 外部工具执行期上下文（会话 ID 等）；与 ``kernel.tool_runtime`` 一致。
    tool_invocation: ToolInvocationContext | None = None


class AgentGraphStrategy(Protocol):
    """编排策略：实现开闭原则，新增策略只实现本协议并注册（实现位于 ``adapters.graph.strategies``）。"""

    def compile(self, deps: GraphCompileDeps) -> CompiledAgentGraph:  # type: ignore[name-defined]
        ...


@runtime_checkable
class AgentGraphHandle(Protocol):
    async def ainvoke(
        self,
        state: AgentState,
        config: Any | None = None,
        **kwargs: Any,
    ) -> AgentState: ...

    async def astream(
        self,
        state: AgentState,
        config: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]: ...


class AgentGraphFactoryPort(Protocol):
    def build(self, kind: AgentKind, ctx: GraphBuildContext) -> AgentGraphHandle: ...


@dataclass(frozen=True, slots=True)
class RollingSummarySnapshot:
    """会话级滚动摘要只读快照（跨 Run 记忆；与 LangGraph 单 Run 状态区分）。"""

    text: str | None
    version: int
    job_status: int


class MemoryPort(Protocol):
    """滚动摘要：加载已落库摘要、调度异步摘要任务（实现位于 ``services`` / ``infrastructure``）。"""

    async def load_rolling_summary(self, session_id: str) -> RollingSummarySnapshot | None: ...

    async def schedule_summarize(
        self,
        session_id: str,
        *,
        reason: Literal["threshold", "urgent", "manual"],
    ) -> None: ...
