from __future__ import annotations

import time
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool

from app.agent.adapters.graph.deps.compile_deps import GraphCompileDeps
from app.agent.adapters.graph.strategies.plan_execute import PlanExecuteStrategy
from app.agent.adapters.graph.strategies.react import ReactStrategy
from app.agent.adapters.graph.strategies.simple_chat import SimpleChatStrategy
from app.agent.adapters.graph.strategies.workbench import WorkbenchStrategy
from app.agent.adapters.models.chat_model_registry import RegistryDispatchingChatModelFactory
from app.agent.kernel.ports import (
    AgentGraphFactoryPort,
    AgentGraphHandle,
    AgentGraphStrategy,
    ChatModelFactoryPort,
    GraphBuildContext,
)
from app.agent.kernel.spec import AgentKind


class AgentGraphStrategyRegistry:
    """`AgentKind` → 策略；仅注册/查询，不含业务逻辑。"""

    __slots__ = ("_strategy_map",)

    def __init__(self) -> None:
        self._strategy_map: dict[AgentKind, AgentGraphStrategy] = {}

    def register(self, kind: AgentKind, strategy: AgentGraphStrategy) -> None:
        if kind in self._strategy_map:
            msg = f"策略 {kind!r} 重复注册"
            raise ValueError(msg)
        self._strategy_map[kind] = strategy

    def get(self, kind: AgentKind) -> AgentGraphStrategy:
        try:
            return self._strategy_map[kind]
        except KeyError as e:
            raise KeyError(f"未注册 AgentKind={kind!r} 的策略，请先 register") from e


class LangGraphAgentFactory(AgentGraphFactoryPort):
    """依赖组装 + 策略调度；不写编排业务逻辑。"""

    __slots__ = ("_chat_model_factory", "_strategy_registry")

    @classmethod
    def build_default_registry(cls) -> AgentGraphStrategyRegistry:
        registry = AgentGraphStrategyRegistry()
        registry.register(AgentKind.SIMPLE_CHAT, SimpleChatStrategy())
        registry.register(AgentKind.REACT, ReactStrategy())
        registry.register(AgentKind.PLAN_EXECUTE, PlanExecuteStrategy())
        registry.register(AgentKind.WORKBENCH, WorkbenchStrategy())
        return registry

    def __init__(
        self,
        *,
        chat_model_factory: ChatModelFactoryPort[BaseChatModel] | None = None,
        strategy_registry: AgentGraphStrategyRegistry | None = None,
    ) -> None:
        self._chat_model_factory = chat_model_factory or RegistryDispatchingChatModelFactory()
        self._strategy_registry = strategy_registry or type(self).build_default_registry()

    def build(self, kind: AgentKind, ctx: GraphBuildContext) -> AgentGraphHandle:
        t0 = time.perf_counter()
        err: str | None = None
        success = True
        try:
            chat_model = self._chat_model_factory.build(ctx.model_snapshot)
            self._validate_chat_model(chat_model)
            tools = cast(tuple[BaseTool | dict, ...], tuple(ctx.tools or ()))
            compile_deps = GraphCompileDeps(
                chat_model=chat_model,
                tools=tools,
                telemetry=ctx.telemetry,
                tool_choice_policy=ctx.model_snapshot.tool_choice,
                input_content_filter=ctx.model_snapshot.input_content_filter,
            )
            strategy = self._strategy_registry.get(kind)
            return strategy.compile(compile_deps)
        except Exception as e:
            success = False
            err = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            ctx.telemetry.tracer.record_graph_compile(
                agent_kind=kind,
                duration_ms=duration_ms,
                success=success,
                error=err,
                trace_id=ctx.telemetry.trace_id,
                trace_parent_span_id=ctx.telemetry.trace_parent_span_id,
            )

    @staticmethod
    def _validate_chat_model(model: object) -> None:
        ok = isinstance(model, BaseChatModel) or (
            isinstance(model, RunnableBinding)
            and isinstance(getattr(model, "bound", None), BaseChatModel)
        )
        if not ok:
            raise TypeError(
                f"聊天模型须为 BaseChatModel 或经 bind 的 RunnableBinding（其 bound 为 BaseChatModel），实际为 {type(model).__name__}",
            )
