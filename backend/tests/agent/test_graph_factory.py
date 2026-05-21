import pytest

from app.agent.adapters.graph.factory import LangGraphAgentFactory
from app.agent.adapters.graph.factory.langgraph_factory import AgentGraphStrategyRegistry
from app.agent.adapters.graph.strategies.simple_chat import SimpleChatStrategy
from app.agent.kernel.ports import AgentGraphTelemetry, GraphBuildContext
from app.agent.kernel.spec import (
    AgentKind,
    InferenceHyperparameters,
    ModelConfigSnapshot,
    ModelIdentity,
    ResponseConstraints,
)


class _RecordingTracer:
    """duck-type ``AgentTracerPort`` for tests."""

    def __init__(self) -> None:
        self.compiles: list[dict] = []

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
        self.compiles.append(
            {
                "agent_kind": agent_kind,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "trace_id": trace_id,
                "trace_parent_span_id": trace_parent_span_id,
            }
        )

    def record_llm_call(self, **kwargs: object) -> None:
        return None


def _snapshot() -> ModelConfigSnapshot:
    return ModelConfigSnapshot(
        identity=ModelIdentity(provider="openai", model_name="gpt-4o-mini"),
        hyperparameters=InferenceHyperparameters(),
        response=ResponseConstraints(),
    )


def test_duplicate_strategy_register_raises() -> None:
    reg = AgentGraphStrategyRegistry()
    reg.register(AgentKind.SIMPLE_CHAT, SimpleChatStrategy())
    with pytest.raises(ValueError, match="重复注册"):
        reg.register(AgentKind.SIMPLE_CHAT, SimpleChatStrategy())


def test_factory_compiles_each_agent_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-in-compile-only")
    factory = LangGraphAgentFactory()
    ctx = GraphBuildContext(model_snapshot=_snapshot())
    for kind in AgentKind:
        graph = factory.build(kind, ctx)
        assert graph is not None


def test_factory_records_graph_compile_via_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-in-compile-only")
    tracer = _RecordingTracer()
    factory = LangGraphAgentFactory()
    ctx = GraphBuildContext(
        model_snapshot=_snapshot(),
        telemetry=AgentGraphTelemetry(
            tracer=tracer,
            trace_id="a" * 32,
            trace_parent_span_id="b" * 16,
        ),
    )
    factory.build(AgentKind.SIMPLE_CHAT, ctx)
    assert len(tracer.compiles) == 1
    assert tracer.compiles[0]["success"] is True
    assert tracer.compiles[0]["agent_kind"] == AgentKind.SIMPLE_CHAT
    assert tracer.compiles[0]["error"] is None
    assert tracer.compiles[0]["trace_id"] == "a" * 32
    assert tracer.compiles[0]["trace_parent_span_id"] == "b" * 16
