"""工具注册表：容错解析、严格模式、重复注册。"""

import pytest
from langchain_core.tools import tool

from app.agent.adapters.tools.registry import (
    ToolRegistrationConflictError,
    ToolResolutionStrictError,
    tool_registry,
)


@pytest.fixture
def clear_registry() -> None:
    tool_registry._tools.clear()
    tool_registry._resolve_cache.clear()
    yield
    tool_registry._tools.clear()
    tool_registry._resolve_cache.clear()


@tool("demo_a")
def demo_a(x: str) -> str:
    """Demo tool a."""
    return x


@tool("demo_b")
def demo_b(x: str) -> str:
    """Demo tool b."""
    return x


def test_register_conflict_without_overwrite(clear_registry: None) -> None:
    tool_registry.register("t1", demo_a, overwrite=False)
    with pytest.raises(ToolRegistrationConflictError):
        tool_registry.register("t1", demo_b, overwrite=False)


def test_register_overwrite(clear_registry: None) -> None:
    tool_registry.register("t1", demo_a, overwrite=False)
    tool_registry.register("t1", demo_b, overwrite=True)
    assert tool_registry.resolve_tool_names(["t1"])[0] is demo_b


def test_resolve_skips_unknown_with_log(clear_registry: None) -> None:
    tool_registry.register("known", demo_a, overwrite=False)
    r = tool_registry.resolve_tool_names_with_report(["known", "ghost"], strict=False)
    assert len(r.tools) == 1
    assert r.missing_names == ("ghost",)


def test_resolve_strict_raises(clear_registry: None) -> None:
    tool_registry.register("known", demo_a, overwrite=False)
    with pytest.raises(ToolResolutionStrictError) as ei:
        tool_registry.resolve_tool_names_with_report(["known", "ghost"], strict=True)
    assert "ghost" in str(ei.value)
