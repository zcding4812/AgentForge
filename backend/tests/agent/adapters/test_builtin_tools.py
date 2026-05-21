"""内置工具注册与解析。"""

import asyncio
import re

import pytest

from app.agent.adapters.tools.builtins import BUILTIN_TOOL_NAMES, register_builtin_tools
from app.agent.adapters.tools.registry import resolve_tool_names_with_report, tool_registry


@pytest.fixture
def clean_then_builtins() -> None:
    tool_registry._tools.clear()
    tool_registry._resolve_cache.clear()
    register_builtin_tools()
    yield
    tool_registry._tools.clear()
    tool_registry._resolve_cache.clear()


def test_builtin_tools_resolve(clean_then_builtins: None) -> None:
    assert "server_time" in BUILTIN_TOOL_NAMES
    r = resolve_tool_names_with_report(["server_time", "missing"], strict=False)
    assert r.missing_names == ("missing",)
    assert len(r.tools) == 1


def test_server_time_tool_name_matches_registry_key(clean_then_builtins: None) -> None:
    r = resolve_tool_names_with_report(["server_time"], strict=False)
    assert len(r.tools) == 1
    assert getattr(r.tools[0], "name", None) == "server_time"


def test_server_time_tool_returns_beijing_and_utc(clean_then_builtins: None) -> None:
    r = resolve_tool_names_with_report(["server_time"], strict=False)
    assert len(r.tools) == 1
    tool = r.tools[0]
    out = asyncio.run(tool.ainvoke({}))
    assert isinstance(out, str)
    assert "北京时间（UTC+8）:" in out
    assert "UTC:" in out
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", out)
