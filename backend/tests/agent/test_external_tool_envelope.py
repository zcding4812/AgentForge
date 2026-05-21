"""外部工具 JSON 信封（JSON-RPC 形状 + session 注入）。"""

import json

import pytest

from app.agent.kernel.tool_runtime import (
    build_external_tool_error_json,
    build_external_tool_json,
    require_tool_invocation,
    tool_invocation_context,
)
from app.agent.kernel.ports import ToolInvocationContext


def test_build_external_tool_json_without_context() -> None:
    s = build_external_tool_json(
        kind="http",
        result={"status_code": 200, "body": "x"},
    )
    d = json.loads(s)
    assert d["jsonrpc"] == "2.0"
    assert d["tool_protocol"] == "external_v1"
    assert d["kind"] == "http"
    assert d["session_id"] is None
    assert d["request_id"] is None
    assert d["id"] is None
    assert d["isError"] is False
    assert d["result"]["status_code"] == 200


def test_build_external_tool_json_with_session_via_contextvar() -> None:
    ctx = ToolInvocationContext(session_id="sess-1", request_id="req-1", trace_id="tr-1")
    with tool_invocation_context(ctx):
        s = build_external_tool_json(kind="mcp", result={"ok": True})
        d = json.loads(s)
        assert d["session_id"] == "sess-1"
        assert d["request_id"] == "req-1"
        assert d["trace_id"] == "tr-1"
        assert d["id"] == "req-1"


def test_tool_invocation_context_restores() -> None:
    ctx = ToolInvocationContext(session_id="s", request_id="r", trace_id="t")
    with tool_invocation_context(ctx):
        assert require_tool_invocation() is ctx
    with pytest.raises(RuntimeError):
        require_tool_invocation()


def test_build_external_tool_error_json() -> None:
    s = build_external_tool_error_json(
        kind="http",
        code=-32603,
        message="boom",
        data={"x": 1},
    )
    d = json.loads(s)
    assert d["isError"] is True
    assert d["error"]["code"] == -32603
    assert d["error"]["message"] == "boom"
    assert d["error"]["data"]["x"] == 1
