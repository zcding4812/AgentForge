"""MCP Streamable HTTP 解析与工具工厂（无网络）。"""

import httpx
import pytest

from app.agent.adapters.tools.dynamic import mcp_streamable_http as msh


@pytest.fixture(autouse=True)
def _clear_mcp_session_cache() -> None:
    msh._mcp_session_store.clear()
    yield
    msh._mcp_session_store.clear()


def test_parse_sse_finds_matching_id() -> None:
    text = """
data: {"jsonrpc":"2.0","id":42,"result":{"x":1}}

"""
    obj = msh._parse_sse_for_id(text, 42)
    assert obj["result"]["x"] == 1


def test_make_tool_requires_server_url() -> None:
    with pytest.raises(ValueError, match="server_url"):
        msh.make_mcp_streamable_http_tool(
            logical_name="t",
            description="d",
            mcp_config={"transport_type": "http"},
        )


def test_make_tool_remote_name_none_when_no_mcp_tool_name() -> None:
    t = msh.make_mcp_streamable_http_tool(
        logical_name="docs_langchain",
        description="desc",
        mcp_config={
            "server_url": "https://docs.langchain.com/mcp",
            "transport_type": "http",
        },
    )
    assert t.__class__.__name__ == "McpStreamableHttpTool"
    cfg = t._cfg  # type: ignore[attr-defined]
    assert cfg.remote_tool_name is None
    schema = t.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert "arguments" in (schema.get("properties") or {})
    assert "query" not in (schema.get("properties") or {})


def test_make_tool_uses_tools_config_remote_name() -> None:
    t = msh.make_mcp_streamable_http_tool(
        logical_name="my_reg",
        description="desc",
        mcp_config={
            "server_url": "https://8.8.8.8/mcp",
            "transport_type": "http",
            "tools_config": {"mcp_tool_name": "remote_echo"},
        },
    )
    assert t.name == "my_reg"
    cfg = t._cfg  # type: ignore[attr-defined]
    assert cfg.remote_tool_name == "remote_echo"


def test_rpc_id_matches() -> None:
    assert msh._rpc_id_matches(42, 42)
    assert msh._rpc_id_matches("42", 42)
    assert not msh._rpc_id_matches(41, 42)


def test_tools_list_from_result() -> None:
    assert msh._tools_list_from_result({"tools": [{"name": "a"}]}) == [{"name": "a"}]
    assert msh._tools_list_from_result({}) == []
    assert msh._tools_list_from_result(None) == []


def test_initialize_http_error_hint_405_distinguishes_transport() -> None:
    req = httpx.Request("POST", "https://example.com/")
    resp = httpx.Response(405, request=req, headers={"Allow": "GET, HEAD"})
    h_stream = msh._initialize_http_error_hint(405, resp, transport="streamable_http")
    assert "Streamable HTTP" in h_stream
    h_sse = msh._initialize_http_error_hint(405, resp, transport="sse")
    assert "SSE" in h_sse
    assert "Streamable HTTP" not in h_sse


def test_mcp_transport_error_kind_classvar() -> None:
    assert msh.McpStreamableHttpTransport._mcp_http_error_transport == "streamable_http"
    assert msh.McpSseTransport._mcp_http_error_transport == "sse"


def test_normalize_mcp_tool_call_arguments_unwraps_nested() -> None:
    assert msh._normalize_mcp_tool_call_arguments({"arguments": {"foo": 1}}) == {"foo": 1}
    assert msh._normalize_mcp_tool_call_arguments({"query": "  hi  "}) == {"query": "  hi  "}


def test_client_list_tools_completed_flag() -> None:
    c = msh.McpStreamableHttpClient("https://example.com/mcp")
    assert not c.has_completed_list_tools(None)
    c._mark_list_tools_completed(None)
    assert c.has_completed_list_tools(None)


def test_dispatch_mode_registers_dispatch_tool() -> None:
    t = msh.make_mcp_streamable_http_tool(
        logical_name="mcp_router",
        description="router",
        mcp_config={
            "server_url": "https://8.8.8.8/mcp",
            "tools_config": {"dispatch_mode": True},
        },
    )
    assert t.__class__.__name__ == "McpStreamableHttpDispatchTool"
    cfg = t._cfg  # type: ignore[attr-defined]
    assert cfg.remote_tool_name == "__dispatch__"


def test_find_tool_entry_and_input_schema() -> None:
    tools = [
        {"name": "search_docs_by_lang_chain", "inputSchema": {"type": "object"}},
        {
            "name": "other",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    ]
    assert msh._find_tool_entry_by_name(tools, "search_docs_by_lang_chain") is not None
    assert msh._input_schema_from_tool_entry(tools[1]) == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }


def test_leaf_exception_unwraps_exception_group() -> None:
    inner = ValueError("real cause")
    eg = ExceptionGroup("wrapped", (inner,))
    assert msh._leaf_exception(eg) is inner


def test_create_transport_sse() -> None:
    cfg = msh.McpStreamableToolConfig(
        logical_name="t",
        description="d",
        server_url="https://8.8.8.8/sse",
        transport_type="sse",
    )
    tr = msh.create_mcp_transport_for_tool_config(cfg)
    assert isinstance(tr, msh.McpSseTransport)
