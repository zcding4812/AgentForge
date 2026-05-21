"""MCP URL 校验（scheme / 端口等）；解析 IP 网段不再拦截。"""

import pytest

from app.agent.kernel.tool_runtime import (
    McpUrlNotAllowedError,
    assert_mcp_http_url_allowed,
)


def test_allows_private_literal_ip() -> None:
    u = assert_mcp_http_url_allowed("http://192.168.1.1/mcp")
    assert u.startswith("http://")


def test_allows_public_https() -> None:
    # 字面公网 IP，避免测试机无 DNS
    u = assert_mcp_http_url_allowed("https://8.8.8.8/mcp")
    assert u.startswith("https://")


def test_rejects_forbidden_port() -> None:
    with pytest.raises(McpUrlNotAllowedError):
        assert_mcp_http_url_allowed("http://8.8.8.8:22/mcp")


def test_allows_loopback_without_env() -> None:
    u = assert_mcp_http_url_allowed("http://127.0.0.1:8080/mcp")
    assert "127.0.0.1" in u
