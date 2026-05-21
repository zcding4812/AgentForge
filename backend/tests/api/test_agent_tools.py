"""GET /api/agents/tools：进程内工具注册表。"""

import uuid

from fastapi.testclient import TestClient


def test_post_mcp_probe_list_rejects_unknown_transport(client: TestClient) -> None:
    """OpenAPI 仅允许 http/sse；非法 transport_type 须 422。"""
    r = client.post(
        "/api/agents/tools/mcp/probe-list",
        json={
            "server_url": "https://example.com/mcp",
            "transport_type": "grpc",
        },
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 422


def test_get_agents_registered_tools(client: TestClient) -> None:
    r = client.get("/api/agents/tools", headers={"Accept": "application/json"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("message") == "ok"
    data = body["data"]
    assert data["namespace"] == "default"
    names = {item["name"] for item in data["items"]}
    assert "server_time" in names
    for item in data["items"]:
        assert item["origin"] in ("builtin", "runtime", "http", "mcp")
        assert "parameters_summary" in item


def test_post_register_http_tool_then_list_contains(client: TestClient) -> None:
    name = f"dyn_http_{uuid.uuid4().hex[:10]}"
    r = client.post(
        "/api/agents/tools/http",
        json={
            "name": name,
            "description": "测试用 GET 工具",
            "url": "https://example.com/path",
            "method": "GET",
            "timeout_seconds": 10,
        },
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("message") == "ok"
    assert body["data"]["name"] == name

    gr = client.get("/api/agents/tools", headers={"Accept": "application/json"})
    assert gr.status_code == 200
    gdata = gr.json()["data"]
    names = {item["name"] for item in gdata["items"]}
    assert name in names
    row = next(x for x in gdata["items"] if x["name"] == name)
    assert row["origin"] == "http"
    assert row["http_request"] is not None
    assert row["http_request"]["method"] == "GET"

    lr = client.get("/api/agents/tools/http", headers={"Accept": "application/json"})
    assert lr.status_code == 200
    persisted = {x["name"] for x in lr.json()["data"]["items"]}
    assert name in persisted

    del_r = client.delete(
        f"/api/agents/tools/http/{name}",
        headers={"Accept": "application/json"},
    )
    assert del_r.status_code == 204

    lr2 = client.get("/api/agents/tools/http", headers={"Accept": "application/json"})
    assert name not in {x["name"] for x in lr2.json()["data"]["items"]}


def test_post_register_mcp_tool_then_list_contains(client: TestClient) -> None:
    name = f"dyn_mcp_{uuid.uuid4().hex[:10]}"
    r = client.post(
        "/api/agents/tools/mcp",
        json={
            "name": name,
            "description": "测试 MCP 占位",
            "transport_type": "http",
            "server_url": "https://example.com/mcp",
        },
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("message") == "ok"
    assert body["data"]["name"] == name

    gr = client.get("/api/agents/tools", headers={"Accept": "application/json"})
    assert gr.status_code == 200
    gdata = gr.json()["data"]
    row = next(x for x in gdata["items"] if x["name"] == name)
    assert row["origin"] == "mcp"
    assert row["mcp_config"] is not None
    assert row["mcp_config"]["server_url"] == "https://example.com/mcp"

    pr = client.patch(
        f"/api/agents/tools/mcp/{name}",
        json={"description": "已更新", "version": 1},
        headers={"Accept": "application/json"},
    )
    assert pr.status_code == 200
    assert pr.json()["data"]["description"] == "已更新"

    del_r = client.delete(
        f"/api/agents/tools/http/{name}",
        headers={"Accept": "application/json"},
    )
    assert del_r.status_code == 204
