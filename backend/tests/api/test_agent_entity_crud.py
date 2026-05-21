"""Agent 资源：PATCH 元数据、DELETE 物理删除。"""

import uuid

from fastapi.testclient import TestClient


def test_agent_patch_metadata_and_delete(client: TestClient) -> None:
    name = f"pytest-agent-{uuid.uuid4().hex[:10]}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    cr = client.post(
        "/api/agents",
        json={"name": name, "description": "d0", "agent_kind": "simple_chat"},
        headers=headers,
    )
    assert cr.status_code == 200
    aid = cr.json()["data"]["id"]

    pr = client.patch(
        f"/api/agents/{aid}",
        json={"name": f"{name}-renamed", "description": "d1", "agent_kind": "react"},
        headers=headers,
    )
    assert pr.status_code == 200
    body = pr.json()["data"]
    assert body["name"] == f"{name}-renamed"
    assert body["description"] == "d1"
    assert body["agent_kind"] == "react"

    dr = client.delete(f"/api/agents/{aid}", headers={"Accept": "application/json"})
    assert dr.status_code == 204
    assert dr.content == b""

    gr = client.get(f"/api/agents/{aid}", headers={"Accept": "application/json"})
    assert gr.status_code == 404


def test_delete_workbench_purges_namespace_and_sibling_agents(client: TestClient) -> None:
    """删除工作台 Agent 时级联删除命名空间与同一空间内其它 Agent。"""
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    ns = f"pytest-ws-{uuid.uuid4().hex[:12]}"
    wr = client.post(
        "/api/agents/workbench",
        json={
            "workspace_namespace": ns,
            "name": f"wb-{ns}",
            "description": "d",
        },
        headers=headers,
    )
    assert wr.status_code == 200, wr.text
    wid = wr.json()["data"]["id"]

    ar = client.post(
        "/api/agents",
        json={
            "name": f"child-{ns}",
            "description": "c",
            "agent_kind": "simple_chat",
            "workspace_namespace": ns,
        },
        headers=headers,
    )
    assert ar.status_code == 200, ar.text
    cid = ar.json()["data"]["id"]

    dr = client.delete(f"/api/agents/{wid}", headers={"Accept": "application/json"})
    assert dr.status_code == 204, dr.text

    assert client.get(f"/api/agents/{wid}", headers=headers).status_code == 404
    assert client.get(f"/api/agents/{cid}", headers=headers).status_code == 404

    ns_list = client.get("/api/workspace-namespaces", headers=headers)
    assert ns_list.status_code == 200
    slugs = {it["slug"] for it in ns_list.json()["data"]["items"]}
    assert ns not in slugs
