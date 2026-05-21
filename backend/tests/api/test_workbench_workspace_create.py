"""POST /api/agents/workbench：按命名空间创建工作台 Agent。"""

import uuid

from fastapi.testclient import TestClient


def test_create_workbench_workspace_and_reject_duplicate(client: TestClient) -> None:
    ns = f"pytest-ws-{uuid.uuid4().hex[:12]}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r1 = client.post(
        "/api/agents/workbench",
        json={
            "workspace_namespace": ns,
            "name": "工作台",
            "description": "pytest",
        },
        headers=headers,
    )
    assert r1.status_code == 200, r1.text
    data = r1.json()["data"]
    assert data["agent_kind"] == "workbench"
    assert data["workspace_namespace"] == ns

    r2 = client.post(
        "/api/agents/workbench",
        json={"workspace_namespace": ns, "name": "第二条"},
        headers=headers,
    )
    assert r2.status_code == 400
    assert "已存在" in r2.json()["detail"] or "工作台" in r2.json()["detail"]
