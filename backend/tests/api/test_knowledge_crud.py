"""知识库：创建、列表、详情、PATCH、软删。"""

import uuid

from fastapi.testclient import TestClient


def test_knowledge_crud_flow(client: TestClient) -> None:
    name = f"pytest-kb-{uuid.uuid4().hex[:10]}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    cr = client.post(
        "/api/knowledge",
        json={"name": name, "description": "d0"},
        headers=headers,
    )
    assert cr.status_code == 200
    kid = cr.json()["data"]["id"]
    slug = cr.json()["data"]["slug"]

    lr = client.get("/api/knowledge", headers={"Accept": "application/json"})
    assert lr.status_code == 200
    ids = [x["id"] for x in lr.json()["data"]["items"]]
    assert kid in ids

    gr = client.get(f"/api/knowledge/{kid}", headers={"Accept": "application/json"})
    assert gr.status_code == 200
    assert gr.json()["data"]["name"] == name
    assert gr.json()["data"]["slug"] == slug

    pr = client.patch(
        f"/api/knowledge/{kid}",
        json={"description": "d1", "chunk_size": 256},
        headers=headers,
    )
    assert pr.status_code == 200
    assert pr.json()["data"]["description"] == "d1"
    assert pr.json()["data"]["chunk_size"] == 256

    sr = client.get(f"/api/knowledge/{kid}", headers={"Accept": "application/json"})
    assert sr.status_code == 200
    assert "chunk_strategy" in sr.json()["data"]
    assert sr.json()["data"]["chunk_strategy"]["strategy_type"] == "length"

    strat = {
        "version": 1,
        "common": {"chunk_size": 400, "chunk_overlap": 40, "trim_whitespace": True},
        "strategy_type": "length",
        "strategy_params": {"separators": ["\n\n", "\n"], "hard_limit": False},
    }
    pr2 = client.patch(
        f"/api/knowledge/{kid}",
        json={"chunk_strategy": strat},
        headers=headers,
    )
    assert pr2.status_code == 200
    assert pr2.json()["data"]["chunk_size"] == 400
    assert pr2.json()["data"]["chunk_overlap"] == 40
    cj = pr2.json()["data"].get("config_json") or {}
    assert isinstance(cj.get("chunk_strategy"), dict)
    assert cj["chunk_strategy"]["strategy_type"] == "length"

    dr = client.delete(f"/api/knowledge/{kid}", headers={"Accept": "application/json"})
    assert dr.status_code == 204

    gr2 = client.get(f"/api/knowledge/{kid}", headers={"Accept": "application/json"})
    assert gr2.status_code == 404
