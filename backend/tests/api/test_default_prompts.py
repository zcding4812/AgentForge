from fastapi.testclient import TestClient

from app.agent.kernel.default_prompts import get_default_prompt_entry


def test_get_default_prompt_by_kind(client: TestClient) -> None:
    r = client.get("/api/agents/default-prompts/system", headers={"Accept": "application/json"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["id"] == "system"
    assert data["text"] == get_default_prompt_entry("system").text


def test_get_default_prompt_unknown_404(client: TestClient) -> None:
    r = client.get(
        "/api/agents/default-prompts/__no_such__", headers={"Accept": "application/json"}
    )
    assert r.status_code == 404
