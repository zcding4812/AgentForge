from fastapi.testclient import TestClient


def test_monitor_dashboard(client: TestClient) -> None:
    response = client.get("/api/system/monitor-dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "ok"
    data = body["data"]
    assert "health" in data
    assert data["health"]["status"] in ("ok", "degraded")
    assert isinstance(data["infrastructure"], list)
    assert len(data["infrastructure"]) >= 1
    assert "address" in data["infrastructure"][0]
    assert "token_usage" in data
    assert isinstance(data["token_usage"], list)
    assert "token_usage_grand_total" in data
    assert isinstance(data["token_usage_grand_total"], int)


def test_monitor_dashboard_token_days(client: TestClient) -> None:
    response = client.get("/api/system/monitor-dashboard?token_days=7")
    assert response.status_code == 200
    assert len(response.json()["data"]["token_usage"]) == 7
