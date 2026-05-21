from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "ok"
    assert body["data"]["status"] == "ok"


def test_x_request_id_roundtrip(client: TestClient) -> None:
    rid = "abcdef1234567890abcdef1234567890"
    response = client.get("/health", headers={"X-Request-Id": rid})
    assert response.headers.get("X-Request-Id") == rid


def test_unhandled_exception_returns_500_with_request_id(client: TestClient) -> None:
    from app.main import app

    if not any(route.path == "/_test/error" for route in app.routes):

        @app.get("/_test/error")
        async def _raise_error() -> None:
            raise RuntimeError("boom")

    error_client = TestClient(app, raise_server_exceptions=False)
    rid = "fedcba0987654321fedcba0987654321"
    response = error_client.get("/_test/error", headers={"X-Request-Id": rid})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    assert response.headers.get("X-Request-Id") == rid
