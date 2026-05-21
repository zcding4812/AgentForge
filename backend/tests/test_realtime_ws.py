"""WebSocket 主题订阅与 TopicBroker.publish 集成测试。"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def debug_client() -> TestClient:
    os.environ["DEBUG"] = "true"
    try:
        from app.config import get_settings

        get_settings.cache_clear()
    except ImportError:
        pass
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("DEBUG", None)
    try:
        from app.config import get_settings

        get_settings.cache_clear()
    except ImportError:
        pass


def test_websocket_subscribe_and_receive_event(client: TestClient) -> None:
    broker = client.app.state.topic_broker
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({"type": "subscribe", "topics": ["demo"]}, ensure_ascii=False))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "subscribed"
        assert "demo" in ack["topics"]

        n = asyncio.run(broker.publish("demo", {"x": 1}))
        assert n == 1
        ev = json.loads(ws.receive_text())
        assert ev["type"] == "event"
        assert ev["topic"] == "demo"
        assert ev["data"] == {"x": 1}


def test_publish_demo_requires_debug(client: TestClient) -> None:
    r = client.post("/api/realtime/publish-demo", params={"topic": "t", "message": "m"})
    assert r.status_code == 404


def test_publish_demo_ok_when_debug(debug_client: TestClient) -> None:
    r = debug_client.post("/api/realtime/publish-demo", params={"topic": "t", "message": "m"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("delivered") == 0


def test_websocket_client_publish_to_pixel_topic(client: TestClient) -> None:
    """画布等客户端可经 WS 向 pixel: 前缀主题发布，与订阅方广播一致。"""
    with (
        client.websocket_connect("/api/realtime/ws") as sub,
        client.websocket_connect("/api/realtime/ws") as pub,
    ):
        sub.send_text(
            json.dumps(
                {"type": "subscribe", "topics": ["pixel:namespace:default"]}, ensure_ascii=False
            )
        )
        assert json.loads(sub.receive_text())["type"] == "subscribed"

        pub.send_text(
            json.dumps(
                {
                    "type": "publish",
                    "topic": "pixel:namespace:default",
                    "data": {"hello": 1, "s": "canvas"},
                },
                ensure_ascii=False,
            )
        )
        ok = json.loads(pub.receive_text())
        assert ok["type"] == "published"
        assert ok.get("delivered") == 1

        ev = json.loads(sub.receive_text())
        assert ev["type"] == "event"
        assert ev["topic"] == "pixel:namespace:default"
        assert ev["data"] == {"hello": 1, "s": "canvas"}


def test_websocket_publish_rejects_non_pixel_topic(client: TestClient) -> None:
    with client.websocket_connect("/api/realtime/ws") as pub:
        pub.send_text(
            json.dumps(
                {
                    "type": "publish",
                    "topic": "other:ns",
                    "data": {"x": 1},
                },
                ensure_ascii=False,
            )
        )
        err = json.loads(pub.receive_text())
        assert err["type"] == "error"
        assert "topic" in err.get("message", "")
