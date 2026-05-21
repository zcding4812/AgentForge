"""实时推送：WebSocket 会话编排与主题操作（与 :class:`app.core.realtime.TopicBroker` 协作）。"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from app.core.realtime import TopicBroker

# 与前端画布约定：仅允许以此前缀发布的主题，避免未鉴权连接滥用为任意广播。
_PIXEL_TOPIC_PREFIX = "pixel:"
_MAX_WS_TEXT_BYTES = 64 * 1024


class RealtimeService:
    """进程内 Pub/Sub 的用例层：解析客户端帧、调用 Broker、下行 JSON 文本。"""

    def __init__(self, broker: TopicBroker) -> None:
        self._broker = broker

    async def serve_websocket(self, websocket: WebSocket) -> None:
        """接受连接并循环处理文本帧，直至对端断开。"""
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                await self._handle_client_text(websocket, raw)
        except WebSocketDisconnect:
            await self._broker.disconnect(websocket)

    async def _handle_client_text(self, websocket: WebSocket, raw: str) -> None:
        if len(raw.encode("utf-8")) > _MAX_WS_TEXT_BYTES:
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": "frame_too_large"},
                    ensure_ascii=False,
                )
            )
            return
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "invalid_json"}, ensure_ascii=False)
            )
            return
        mtype = str(msg.get("type", "")).lower()
        if mtype == "ping":
            await websocket.send_text(
                json.dumps({"type": "pong", "ts": time.time()}, ensure_ascii=False)
            )
            return
        if mtype == "subscribe":
            topics = msg.get("topics")
            if not isinstance(topics, list):
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "topics_must_be_array"},
                        ensure_ascii=False,
                    )
                )
                return
            tlist = [str(x) for x in topics if isinstance(x, (str, int))]
            await self._broker.subscribe(websocket, tlist)
            await websocket.send_text(
                json.dumps({"type": "subscribed", "topics": tlist}, ensure_ascii=False)
            )
            return
        if mtype == "unsubscribe":
            topics = msg.get("topics")
            if not isinstance(topics, list):
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "topics_must_be_array"},
                        ensure_ascii=False,
                    )
                )
                return
            tlist = [str(x) for x in topics if isinstance(x, (str, int))]
            await self._broker.unsubscribe(websocket, tlist)
            await websocket.send_text(
                json.dumps({"type": "unsubscribed", "topics": tlist}, ensure_ascii=False)
            )
            return
        if mtype == "publish":
            topic_raw = msg.get("topic")
            if not isinstance(topic_raw, str):
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "topic_must_be_str"},
                        ensure_ascii=False,
                    )
                )
                return
            topic = topic_raw.strip()
            if not topic.startswith(_PIXEL_TOPIC_PREFIX) or len(topic) > 256:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "invalid_or_forbidden_topic"},
                        ensure_ascii=False,
                    )
                )
                return
            raw_data = msg.get("data")
            data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
            n = await self._broker.publish(topic, data)
            await websocket.send_text(
                json.dumps(
                    {"type": "published", "topic": topic, "delivered": n},
                    ensure_ascii=False,
                )
            )
            return
        await websocket.send_text(
            json.dumps(
                {"type": "error", "message": f"unknown_type:{mtype}"},
                ensure_ascii=False,
            )
        )

    async def publish_demo_payload(self, topic: str, message: str) -> dict[str, Any]:
        """向主题推送演示负载（供联调路由调用）。"""
        n = await self._broker.publish(topic, {"message": message})
        return {"ok": True, "topic": topic, "delivered": n}

    async def publish_demo_if_debug(
        self, *, debug: bool, topic: str, message: str
    ) -> dict[str, Any]:
        """``DEBUG`` 关闭时 ``404``，避免误暴露。"""
        if not debug:
            raise HTTPException(status_code=404, detail="Not found")
        return await self.publish_demo_payload(topic, message)
