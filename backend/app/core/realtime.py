"""进程内实时推送：主题订阅与广播（Pub/Sub，单文件）。

- **Topic**：字符串频道；**Subscriber**：WebSocket；**Publisher**：``await broker.publish(topic, data)``。
- 实例由 :mod:`app.core.lifespan` 挂在 ``app.state.topic_broker``；路由用 :class:`app.core.deps.TopicBrokerDep` 注入。
- 多 worker / 多机时需换 Redis Pub/Sub 等，本类可作单进程分发层。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

from starlette.websockets import WebSocket, WebSocketState


class TopicBroker:
    """asyncio 锁下维护 ``topic → WebSocket`` 集合。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._topic_peers: dict[str, set[WebSocket]] = defaultdict(set)
        self._peer_topics: dict[WebSocket, set[str]] = defaultdict(set)

    async def subscribe(self, ws: WebSocket, topics: list[str]) -> None:
        """将连接加入给定主题（幂等）。"""
        if not topics:
            return
        async with self._lock:
            for t in topics:
                key = str(t)
                self._topic_peers[key].add(ws)
                self._peer_topics[ws].add(key)

    async def unsubscribe(self, ws: WebSocket, topics: list[str]) -> None:
        async with self._lock:
            for t in topics:
                key = str(t)
                self._topic_peers[key].discard(ws)
                if not self._topic_peers[key]:
                    del self._topic_peers[key]
                self._peer_topics[ws].discard(key)
            if ws in self._peer_topics and not self._peer_topics[ws]:
                del self._peer_topics[ws]

    async def disconnect(self, ws: WebSocket) -> None:
        """连接关闭时移除其全部订阅。"""
        async with self._lock:
            topics = list(self._peer_topics.pop(ws, ()))
            for key in topics:
                self._topic_peers[key].discard(ws)
                if not self._topic_peers[key]:
                    del self._topic_peers[key]

    async def publish(self, topic: str, data: dict[str, Any]) -> int:
        """向主题所有订阅者推送；返回成功发送数。"""
        key = str(topic)
        async with self._lock:
            peers = list(self._topic_peers.get(key, ()))
        if not peers:
            return 0
        envelope = {
            "type": "event",
            "topic": key,
            "data": data,
            "ts": time.time(),
        }
        raw = json.dumps(envelope, ensure_ascii=False)
        sent = 0
        dead: list[WebSocket] = []
        for ws in peers:
            try:
                if ws.client_state != WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                await ws.send_text(raw)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)
        return sent


__all__ = ["TopicBroker"]
