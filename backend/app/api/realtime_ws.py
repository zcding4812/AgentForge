"""WebSocket：主题订阅与下行事件推送（薄路由，逻辑见 :mod:`app.services.realtime_svc`）。

协议（JSON 文本帧）：

- 客户端 → 服务端
  - ``{"type":"subscribe","topics":["notifications","agent:1"]}``
  - ``{"type":"unsubscribe","topics":["notifications"]}``
  - ``{"type":"publish","topic":"pixel:namespace:default","data":{...}}`` → 向主题广播；``topic`` 须以 ``pixel:`` 开头，供画布等上行事件使用
  - ``{"type":"ping"}`` → 服务端 ``{"type":"pong","ts":...}``
- 服务端 → 客户端（推送）
  - ``{"type":"event","topic":"...","data":{...},"ts":...}``

业务在路由或服务中注入 :class:`app.core.deps.TopicBrokerDep` 并 ``await broker.publish`` 即可下行推送。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket

from app.config import get_settings
from app.core.deps import TopicBrokerDep, TopicBrokerWsDep
from app.services.realtime_svc import RealtimeService

router = APIRouter()


def get_realtime_service(broker: TopicBrokerDep) -> RealtimeService:
    return RealtimeService(broker)


RealtimeServiceDep = Annotated[RealtimeService, Depends(get_realtime_service)]


def get_realtime_service_ws(broker: TopicBrokerWsDep) -> RealtimeService:
    return RealtimeService(broker)


RealtimeServiceWsDep = Annotated[RealtimeService, Depends(get_realtime_service_ws)]


@router.websocket("/ws")
async def realtime_websocket(
    websocket: WebSocket,
    svc: RealtimeServiceWsDep,
) -> None:
    await svc.serve_websocket(websocket)


@router.post("/publish-demo", summary="演示向主题推送（仅 DEBUG）")
async def publish_demo(
    svc: RealtimeServiceDep,
    topic: str,
    message: str = "hello",
) -> dict[str, Any]:
    return await svc.publish_demo_if_debug(
        debug=get_settings().debug,
        topic=topic,
        message=message,
    )
