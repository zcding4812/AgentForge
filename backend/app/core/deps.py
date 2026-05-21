from collections.abc import AsyncIterator
from typing import Annotated

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocket

from app.config import Settings, get_settings
from app.core.realtime import TopicBroker
from app.infrastructure.db import DatabaseManager
from app.infrastructure.minio import MinioObjectStore

# ====================== 配置 ======================
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ====================== DB ======================
def get_db_manager(request: Request) -> DatabaseManager:
    db_manager = getattr(request.app.state, "db_manager", None)
    if db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DatabaseManager 未初始化",
        )
    return db_manager


DBManagerDep = Annotated[DatabaseManager, Depends(get_db_manager)]


async def get_db(db_manager: DBManagerDep) -> AsyncIterator[AsyncSession]:
    async with db_manager.session_factory() as session:
        yield session


DBSessionDep = Annotated[AsyncSession, Depends(get_db)]


# ====================== Redis ======================
def get_redis_client(request: Request) -> redis.Redis | None:
    """注入进程级异步 Redis 客户端；未配置 ``REDIS_URL`` 时为 ``None``。"""
    return getattr(request.app.state, "redis_client", None)


OptionalRedisDep = Annotated[redis.Redis | None, Depends(get_redis_client)]


def get_redis_client_required(request: Request) -> redis.Redis:
    """依赖方**必须**使用 Redis 时注入；未启用则 ``503``。"""
    client = get_redis_client(request)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis 未配置或不可用",
        )
    return client


RedisRequiredDep = Annotated[redis.Redis, Depends(get_redis_client_required)]


# ====================== MinIO ======================
def get_minio_object_store(request: Request) -> MinioObjectStore:
    """进程级 MinIO 封装（``lifespan`` 写入 ``app.state.minio_object_store``）。"""
    store = getattr(request.app.state, "minio_object_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MinIOObjectStore 未初始化",
        )
    return store


MinioObjectStoreDep = Annotated[MinioObjectStore, Depends(get_minio_object_store)]


# ====================== 实时推送（WebSocket Pub/Sub） ======================
def get_topic_broker(request: Request) -> TopicBroker:
    """进程内主题广播器；由 ``lifespan`` 写入 ``app.state.topic_broker``。"""
    broker = getattr(request.app.state, "topic_broker", None)
    if broker is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TopicBroker 未初始化",
        )
    return broker


TopicBrokerDep = Annotated[TopicBroker, Depends(get_topic_broker)]


def get_topic_broker_ws(websocket: WebSocket) -> TopicBroker:
    """WebSocket 路由专用：从 ``websocket.app.state`` 取与 HTTP 相同的实例。"""
    broker = getattr(websocket.app.state, "topic_broker", None)
    if broker is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TopicBroker 未初始化",
        )
    return broker


TopicBrokerWsDep = Annotated[TopicBroker, Depends(get_topic_broker_ws)]
