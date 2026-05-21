"""MongoDB 基础设施：Motor 异步客户端与进程级管理（形态对齐 ``db.SQLAlchemyDatabaseManager``）。"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class MongoDatabaseManager:
    """单例 ``AsyncIOMotorClient``；由 ``lifespan`` 初始化，``close`` 释放。"""

    _instance: MongoDatabaseManager | None = None
    _client: AsyncIOMotorClient | None = None

    def __init__(self, connection_uri: str) -> None:
        cls = type(self)
        if cls._client is not None:
            return
        cls._client = AsyncIOMotorClient(connection_uri)

    @classmethod
    def get_instance(
        cls,
        connection_uri: str | None = None,
    ) -> MongoDatabaseManager:
        if cls._instance is None:
            if not connection_uri:
                raise RuntimeError("首次初始化必须传入 connection_uri")
            cls._instance = cls(connection_uri)
        return cls._instance

    @property
    def client(self) -> AsyncIOMotorClient:
        cls = type(self)
        if not cls._client:
            raise RuntimeError("Mongo 客户端未初始化")
        return cls._client

    def get_database(self, name: str) -> AsyncIOMotorDatabase:
        return self.client[name]

    async def check_connection(self) -> bool:
        """执行 ``ping`` 探活。"""
        cls = type(self)
        if not cls._client:
            logger.error("Mongo 客户端未初始化，无法检查连接（技术日志）")
            return False
        try:
            await self.client.admin.command("ping")
            logger.debug("Mongo 连接检查通过（技术日志）")
            return True
        except Exception as e:
            logger.error("Mongo 连接检查失败（技术异常）：%s", e, exc_info=True)
            return False

    @classmethod
    def close(cls) -> None:
        if cls._client is not None:
            cls._client.close()
        cls._instance = None
        cls._client = None
        logger.info("Mongo 客户端已关闭（技术日志）")


MongoManager = MongoDatabaseManager
