"""进程级异步 Redis 客户端：连接管理见 :class:`RedisManager`。"""

from __future__ import annotations

import logging

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# 周期性 PING，归还连接池前剔除死连接；减轻 asyncio 在连接已断时 writelines 触发 TypeError 的概率（见 redis-py #3546）。
_DEFAULT_HEALTH_CHECK_INTERVAL_SEC = 30


class RedisManager:
    """异步 Redis 单例管理器，支持优雅启停、降级、依赖注入"""

    _client: redis.Redis | None = None

    @classmethod
    def configure(
        cls,
        url: str | None = None,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        decode_responses: bool = True,
    ) -> None:
        if cls._client is not None:
            logger.warning("Redis 客户端已初始化，忽略重复配置")
            return

        if not url or not url.strip():
            cls._client = None
            logger.info("Redis 未配置，已禁用缓存")
            return

        cls._client = redis.from_url(
            url.strip(),
            encoding="utf-8",
            decode_responses=decode_responses,
            max_connections=max_connections,
            socket_connect_timeout=socket_timeout,
            socket_timeout=socket_timeout,
            health_check_interval=_DEFAULT_HEALTH_CHECK_INTERVAL_SEC,
            retry_on_timeout=True,
        )
        logger.info("Redis 异步客户端初始化完成")

    @classmethod
    def get_client(cls) -> redis.Redis | None:
        return cls._client

    @classmethod
    async def ping(cls) -> bool:
        client = cls.get_client()
        if not client:
            return False
        try:
            return await client.ping()
        except RedisError:
            logger.exception("Redis ping 失败")
            return False
        except (ConnectionError, OSError, RuntimeError, TypeError):
            logger.warning("Redis ping 遇到连接/事件循环类错误（已视为不可用）", exc_info=True)
            return False

    @classmethod
    async def aclose(cls) -> None:
        client = cls.get_client()
        if client:
            await client.aclose()
            cls._client = None
            logger.info("Redis 连接已优雅关闭")
