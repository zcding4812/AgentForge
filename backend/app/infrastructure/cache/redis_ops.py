"""异步 Redis 常用操作：统一异常吞掉并打日志，业务侧降级。"""

from __future__ import annotations

import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# 缓存层降级：除 RedisError 外，捕获连接断开、事件循环关闭及 Windows/asyncio 在失效传输上写入时的 TypeError（redis-py #3546）。
_REDIS_CACHE_SOFT_ERRORS = (
    redis.RedisError,
    ConnectionError,
    OSError,
    RuntimeError,
    TypeError,
)


async def get_text(client: redis.Redis, key: str) -> str | None:
    """GET 文本；miss 或 Redis 异常时返回 ``None``。"""
    try:
        raw = await client.get(key)
    except _REDIS_CACHE_SOFT_ERRORS:
        logger.warning("Redis GET 失败 | key=%s", key, exc_info=True)
        return None
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


async def setex_text(client: redis.Redis, key: str, ttl_seconds: int, text: str) -> None:
    try:
        await client.setex(key, ttl_seconds, text)
    except _REDIS_CACHE_SOFT_ERRORS:
        logger.warning("Redis SETEX 失败 | key=%s", key, exc_info=True)


async def delete_key(client: redis.Redis, key: str) -> None:
    try:
        await client.delete(key)
    except _REDIS_CACHE_SOFT_ERRORS:
        logger.warning("Redis DEL 失败 | key=%s", key, exc_info=True)


async def delete_pattern(client: redis.Redis, pattern: str) -> None:
    """``SCAN`` + ``DEL``，用于一类 key 批量失效。"""
    try:
        async for k in client.scan_iter(match=pattern):
            await client.delete(k)
    except _REDIS_CACHE_SOFT_ERRORS:
        logger.warning("Redis SCAN/DEL 失败 | pattern=%s", pattern, exc_info=True)
