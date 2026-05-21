"""进程级异步 HTTP 客户端：连接池与上限，避免高并发下耗尽文件描述符。"""

from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None

# 限制并发连接与 keep-alive，防止 FD 与 TIME_WAIT 堆积
_DEFAULT_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_DEFAULT_TIMEOUT = httpx.Timeout(120.0)


def get_async_http_client() -> httpx.AsyncClient:
    """单进程共享 ``AsyncClient``；仅供基础设施与出站适配复用。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=_DEFAULT_LIMITS,
        )
    return _client


get_embedding_http_client = get_async_http_client
get_http_client = get_async_http_client


async def aclose_async_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


aclose_embedding_http_client = aclose_async_http_client
aclose_http_client = aclose_async_http_client
