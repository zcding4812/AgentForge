import asyncio

import pytest

from app.infrastructure.redis import RedisManager


def _run(coro):  # noqa: ANN001
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_redis_manager() -> None:
    _run(RedisManager.aclose())
    yield
    _run(RedisManager.aclose())


def test_configure_empty_url_yields_no_client() -> None:
    RedisManager.configure(None)
    assert RedisManager.get_client() is None
    assert _run(RedisManager.ping()) is False


def test_configure_blank_url_yields_no_client() -> None:
    RedisManager.configure("   ")
    assert RedisManager.get_client() is None
