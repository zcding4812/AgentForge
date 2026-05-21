"""Pytest 全局：提供 TestClient；默认使用内存 SQLite，避免污染 MySQL 开发库。

- **默认**：未设置 ``TEST_DATABASE_URL`` 且未设 ``TEST_USE_MYSQL=1`` 时，将 ``DATABASE_URL``
  设为 ``sqlite+aiosqlite:///:memory:``，启动时在 lifespan 内 ``create_all``（无需 Alembic）。
- **MySQL 集成测试**：设置 ``TEST_USE_MYSQL=1`` 使用 ``.env`` 中的 ``DATABASE_URL``；或设置
  ``TEST_DATABASE_URL`` 指向专用测试库（如 ``mysql+pymysql://...@/ai_agents_test``）。

知识库上传等测试仍需要 MinIO 等环境；否则相关接口会失败。未设置 ``MONGODB_URL`` 时 pytest 默认指向本机
``127.0.0.1``，并附带 ``serverSelectionTimeoutMS=2000``，避免 lifespan 内 ``ensure_indexes`` 在无 Mongo 时每用例卡约 30s。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


def pytest_configure(config: pytest.Config) -> None:
    test_url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if test_url:
        os.environ["DATABASE_URL"] = test_url
    elif os.environ.get("TEST_USE_MYSQL", "").strip() != "1":
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    # TEST_USE_MYSQL=1：沿用环境已有 DATABASE_URL

    if not os.environ.get("MONGODB_URL", "").strip():
        # 未显式配置时缩短 server selection，避免 lifespan 里 ensure_indexes 卡满默认 ~30s/次
        os.environ["MONGODB_URL"] = "mongodb://127.0.0.1:27017/?serverSelectionTimeoutMS=2000"
    try:
        from app.config import get_settings

        get_settings.cache_clear()
    except ImportError:
        pass


@pytest.fixture
def client() -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
