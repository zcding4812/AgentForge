"""
应用生命周期：仅 **基础设施** —— 连接 / 客户端 / 池 的创建与释放。

业务初始化见 :mod:`app.bootstrap` 模块（``run_application_startup`` / ``run_application_shutdown``），
由本模块在基础设施就绪后调用，避免 ``core`` 直接依赖 ``repositories`` / ``services`` / ``knowledge`` 实现。

依赖方向：``lifespan`` → ``bootstrap`` → 业务层；无反向依赖。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from app.config import get_settings

if TYPE_CHECKING:
    from app.bootstrap import CreateRollingSummaryWorker
from app.core.realtime import TopicBroker
from app.infrastructure.db import Base, SQLAlchemyDatabaseManager
from app.infrastructure.minio import MinioConfig, MinioObjectStore
from app.infrastructure.mongo import MongoDatabaseManager
from app.infrastructure.redis import RedisManager


def create_lifespan(
    *,
    create_rolling_summary_worker: CreateRollingSummaryWorker,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 延迟导入：保持 ``core.lifespan`` 模块加载时不触碰业务包
        from app.bootstrap import run_application_shutdown, run_application_startup

        settings = get_settings()

        # ---------- 启动：仅基础设施 ----------
        app.state.db_manager = SQLAlchemyDatabaseManager.get_instance(
            settings.async_database_url,
            pool_config={"pool_pre_ping": settings.db_pool_pre_ping},
        )

        if settings.database_url.strip().lower().startswith("sqlite"):
            from app import models as _orm_models  # noqa: F401 — 注册 ORM 到 metadata

            async with app.state.db_manager.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        RedisManager.configure(
            settings.redis_url if settings.redis_enabled else None,
            max_connections=settings.redis_max_connections,
        )
        app.state.redis_client = RedisManager.get_client()

        app.state.minio_object_store = MinioObjectStore(MinioConfig.from_settings(settings))

        app.state.topic_broker = TopicBroker()

        MongoDatabaseManager.get_instance(settings.mongodb_url)

        await run_application_startup(
            app,
            create_rolling_summary_worker=create_rolling_summary_worker,
        )

        try:
            yield
        finally:
            await run_application_shutdown(app)
            await RedisManager.aclose()
            if hasattr(app.state, "topic_broker"):
                delattr(app.state, "topic_broker")
            if hasattr(app.state, "minio_object_store"):
                delattr(app.state, "minio_object_store")
            MongoDatabaseManager.close()
            await SQLAlchemyDatabaseManager.close()

    return lifespan
