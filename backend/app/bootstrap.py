"""
应用级启动 / 关闭：业务钩子（与 ``core.lifespan`` 基础设施分离）。

依赖方向：``lifespan`` → 本模块 → 业务层；``core`` 仅对 ``CreateRollingSummaryWorker`` 做类型引用，
``run_application_*`` 由 ``lifespan`` 内延迟 ``import``，避免加载 ``core.lifespan`` 时拉全业务依赖。
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from fastapi import FastAPI

from app.agent.adapters.tools.builtins import register_builtin_tools
from app.agent.adapters.tools.registry import aclose_all_registered_tools
from app.config import get_settings
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.infrastructure.http_client import aclose_embedding_http_client
from app.infrastructure.jobs.rolling_summary_broker import (
    RollingSummaryWorker,
    set_rolling_summary_worker,
)
from app.infrastructure.mongo import MongoDatabaseManager
from app.infrastructure.redis import RedisManager
from app.knowledge.adapters import get_vector_index_port
from app.repositories.chunk_repo import ChunkRepository
from app.services.ext_tool_svc import replay_persisted_external_tools


# ------------------------------------------------------------------------------
# Protocol 类型定义
# ------------------------------------------------------------------------------
class CreateRollingSummaryWorker(Protocol):
    """由 ``main`` 注入，内部可依赖 ``ConversationService`` 等。"""

    def __call__(
        self,
        db_manager: SQLAlchemyDatabaseManager,
        redis_client: object | None,
        *,
        max_queue_size: int = 512,
    ) -> RollingSummaryWorker: ...


# ------------------------------------------------------------------------------
# 工具函数：统一清理 app.state
# ------------------------------------------------------------------------------
def _cleanup_app_state(app: FastAPI, *keys: str) -> None:
    for key in keys:
        if hasattr(app.state, key):
            delattr(app.state, key)


# ------------------------------------------------------------------------------
# Knowledge 模块启动/关闭
# ------------------------------------------------------------------------------
async def _startup_knowledge(app: FastAPI) -> None:
    settings = get_settings()
    mongo = MongoDatabaseManager.get_instance(settings.mongodb_url)
    mongo_db = mongo.get_database(settings.mongodb_database)

    chunk_repo = ChunkRepository(mongo_db)
    await chunk_repo.ensure_indexes()

    app.state.chunk_repository = chunk_repo
    app.state.vector_index_port = get_vector_index_port(settings)


async def _shutdown_knowledge(app: FastAPI) -> None:
    await aclose_embedding_http_client()
    _cleanup_app_state(app, "chunk_repository", "vector_index_port")


# ------------------------------------------------------------------------------
# Agent Tools 启动/关闭
# ------------------------------------------------------------------------------
async def _startup_agent_tools(app: FastAPI) -> None:
    register_builtin_tools()
    await replay_persisted_external_tools(app.state.db_manager)


async def _shutdown_agent_tools() -> None:
    await aclose_all_registered_tools()


# ------------------------------------------------------------------------------
# Rolling Summary Worker 启动/关闭
# ------------------------------------------------------------------------------
async def _startup_rolling_worker(app: FastAPI, factory: CreateRollingSummaryWorker) -> None:
    worker = factory(
        app.state.db_manager,
        RedisManager.get_client(),
        max_queue_size=512,
    )
    await worker.start()

    app.state.rolling_summary_worker = worker
    set_rolling_summary_worker(worker)


async def _shutdown_rolling_worker(app: FastAPI) -> None:
    worker: RollingSummaryWorker | None = getattr(app.state, "rolling_summary_worker", None)
    if worker:
        await worker.stop()

    set_rolling_summary_worker(None)
    _cleanup_app_state(app, "rolling_summary_worker")


# ------------------------------------------------------------------------------
# 对外暴露：统一启动 / 关闭入口
# ------------------------------------------------------------------------------
async def run_application_startup(
    app: FastAPI,
    *,
    create_rolling_summary_worker: CreateRollingSummaryWorker,
) -> None:
    """业务层初始化：在基础设施就绪后调用。"""
    app.state.asyncio_loop = asyncio.get_running_loop()

    await _startup_knowledge(app)
    await _startup_agent_tools(app)
    await _startup_rolling_worker(app, create_rolling_summary_worker)


async def run_application_shutdown(app: FastAPI) -> None:
    """业务层逆序释放。"""
    await _shutdown_rolling_worker(app)
    await _shutdown_agent_tools()
    await _shutdown_knowledge(app)


__all__ = [
    "CreateRollingSummaryWorker",
    "run_application_startup",
    "run_application_shutdown",
]
