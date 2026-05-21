"""工作台运行时上下文（ContextVar + 建造者）：供工具读取 DB、命名空间与子 Agent 调用。

顶层一次 ``invoke(workbench)`` 的典型链路（与 ``AgentService`` / ``prepare_invoke`` 对齐）：

1. **校验与对账**：``agent_kind=workbench`` 须带 ``agent_id``，且请求体 ``workspace_namespace`` 与入库实体一致。
2. **初始化运行时**：:meth:`WorkbenchRuntimeBuilder.top_level` 或 :meth:`WorkbenchRuntime.for_top_level` 注入 ``depth=0``。
3. **工具集**：根层工作台挂载内置 ``workbench_*``；子调用经 ``workbench_invoke_sub_agent`` 内 ``depth+1``。
4. **执行**：``WorkbenchStrategy``（ReAct）+ 工作台专属系统提示词。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.constants.agent import (
    WORKBENCH_PARALLEL_MAX_TASKS,
    WORKBENCH_SUB_AGENT_NESTED_OBSERVABILITY_MAX_CHARS,
    WORKBENCH_SUB_AGENT_TOOL_RESULT_MAX_CHARS,
)


class WorkbenchConstants:
    """与 ``app.core.constants`` 对齐的工作台契约常量。"""

    MAX_NEST_DEPTH: int = 3
    MAX_PARALLEL_TASKS: int = WORKBENCH_PARALLEL_MAX_TASKS
    MAX_PARENT_MESSAGE_TURNS: int = 50
    TOOL_RESULT_MAX_CHARS: int = WORKBENCH_SUB_AGENT_TOOL_RESULT_MAX_CHARS
    NESTED_OBSERVABILITY_MAX_CHARS: int = WORKBENCH_SUB_AGENT_NESTED_OBSERVABILITY_MAX_CHARS
    CONTEXT_VAR_NAME: str = "workbench_runtime"


# 历史兼容：曾定义于 ``constants`` / 本模块
WORKBENCH_MAX_NEST_DEPTH: int = WorkbenchConstants.MAX_NEST_DEPTH

if TYPE_CHECKING:
    from app.infrastructure.db import SQLAlchemyDatabaseManager
    from app.schemas.agent import AgentInvokeRequest
    from app.services.agent_svc import AgentService


@dataclass(frozen=True, slots=True)
class WorkbenchRuntime:
    """单次 invoke/stream 内有效；由 :class:`WorkbenchContextHolder` 注入。"""

    db_manager: SQLAlchemyDatabaseManager
    namespace: str
    agent_service: AgentService
    parent_invoke: AgentInvokeRequest
    effective_request_id: str | None
    depth: int = 0
    max_depth: int = WorkbenchConstants.MAX_NEST_DEPTH
    redis: object | None = None
    chunk_repository: object | None = None
    stream_event_queue: asyncio.Queue[dict[str, Any]] | None = None
    root_conversation_session_id: str | None = None

    @classmethod
    def for_top_level(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        namespace: str,
        agent_service: AgentService,
        parent_invoke: AgentInvokeRequest,
        effective_request_id: str | None,
        redis: object | None = None,
        chunk_repository: object | None = None,
        stream_event_queue: asyncio.Queue[dict[str, Any]] | None = None,
        root_conversation_session_id: str | None = None,
    ) -> WorkbenchRuntime:
        """根层工作台会话（``depth=0``）。委托 :meth:`WorkbenchRuntimeBuilder.top_level`。"""
        return WorkbenchRuntimeBuilder.top_level(
            db_manager=db_manager,
            namespace=namespace,
            agent_service=agent_service,
            parent_invoke=parent_invoke,
            effective_request_id=effective_request_id,
            redis=redis,
            chunk_repository=chunk_repository,
            stream_event_queue=stream_event_queue,
            root_conversation_session_id=root_conversation_session_id,
        )


class WorkbenchRuntimeBuilder:
    """建造者：统一顶层 / 嵌套运行时构造，避免参数字段遗漏。"""

    @staticmethod
    def top_level(
        *,
        db_manager: SQLAlchemyDatabaseManager,
        namespace: str,
        agent_service: AgentService,
        parent_invoke: AgentInvokeRequest,
        effective_request_id: str | None,
        redis: object | None = None,
        chunk_repository: object | None = None,
        stream_event_queue: asyncio.Queue[dict[str, Any]] | None = None,
        root_conversation_session_id: str | None = None,
    ) -> WorkbenchRuntime:
        return WorkbenchRuntime(
            db_manager=db_manager,
            namespace=namespace,
            agent_service=agent_service,
            parent_invoke=parent_invoke,
            effective_request_id=effective_request_id,
            depth=0,
            max_depth=WorkbenchConstants.MAX_NEST_DEPTH,
            redis=redis,
            chunk_repository=chunk_repository,
            stream_event_queue=stream_event_queue,
            root_conversation_session_id=root_conversation_session_id,
        )

    @staticmethod
    def nested(parent: WorkbenchRuntime) -> WorkbenchRuntime:
        """子调用：共享父级连接与流式队列，``depth+1``。"""
        return WorkbenchRuntime(
            db_manager=parent.db_manager,
            namespace=parent.namespace,
            agent_service=parent.agent_service,
            parent_invoke=parent.parent_invoke,
            effective_request_id=parent.effective_request_id,
            depth=parent.depth + 1,
            max_depth=parent.max_depth,
            redis=parent.redis,
            chunk_repository=parent.chunk_repository,
            stream_event_queue=parent.stream_event_queue,
            root_conversation_session_id=parent.root_conversation_session_id,
        )


_RUNTIME: ContextVar[WorkbenchRuntime | None] = ContextVar(
    WorkbenchConstants.CONTEXT_VAR_NAME, default=None
)


class WorkbenchContextHolder:
    """上下文持有者：封装 ContextVar，便于测试替换与显式依赖说明。"""

    @staticmethod
    def get() -> WorkbenchRuntime | None:
        return _RUNTIME.get()

    @staticmethod
    def require() -> WorkbenchRuntime:
        ctx = WorkbenchContextHolder.get()
        if ctx is None:
            msg = "工作台运行时未初始化（非 workbench 编排或作用域未设置）"
            raise RuntimeError(msg)
        return ctx

    @classmethod
    @asynccontextmanager
    async def scope(cls, ctx: WorkbenchRuntime):
        token = _RUNTIME.set(ctx)
        try:
            yield
        finally:
            _RUNTIME.reset(token)


def get_workbench_runtime() -> WorkbenchRuntime | None:
    return WorkbenchContextHolder.get()


def require_workbench_runtime() -> WorkbenchRuntime:
    return WorkbenchContextHolder.require()


# 与历史代码一致：``async with workbench_runtime_scope(wb):``
workbench_runtime_scope = WorkbenchContextHolder.scope
