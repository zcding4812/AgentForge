"""API 路由包：在此聚合各子模块 `router` 并统一 `prefix` / `tags`；`main` 使用 `from app.api import api_router`。

- `/health`：系统探活（例外，不在 `/api` 下）— `system.router`
- `/api/agents`：Agent（列表/创建/详情/PATCH/DELETE、默认提示词、已注册工具 `GET /tools`）— `agent.router`
- `/api/traces`：链路追踪 — `trace.router`
- `/api/providers`：LLM 提供商与挂载模型 — `provider_api.router`
- `/api/agent-conversations`：Agent 对话会话与消息持久化 — `conversation_api.router`
- `/api/knowledge`：知识库元数据 CRUD、文档列表与上传、检索、文档分片列表 — `knowledge_api.router`
- `/api/realtime`：WebSocket 主题订阅与推送 — `realtime_ws.router`

各子文件中的 `APIRouter` 仅声明相对路径（如 `/invoke`、`/models`）。
"""

from fastapi import APIRouter

from . import (
    agent_api,
    conversation_api,
    knowledge_api,
    provider_api,
    realtime_ws,
    system_api,
    trace_api,
    workspace_api,
)

api_router = APIRouter()
api_router.include_router(system_api.router)
api_router.include_router(trace_api.router, prefix="/api/traces", tags=["trace"])
api_router.include_router(agent_api.router, prefix="/api/agents", tags=["agents"])
api_router.include_router(
    workspace_api.router,
    prefix="/api/workspace-namespaces",
    tags=["workspace-namespaces"],
)
api_router.include_router(
    conversation_api.router,
    prefix="/api/agent-conversations",
    tags=["agent-conversations"],
)
api_router.include_router(provider_api.router, prefix="/api/providers", tags=["providers"])
api_router.include_router(knowledge_api.router, prefix="/api/knowledge", tags=["knowledge"])
api_router.include_router(realtime_ws.router, prefix="/api/realtime", tags=["realtime"])

__all__ = ["api_router"]
