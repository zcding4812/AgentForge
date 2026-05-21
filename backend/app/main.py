from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.lifespan import create_lifespan
from app.core.logger import setup_logging
from app.core.middleware import HttpMiddleware
from app.services.conversation_svc import create_rolling_summary_worker

_tags = [
    {"name": "system", "description": "系统状态与健康检查。"},
    {"name": "trace", "description": "链路追踪查询接口。"},
    {
        "name": "agents",
        "description": "Agent 编排与调用（非流式基线）；按类型获取默认系统提示词（GET /api/agents/default-prompts/{kind}）。",
    },
    {
        "name": "providers",
        "description": "LLM 提供商（Provider）与挂载模型（sys_model_provider / sys_model）。",
    },
    {
        "name": "agent-conversations",
        "description": "Agent 多轮对话会话与消息持久化",
    },
    {
        "name": "knowledge",
        "description": "知识库元数据、文档上传与列表、运维（RAG）；派生索引与异步清理见后续任务。",
    },
    {
        "name": "realtime",
        "description": "WebSocket 主题订阅（Pub/Sub）；业务通过 TopicBroker.publish 下行推送。",
    },
]


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings)

    # FastAPI(lifespan=...) 与官方示例一致；此处经工厂集中装配中间件与路由
    app = FastAPI(
        title="AI Agents API",
        description="AI Agents 平台后端 HTTP API。",
        version="0.1.0",
        openapi_tags=_tags,
        lifespan=create_lifespan(
            create_rolling_summary_worker=create_rolling_summary_worker,
        ),
        root_path=settings.root_path,
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # 注册中间件
    app.add_middleware(HttpMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册异常处理器
    register_exception_handlers(app)

    # 注册路由
    app.include_router(api_router)

    return app


app = create_app()
