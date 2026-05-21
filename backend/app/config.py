from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants.agent import AGENT_GRAPH_RECURSION_LIMIT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    agent_graph_recursion_limit: int = Field(
        default=AGENT_GRAPH_RECURSION_LIMIT,
        ge=8,
        le=512,
        alias="AGENT_GRAPH_RECURSION_LIMIT",
        description=(
            "LangGraph 单轮 recursion_limit（模型/工具步数）；"
            "多子 Agent 编排或长 ReAct 不足时可调大，过大可能拖长单次请求"
        ),
    )

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")
    debug: bool = Field(default=False, alias="DEBUG")
    api_docs_enabled: bool = Field(default=True, alias="API_DOCS_ENABLED")
    root_path: str = Field(default="", alias="ROOT_PATH")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    database_url: str = Field(default="", alias="DATABASE_URL")
    db_pool_pre_ping: bool = Field(
        default=True,
        alias="DB_POOL_PRE_PING",
        description="连接池取出连接前是否预 ping；关可略降延迟，但可能使用已断开的连接（生产慎用 false）",
    )

    redis_url: str = Field(
        default="",
        alias="REDIS_URL",
        description="异步 Redis 连接串；留空表示不启用 Redis（业务侧须降级为直连 DB 等）",
    )
    redis_max_connections: int = Field(
        default=50,
        ge=1,
        alias="REDIS_MAX_CONNECTIONS",
        description="redis.asyncio 连接池上限",
    )

    #: MinIO（S3 兼容）— 知识库对象存储；以下三项为启用知识库上传的**必填**环境变量；endpoint 不含 scheme，如 ``127.0.0.1:9000``
    minio_endpoint: str = Field(default="", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", alias="MINIO_SECRET_KEY")
    minio_bucket_knowledge: str = Field(
        default="knowledge",
        alias="MINIO_BUCKET_KNOWLEDGE",
        description="知识库原始文件所用桶名",
    )

    mongodb_url: str = Field(
        ...,
        alias="MONGODB_URL",
        description="MongoDB 连接串（知识库分片）；必填",
    )
    mongodb_database: str = Field(
        default="ai_agents",
        alias="MONGODB_DATABASE",
        description="MongoDB 数据库名",
    )

    #: Milvus 连接 URI，如 ``http://127.0.0.1:19530``；留空则不做向量入库与 ANN 检索
    milvus_uri: str = Field(default="", alias="MILVUS_URI")
    milvus_token: str = Field(
        default="", alias="MILVUS_TOKEN", description="Milvus 鉴权 token（可选）"
    )

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url.strip())

    @property
    def minio_configured(self) -> bool:
        return bool(
            self.minio_endpoint.strip() and self.minio_access_key and self.minio_secret_key,
        )

    @property
    def milvus_configured(self) -> bool:
        return bool(self.milvus_uri.strip())

    knowledge_vector_search_use_li_nodes: bool = Field(
        default=False,
        alias="KNOWLEDGE_VECTOR_SEARCH_USE_LI_NODES",
        description=(
            "向量检索经 Milvus 返回的 TextNode 路径再映射为命中（与 tuple ANN 等价，便于接官方 Retriever）"
        ),
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    @property
    def async_database_url(self) -> str:
        """运行时异步引擎 URL（MySQL 由 ``normalize_mysql_database_url``；SQLite ``sqlite+aiosqlite`` 直通）。"""
        from app.infrastructure.db import normalize_database_url_async

        return normalize_database_url_async(self.database_url)


@cache
def get_settings() -> Settings:
    return Settings()
