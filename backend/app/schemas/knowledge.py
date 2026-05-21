"""知识库 HTTP 契约（P0：CRUD）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from app.domain.knowledge.chunk_config import (
    ChunkStrategyConfig,
    chunk_strategy_from_kb_row,
)

StorageType = Literal["keyword", "vector", "hybrid"]
RetrievalType = Literal["keyword", "vector", "hybrid"]
KnowledgeLifecycleStatus = Literal["empty", "draft", "ready", "processing", "failed"]


class KnowledgeBaseOut(BaseModel):
    id: int
    namespace_id: int = Field(description="命名空间主键，对应表 namespace.id")
    workspace_namespace: str = "default"
    name: str
    slug: str
    description: str | None
    storage_type: str
    retrieval_type: str
    status: str
    embedding_model_config_id: int | None
    milvus_collection: str | None
    minio_prefix: str | None
    chunk_method: str
    chunk_size: int
    chunk_overlap: int
    chunk_separator: str | None
    config_json: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_strategy(self) -> dict[str, Any]:
        """结构化切块策略（由 ``config_json.chunk_strategy`` 与列字段推导）。"""
        return chunk_strategy_from_kb_row(self).model_dump(mode="json")


class KnowledgeListData(BaseModel):
    items: list[KnowledgeBaseOut]
    total: int
    page: int
    page_size: int


class KnowledgeDocumentOut(BaseModel):
    """知识库内单条文档目录；与 ``GET .../documents`` 等接口对齐时使用。"""

    id: int
    kb_id: int
    filename: str
    object_key: str | None
    size_bytes: int | None
    mime: str | None
    sha256: str | None
    status: str
    content_version: str = Field(default="", description="内容版本串，用于与派生存储幂等对齐")
    chunk_count: int = Field(ge=0, description="该文档已生成的分片条数")
    chunk_method: str | None = Field(
        default=None,
        description="文档级切块方式覆盖；为空则 ingest 使用知识库默认",
    )
    chunk_size: int | None = Field(default=None, description="文档级 chunk_size；为空则与库级对齐")
    chunk_overlap: int | None = Field(
        default=None, description="文档级 chunk_overlap；为空则与库级对齐"
    )
    chunk_separator: str | None = Field(default=None, description="文档级分隔串覆盖")
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeDocumentIngestAccepted(BaseModel):
    """上传或重新入队后的文档与 ingest 任务 id（``GET ...?async=true`` 时 HTTP 202）。"""

    document: KnowledgeDocumentOut
    task_id: str | None = Field(
        default=None,
        description="当次调度关联的 ingest 任务 task_id；幂等去重未新建时为 null",
    )


class KnowledgeDocumentPatchBody(BaseModel):
    """PATCH 文档：更新文档级切块覆盖；``chunk_method`` 显式 ``null`` 表示清除覆盖（恢复知识库默认）。"""

    chunk_method: str | None = Field(
        default=None,
        max_length=32,
        description="为 null 且本字段出现在请求中时表示清除文档级切块覆盖",
    )
    chunk_size: int | None = Field(default=None, ge=32, le=32768)
    chunk_overlap: int | None = Field(default=None, ge=0, le=8192)
    chunk_separator: str | None = Field(default=None, max_length=64)


class KnowledgeDocumentListData(BaseModel):
    """知识库文档分页列表。"""

    items: list[KnowledgeDocumentOut]
    total: int
    page: int
    page_size: int


class KnowledgeCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    workspace_namespace: str = Field(default="default", max_length=64, description="工作区命名空间")
    slug: str | None = Field(default=None, max_length=128, description="留空则自名称推导")
    description: str | None = Field(default=None, max_length=512)
    storage_type: StorageType = "keyword"
    retrieval_type: RetrievalType = "keyword"
    status: KnowledgeLifecycleStatus = "empty"
    embedding_model_config_id: int | None = None
    milvus_collection: str | None = Field(default=None, max_length=256)
    minio_prefix: str | None = Field(default=None, max_length=512)
    chunk_method: str = Field(
        default="length",
        max_length=32,
        description=(
            "分片策略键：length/char（字符窗）、token（词元）、sentence（句边界 token）、"
            "markdown/md（标题分节）、json、html、code（AST，按文件名推断语言）；未知键回退为 length"
        ),
    )
    chunk_size: int = Field(default=512, ge=32, le=32768)
    chunk_overlap: int = Field(default=50, ge=0, le=8192)
    chunk_separator: str | None = Field(default=None, max_length=64)
    config_json: dict[str, Any] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("workspace_namespace", mode="after")
    @classmethod
    def _strip_kb_workspace_namespace(cls, v: str) -> str:
        s = (v or "").strip()
        return s if s else "default"

    @field_validator("slug", mode="before")
    @classmethod
    def _strip_slug(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class KnowledgeSearchBody(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=100, description="最终返回条数（Top K）")
    doc_id: int | None = Field(default=None, description="若指定，仅在 Mongo 中该文档的分片内匹配")
    retrieval_override: RetrievalType | None = Field(
        default=None,
        description="覆盖知识库 ``retrieval_type``：单次请求使用关键词 / 向量 / 混合；不传则与库配置一致",
    )
    recall_limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="各子路召回上限（混合两路、向量降级前的关键词路同此上限）；须 ≥ limit；不传则混合路按默认公式、其余同 limit",
    )
    rrf_k: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="混合检索 RRF 常数 k（score 贡献 1/(k+rank)）；越大尾部排名影响越小。默认与 ``HYBRID_RRF_K`` 一致",
    )

    @model_validator(mode="after")
    def _recall_coherent(self) -> KnowledgeSearchBody:
        if self.recall_limit is not None and self.recall_limit < self.limit:
            raise ValueError("recall_limit 须大于等于 limit")
        return self


class KnowledgeSearchHit(BaseModel):
    doc_id: int
    chunk_index: int | None = None
    text_snippet: str | None = None
    filename: str | None = None
    match_type: Literal["chunk", "filename", "keyword", "vector"]
    #: 混合检索 RRF 合分；其它模式可为空
    score: float | None = Field(
        default=None,
        description="检索相关度分数（混合路为 RRF 合分；越大越靠前）",
    )


class KnowledgeSearchData(BaseModel):
    items: list[KnowledgeSearchHit]
    total: int
    configured_retrieval: RetrievalType = Field(
        description="知识库 ``retrieval_type``（索引配置）",
    )
    applied_retrieval: RetrievalType = Field(
        description="本次请求实际执行的检索类型（向量未接入时可能为 keyword）",
    )
    note: str | None = Field(
        default=None,
        description="降级或说明（如向量索引未启用时提示）",
    )


class KnowledgeSearchEnvelope(BaseModel):
    """检索门面 ``message`` + ``data`` 定型（与 HTTP 信封字段一致）。"""

    message: Literal["ok"] = "ok"
    data: KnowledgeSearchData


class KnowledgeVectorBuildData(BaseModel):
    """从 Mongo 已有分片构建 Milvus 向量（不重跑 ingest）的结果。"""

    built: int = Field(ge=0, description="成功写入 Milvus 的文档数")
    skipped: int = Field(ge=0, description="跳过数（无分片、缺 content_version 等）")
    errors: list[str] = Field(default_factory=list, description="跳过或单文档失败说明")


class KnowledgeChunkOut(BaseModel):
    """Mongo 分片条目的 API 形态。"""

    chunk_index: int
    text: str
    created_at: datetime | None = None


class KnowledgeDocumentChunksData(BaseModel):
    """某文档在指定 ``content_version`` 下的分片分页列表。"""

    items: list[KnowledgeChunkOut]
    total: int
    page: int
    page_size: int
    doc_id: int
    filename: str | None = None
    content_version: str
    document_status: str | None = Field(default=None, description="文档当前状态（MySQL）")
    empty_hint: str | None = Field(
        default=None,
        description="total=0 时可选说明（排队中/失败/需重建等）",
    )


class KnowledgeUpdateBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    storage_type: StorageType | None = None
    retrieval_type: RetrievalType | None = None
    status: KnowledgeLifecycleStatus | None = None
    embedding_model_config_id: int | None = None
    milvus_collection: str | None = Field(default=None, max_length=256)
    minio_prefix: str | None = Field(default=None, max_length=512)
    chunk_method: str | None = Field(
        default=None,
        max_length=32,
        description="见 KnowledgeCreateBody.chunk_method；未知键回退 length",
    )
    chunk_size: int | None = Field(default=None, ge=32, le=32768)
    chunk_overlap: int | None = Field(default=None, ge=0, le=8192)
    chunk_separator: str | None = Field(default=None, max_length=64)
    config_json: dict[str, Any] | None = None
    chunk_strategy: ChunkStrategyConfig | None = Field(
        default=None,
        description="结构化切块策略；写入 config_json.chunk_strategy 并同步 chunk_method/chunk_size/chunk_overlap",
    )
