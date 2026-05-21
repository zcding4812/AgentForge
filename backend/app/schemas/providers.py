"""LLM 提供商（Provider）与挂载模型：Pydantic 模型，对应 `sys_model_provider` / `sys_model`。"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.schemas.response import PagedData, PageMeta

HealthStatus = Literal["ok", "error", "checking", "unknown"]


class ModelProviderCreate(BaseModel):
    """创建供应商（兼容旧表单字段，写入 sys_model_provider）。"""

    id: str = Field(
        min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$", description="provider_code"
    )
    name: str = Field(min_length=1, max_length=64)
    supported_model_types: list[str] = Field(
        default_factory=list,
        description="展示用；实际类型由子模型行决定，此处可不落库",
    )
    description: str | None = None
    enabled: bool = Field(
        default=False,
        description="对应 status；默认不启用，仅在有密钥等凭据后可启用",
    )
    provider_kind: str = Field(default="openai", description="写入 api_format")
    protocol: str = Field(
        default="openai_api", description="旧字段，映射为 api_format 时优先于 provider_kind"
    )
    base_url: str = Field(min_length=1)
    api_key: str = Field(default="", description="API Key；与供应商表存储")
    org_id: str | None = None
    weight: int = Field(default=50, ge=0, le=10000)
    timeout_sec: int = Field(default=30, ge=1, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    is_default: bool = False


class ModelProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    supported_model_types: list[str] | None = None
    description: str | None = None
    enabled: bool | None = None
    provider_kind: str | None = None
    protocol: str | None = None
    base_url: str | None = None
    api_key: str | None = Field(default=None, description="留空则不修改")
    api_secret: str | None = None
    org_id: str | None = None
    weight: int | None = Field(default=None, ge=0, le=10000)
    timeout_sec: int | None = Field(default=None, ge=1, le=600)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    is_default: bool | None = None


class ModelProviderOut(BaseModel):
    id: str = Field(description="provider_code，对外主键")
    name: str
    supported_model_types: list[str]
    description: str | None = None
    enabled: bool
    model_count: int
    provider_kind: str = Field(description="与 api_format 一致，供列表展示")
    protocol: str = Field(description="兼容前端：由 api_format 推导")
    base_url: str
    api_key_masked: str
    org_id: str | None = None
    weight: int = 50
    timeout_sec: int = 30
    max_retries: int = 2
    is_default: bool = False


class LlmModelCreate(BaseModel):
    """与 `sys_model` 列对齐；请求体仍兼容旧字段 id / name / api_url。"""

    model_config = ConfigDict(populate_by_name=True)

    model_code: str = Field(
        min_length=2,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="对应 sys_model.model_code",
        validation_alias=AliasChoices("model_code", "id"),
    )
    model_name: str = Field(
        min_length=1,
        max_length=128,
        description="对应 sys_model.model_name",
        validation_alias=AliasChoices("model_name", "name"),
    )
    provider_id: str = Field(description="供应商 provider_code 或数字 id → sys_model.provider_id")
    model_type: str = Field(min_length=1, max_length=32, description="对应 sys_model.model_type")
    endpoint: str = Field(
        default="",
        description="对应 sys_model.endpoint；空字符串表示 NULL，请求时走提供商 base_url",
        validation_alias=AliasChoices("endpoint", "api_url"),
    )
    timeout: int | None = Field(
        default=None,
        ge=1,
        le=86400,
        description="对应 sys_model.timeout（秒）；省略则默认 30",
    )
    enabled: bool = Field(
        default=False,
        description="对应 sys_model.is_enabled；默认不启用，仅当所属提供商已配置凭据时可启用",
    )


class LlmModelUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("model_name", "name"),
    )
    provider_id: str | None = None
    model_type: str | None = Field(default=None, min_length=1, max_length=32)
    endpoint: str | None = Field(
        default=None,
        description="对应 sys_model.endpoint；传空字符串可清空为 NULL",
        validation_alias=AliasChoices("endpoint", "api_url"),
    )
    timeout: int | None = Field(default=None, ge=1, le=86400, description="对应 sys_model.timeout")
    enabled: bool | None = Field(default=None, description="对应 sys_model.is_enabled")


class LlmModelOut(BaseModel):
    """与 `sys_model` + 关联提供商展示字段一致。"""

    id: str = Field(description="sys_model.id，更新/删除/探测路径参数")
    model_code: str
    model_name: str
    provider_id: str = Field(description="sys_model_provider.provider_code")
    provider_name: str
    model_type: str
    endpoint: str | None = Field(description="sys_model.endpoint；NULL 表示未单独配置")
    provider_base_url: str | None = Field(
        default=None,
        description="sys_model_provider.base_url，便于理解 endpoint 为空时的继承关系",
    )
    timeout: int = Field(description="sys_model.timeout（秒）")
    api_key_masked: str = Field(description="来自提供商表，模型行不单独存 key")
    enabled: bool = Field(description="sys_model.is_enabled")
    health_status: HealthStatus = "unknown"
    health_message: str | None = None


class ProbeResult(BaseModel):
    ok: bool
    message: str | None = None


def paged_providers(
    items: list[ModelProviderOut], page: int, page_size: int, total: int
) -> PagedData[ModelProviderOut]:
    return PagedData(
        items=items,
        meta=PageMeta(page=page, page_size=page_size, total=total),
    )


def paged_models(
    items: list[LlmModelOut], page: int, page_size: int, total: int
) -> PagedData[LlmModelOut]:
    return PagedData(
        items=items,
        meta=PageMeta(page=page, page_size=page_size, total=total),
    )
