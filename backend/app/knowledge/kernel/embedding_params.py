"""嵌入连接参数（纯数据）；由服务层从 ORM 映射后传入适配器，适配器不依赖 ORM。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingModelBinding:
    """嵌入模型侧可序列化字段。"""

    model_code: str
    endpoint: str
    model_type: str


@dataclass(frozen=True, slots=True)
class EmbeddingProviderBinding:
    """提供商侧可序列化字段。"""

    base_url: str
    api_key: str | None
    api_format: str
