"""嵌入单一入口（规格 EmbeddingAdapter）：带缓存的 ``KnowledgeEmbeddingClient`` 工厂。

检索 / 向量写入 / ingest 经 :meth:`EmbeddingAdapter.client_for_model_config` 获取客户端；
**禁止**在 ``services`` 内直接 ``new`` LlamaIndex 嵌入类。
"""

from __future__ import annotations

import asyncio

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.knowledge.adapters.embedding import KnowledgeEmbeddingClient, build_llama_index_embedding
from app.knowledge.kernel.embedding_params import (
    EmbeddingModelBinding,
    EmbeddingProviderBinding,
)
from app.knowledge.kernel.exceptions import EmbeddingAdapterError
from app.repositories.provider_repo import ProviderRepository

_cache: dict[int, KnowledgeEmbeddingClient] = {}
_lock = asyncio.Lock()


def _bindings_from_orm(
    model: object, prov: object
) -> tuple[EmbeddingModelBinding, EmbeddingProviderBinding]:
    return (
        EmbeddingModelBinding(
            model_code=(getattr(model, "model_code", None) or "").strip(),
            endpoint=(getattr(model, "endpoint", None) or "").strip(),
            model_type=(getattr(model, "model_type", None) or "").strip(),
        ),
        EmbeddingProviderBinding(
            base_url=(getattr(prov, "base_url", None) or "").strip(),
            api_key=(getattr(prov, "api_key", None) or "").strip() or None,
            api_format=(getattr(prov, "api_format", None) or "openai").strip().lower(),
        ),
    )


class EmbeddingAdapter:
    """按 ``embedding_model_config_id`` 解析 ORM 并返回可 ``embed_texts`` 的客户端（进程内缓存）。"""

    @staticmethod
    async def client_for_model_config(
        db_manager: SQLAlchemyDatabaseManager,
        embedding_model_config_id: int,
    ) -> KnowledgeEmbeddingClient:
        async with _lock:
            hit = _cache.get(embedding_model_config_id)
            if hit is not None:
                return hit

        mp = await ProviderRepository.get_model_with_provider(
            embedding_model_config_id,
            db_manager=db_manager,
        )
        if mp is None:
            raise EmbeddingAdapterError("嵌入模型配置不存在")
        model, prov = mp
        if (getattr(model, "model_type", None) or "").strip().lower() != "embedding":
            mt = getattr(model, "model_type", None)
            raise EmbeddingAdapterError(f"模型类型须为 embedding，当前为 {mt!r}")

        model_b, prov_b = _bindings_from_orm(model, prov)
        client = build_llama_index_embedding(model_b, prov_b)

        async with _lock:
            if embedding_model_config_id in _cache:
                return _cache[embedding_model_config_id]
            _cache[embedding_model_config_id] = client
            return client
