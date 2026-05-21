"""嵌入适配：仅根据 kernel DTO 构造 LlamaIndex 官方嵌入与薄客户端（无 ORM、无缓存）。

业务请用 :class:`~app.knowledge.adapters.embedding_adapter.EmbeddingAdapter`。
"""

from __future__ import annotations

from typing import cast

from llama_index.core.embeddings import BaseEmbedding as LlamaBaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding as LIOllamaEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding as LIOpenAIEmbedding,
)
from llama_index.embeddings.openai import (
    OpenAIEmbeddingModelType,
)
from openai import AsyncOpenAI, OpenAI

from app.knowledge.kernel.embedding_params import EmbeddingModelBinding, EmbeddingProviderBinding
from app.knowledge.kernel.exceptions import EmbeddingAdapterError

# DashScope OpenAI 兼容 embeddings 接口：单次 input 条数上限（超出返回 InvalidParameter）
DASHSCOPE_COMPAT_EMBEDDING_BATCH_MAX = 10


def _infer_embed_type(api_format: str, base_url: str) -> str:
    """按 ``api_format`` 与 ``base_url`` 推断嵌入后端类型（ollama / ali / openai）。

    说明：**向量数据库**（如 Milvus、本地 Milvus）只存向量，不在此推断；本函数面向
    **embedding 推理** 服务。本地/内网常见形态：Ollama、OpenAI 兼容网关、TEI 等，均归入
    上述三类之一（TEI 与 LM Studio 等走 ``openai`` 与 ``LIOpenAIEmbedding``）。
    """
    af = (api_format or "openai").strip().lower()
    if af == "ollama":
        return "ollama"
    if af in ("ali", "dashscope", "aliyun"):
        return "ali"
    # 显式：本机或内网 OpenAI 兼容 embedding（TEI、vLLM embedding、LM Studio 等）
    if af in (
        "local",
        "tei",
        "text-embeddings-inference",
        "hf",
        "huggingface",
    ):
        return "openai"
    b = (base_url or "").lower()
    if "dashscope.aliyuncs.com" in b or "compatible-mode/v1" in b:
        return "ali"
    # 未写 api_format=ollama 时，常见 Ollama 监听端口仍走 Ollama 客户端
    if ":11434" in b and af == "openai":
        return "ollama"
    return "openai"


def _openai_compatible_api_base(base: str) -> str:
    b = base.strip().rstrip("/")
    if b.endswith("/v1"):
        return b
    return f"{b}/v1"


def _is_dashscope_compatible_embedding_host(url: str) -> bool:
    u = (url or "").lower()
    return "dashscope.aliyuncs.com" in u


def _llama_index_openai_accepts_model(model_code: str) -> bool:
    """LlamaIndex ``OpenAIEmbedding`` 在 ``__init__`` 内把 ``model`` 强转为枚举；未收录的模型名须走直连 OpenAI 客户端。"""
    key = (model_code or "").strip()
    if not key:
        return False
    try:
        OpenAIEmbeddingModelType(key)
    except ValueError:
        return False
    return True


class _FlexibleOpenAIEmbedding(LlamaBaseEmbedding):
    """OpenAI 兼容 ``/v1/embeddings``：任意 ``model`` 字符串（如 DashScope ``text-embedding-v3``）。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        api_base: str,
        embed_batch_size: int,
        timeout: float,
    ) -> None:
        super().__init__(model_name=model, embed_batch_size=embed_batch_size)
        self._model = model
        self._api_key = api_key or ""
        self._api_base = api_base
        self._timeout = timeout
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def _sync(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            )
        return self._sync_client

    def _async(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            )
        return self._async_client

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ") for t in texts]
        data = self._sync().embeddings.create(input=cleaned, model=self._model).data
        return [cast(list[float], list(d.embedding)) for d in data]

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ") for t in texts]
        data = (await self._async().embeddings.create(input=cleaned, model=self._model)).data
        return [cast(list[float], list(d.embedding)) for d in data]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_sync([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._embed_async([query]))[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_sync([text])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._embed_async([text]))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed_sync(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_async(texts)


class KnowledgeEmbeddingClient:
    """薄封装：对外保持 ``async embed_texts``。"""

    def __init__(self, inner: LlamaBaseEmbedding) -> None:
        self._inner = inner

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = await self._inner.aget_text_embedding_batch(texts)
        return [[float(x) for x in row] for row in raw]


def build_llama_index_embedding(
    model: EmbeddingModelBinding,
    provider: EmbeddingProviderBinding,
) -> KnowledgeEmbeddingClient:
    """根据 DTO 构造官方 LlamaIndex 嵌入 + 客户端。"""
    base = model.endpoint or provider.base_url
    api_key = provider.api_key
    model_code = model.model_code
    api_format = provider.api_format

    embed_type = _infer_embed_type(api_format, base)
    embed_batch_default = 32

    if embed_type == "ali":
        if not api_key:
            raise EmbeddingAdapterError("阿里云 DashScope 嵌入需要提供商 api_key")
        if not model_code:
            raise EmbeddingAdapterError("嵌入模型 model_code 为空")
        ali_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        embed_batch = DASHSCOPE_COMPAT_EMBEDDING_BATCH_MAX
        if _llama_index_openai_accepts_model(model_code):
            inner = LIOpenAIEmbedding(
                model=model_code,
                api_key=api_key,
                api_base=ali_base,
                embed_batch_size=embed_batch,
                timeout=120.0,
            )
        else:
            inner = _FlexibleOpenAIEmbedding(
                model=model_code,
                api_key=api_key,
                api_base=ali_base,
                embed_batch_size=embed_batch,
                timeout=120.0,
            )
        return KnowledgeEmbeddingClient(inner)

    if embed_type == "ollama":
        if not base:
            raise EmbeddingAdapterError("Ollama 嵌入未配置 endpoint 或提供商 base_url")
        if not model_code:
            raise EmbeddingAdapterError("嵌入模型 model_code 为空")
        inner = LIOllamaEmbedding(
            model_name=model_code,
            base_url=base.rstrip("/"),
            embed_batch_size=embed_batch_default,
        )
        return KnowledgeEmbeddingClient(inner)

    if not base:
        raise EmbeddingAdapterError("嵌入模型未配置 endpoint 或提供商 base_url")
    if not model_code:
        raise EmbeddingAdapterError("嵌入模型 model_code 为空")

    api_base = _openai_compatible_api_base(base)
    embed_batch = (
        min(embed_batch_default, DASHSCOPE_COMPAT_EMBEDDING_BATCH_MAX)
        if _is_dashscope_compatible_embedding_host(api_base)
        else embed_batch_default
    )
    if _llama_index_openai_accepts_model(model_code):
        inner = LIOpenAIEmbedding(
            model=model_code,
            api_key=api_key,
            api_base=api_base,
            embed_batch_size=embed_batch,
            timeout=120.0,
        )
    else:
        inner = _FlexibleOpenAIEmbedding(
            model=model_code,
            api_key=api_key,
            api_base=api_base,
            embed_batch_size=embed_batch,
            timeout=120.0,
        )
    return KnowledgeEmbeddingClient(inner)
