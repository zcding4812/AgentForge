"""对话模型工厂：OpenAI 兼容默认实现 + 按 ``provider`` 解析的注册表（与 DB ``api_format`` / 请求体厂商字段对齐）。"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai.chat_models.base import BaseChatOpenAI

from app.agent.kernel.ports import ChatModelFactoryPort
from app.agent.kernel.spec import ModelConfigSnapshot


def _omit_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


# 未在请求中提供 `response_json_schema` 时使用的宽松根 schema；配 `strict=False`，无需用户手抄模板。
_DEFAULT_JSON_RESPONSE_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


class DefaultChatModelFactory(ChatModelFactoryPort[BaseChatModel]):
    """默认 OpenAI 兼容适配：实现 `ChatModelFactoryPort[BaseChatModel]`。

    使用 ``BaseChatOpenAI`` 而非 ``ChatOpenAI``：后者会把 ``max_tokens`` 转成
    ``max_completion_tokens``；多数自建/国产 OpenAI 兼容网关仍只认 ``max_tokens``，
    否则会出现「已设最大令牌数但不生效」。

    快照中的 ``InferenceHyperparameters.top_k`` 不在此下发（见模块注释与 ``InferenceHyperparameters`` 类文档）。
    """

    def build(self, snapshot: ModelConfigSnapshot) -> BaseChatModel:
        identity = snapshot.identity
        hp = snapshot.hyperparameters
        rc = snapshot.response

        init_kwargs: dict[str, Any] = {
            "model": identity.model_name,
            "temperature": hp.temperature,
            "max_tokens": hp.max_tokens,
            "top_p": hp.top_p,
            "frequency_penalty": hp.frequency_penalty,
            "presence_penalty": hp.presence_penalty,
            # LangGraph 流式走模型 stream 时，自定义 base_url 默认会关闭 stream_usage，末态 AIMessage 无用量；
            # 显式开启以便 LangChain 下发 stream_options（网关不支持时可能仍无用量，但不影响非流式 ainvoke）。
            "stream_usage": True,
        }
        if hp.stop is not None:
            init_kwargs["stop"] = list(hp.stop)

        # OpenAI 官方 Chat Completions 与当前 openai SDK 的 create() 不支持 top_k（会报 unexpected keyword）。
        # 兼容网关若需 top_k，应走对应厂商 SDK/适配器，此处不传入 model_kwargs。
        model_kwargs: dict[str, Any] = {}
        if hp.seed is not None:
            model_kwargs["seed"] = hp.seed
        if identity.base_url_override:
            init_kwargs["base_url"] = identity.base_url_override
        if identity.api_key:
            init_kwargs["api_key"] = identity.api_key

        init_kwargs = _omit_none(init_kwargs)
        llm = BaseChatOpenAI(
            **init_kwargs,
            model_kwargs=model_kwargs or {},
        )

        if rc.parallel_tool_calls is not None:
            llm = llm.bind(parallel_tool_calls=rc.parallel_tool_calls)

        if rc.response_format == "json_object":
            llm = llm.bind(response_format={"type": "json_object"})
        elif rc.response_format == "json_schema":
            raw_name = (rc.json_schema_id or "structured_output").strip() or "structured_output"
            name = (
                "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_name)[:64]
                or "structured_output"
            )
            if rc.response_json_schema is not None and not isinstance(
                rc.response_json_schema, dict
            ):
                raise TypeError("response_json_schema 必须为对象（JSON 根）或省略。")
            use_custom = (
                isinstance(rc.response_json_schema, dict) and len(rc.response_json_schema) > 0
            )
            schema: dict[str, Any] = (
                rc.response_json_schema if use_custom else _DEFAULT_JSON_RESPONSE_SCHEMA
            )
            strict = use_custom
            llm = llm.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": strict,
                        "schema": schema,
                    },
                }
            )

        return llm


# 与 ``sys_model_provider.api_format`` / 常见 ``model_identity.provider`` 取值一致；
# 凡走 OpenAI 兼容 HTTP 的通道均映射到 ``DefaultChatModelFactory``。
_OPENAI_COMPAT_PROVIDER_KEYS: frozenset[str] = frozenset(
    {
        "openai",
        "tongyi",
        "doubao",
        "ollama",
        "zhipu",
        "moonshot",
    }
)


class ChatModelFactoryRegistry:
    """``provider`` → 工厂类；实例化后调用 ``build``。"""

    __slots__ = ("_factories",)

    def __init__(self) -> None:
        self._factories: dict[str, type[ChatModelFactoryPort[Any]]] = {}

    def register(self, provider: str, factory_cls: type[ChatModelFactoryPort[Any]]) -> None:
        key = provider.strip()
        if not key:
            raise ValueError("provider 不能为空")
        self._factories[key] = factory_cls

    def get_factory(self, snapshot: ModelConfigSnapshot) -> ChatModelFactoryPort[BaseChatModel]:
        raw = snapshot.identity.provider.strip()
        if not raw:
            raise ValueError("ModelIdentity.provider 不能为空")
        if raw in self._factories:
            return self._factories[raw]()
        if raw in _OPENAI_COMPAT_PROVIDER_KEYS and "openai" in self._factories:
            return self._factories["openai"]()
        raise ValueError(
            f"未注册厂商「{raw}」的 ChatModel 工厂；"
            f"请在 ChatModelFactoryRegistry.register 中补充，或改用已支持的通道（如 openai 兼容）。",
        )


class RegistryDispatchingChatModelFactory(ChatModelFactoryPort[BaseChatModel]):
    """按快照中的 ``provider`` 从注册表解析具体工厂并构建 ``BaseChatModel``。"""

    __slots__ = ("_registry",)

    def __init__(self, registry: ChatModelFactoryRegistry | None = None) -> None:
        self._registry = registry or get_chat_model_registry()

    def build(self, snapshot: ModelConfigSnapshot) -> BaseChatModel:
        return self._registry.get_factory(snapshot).build(snapshot)


class ErnieChatModelFactory(ChatModelFactoryPort[BaseChatModel]):
    """百度文心一言（扩展示例）：需接入 ``langchain_community``、官方 SDK 或自建 HTTP 后再实现 ``build``。

    数据库中文心提供商标记为 ``api_format=baidu``、``provider_code=wenxin``；
    注册时使用 ``registry.register("baidu", ErnieChatModelFactory)``（与 ``ModelIdentity.provider`` 一致）。
    """

    def build(self, snapshot: ModelConfigSnapshot) -> BaseChatModel:
        raise NotImplementedError(
            "ErnieChatModelFactory 尚未实现：请接入文心 ChatModel 后注册到 ChatModelFactoryRegistry。",
        )


def build_default_chat_model_registry() -> ChatModelFactoryRegistry:
    registry = ChatModelFactoryRegistry()
    # OpenAI 兼容端点共用一个实现；多键便于与不同 provider 字段对齐。
    for key in sorted(_OPENAI_COMPAT_PROVIDER_KEYS):
        registry.register(key, DefaultChatModelFactory)
    return registry


_default_registry: ChatModelFactoryRegistry | None = None


def get_chat_model_registry() -> ChatModelFactoryRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_chat_model_registry()
    return _default_registry
