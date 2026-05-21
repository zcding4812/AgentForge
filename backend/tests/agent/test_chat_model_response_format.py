"""DefaultChatModelFactory 对 response_format 的 bind（json_object / json_schema）。"""

from langchain_core.runnables import RunnableBinding
from langchain_openai.chat_models.base import BaseChatOpenAI

from app.agent.adapters.models.chat_model_registry import DefaultChatModelFactory
from app.agent.kernel import (
    InferenceHyperparameters,
    ModelConfigSnapshot,
    ModelIdentity,
    ResponseConstraints,
)


def _snap(*, response: ResponseConstraints) -> ModelConfigSnapshot:
    return ModelConfigSnapshot(
        config_id=None,
        identity=ModelIdentity(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test-key",  # 仅单元测试，避免未设置 OPENAI_API_KEY 时初始化 OpenAI 客户端失败
        ),
        hyperparameters=InferenceHyperparameters(),
        response=response,
        tool_choice=None,
    )


def test_stream_usage_enabled_with_custom_base_url() -> None:
    """自定义网关须显式 stream_usage，否则 LangGraph 流式末态消息常无 token 用量。"""
    fac = DefaultChatModelFactory()
    snap = ModelConfigSnapshot(
        config_id=None,
        identity=ModelIdentity(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test-key",
            base_url_override="http://127.0.0.1:9/v1",
        ),
        hyperparameters=InferenceHyperparameters(),
        response=ResponseConstraints(),
        tool_choice=None,
    )
    llm = fac.build(snap)
    assert isinstance(llm, BaseChatOpenAI)
    assert llm.stream_usage is True


def test_json_object_binds() -> None:
    fac = DefaultChatModelFactory()
    llm = fac.build(
        _snap(
            response=ResponseConstraints(
                response_format="json_object",
            ),
        ),
    )
    assert isinstance(llm, RunnableBinding)
    assert isinstance(llm.bound, BaseChatOpenAI)
    assert llm.kwargs.get("response_format") == {"type": "json_object"}


def test_json_schema_omitted_schema_uses_default() -> None:
    fac = DefaultChatModelFactory()
    llm = fac.build(
        _snap(
            response=ResponseConstraints(
                response_format="json_schema",
            ),
        ),
    )
    assert isinstance(llm, RunnableBinding)
    rf = llm.kwargs.get("response_format")
    assert isinstance(rf, dict) and rf.get("type") == "json_schema"
    inner = rf.get("json_schema")
    assert isinstance(inner, dict)
    assert inner.get("strict") is False
    assert inner.get("schema") == {"type": "object", "additionalProperties": True}


def test_json_schema_binds_with_schema() -> None:
    fac = DefaultChatModelFactory()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    llm = fac.build(
        _snap(
            response=ResponseConstraints(
                response_format="json_schema",
                json_schema_id="my_out",
                response_json_schema=schema,
            ),
        ),
    )
    assert isinstance(llm, RunnableBinding)
    assert isinstance(llm.bound, BaseChatOpenAI)
    rf = llm.kwargs.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") == "json_schema"
    inner = rf.get("json_schema")
    assert isinstance(inner, dict)
    assert inner.get("name") == "my_out"
    assert inner.get("strict") is True
    assert inner.get("schema") == schema
