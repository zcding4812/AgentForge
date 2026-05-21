"""内核 ``spec.ChatModelLike``、工厂 ``_validate_chat_model`` 与遥测解包。"""

from langchain_core.runnables import RunnableBinding
from langchain_openai.chat_models.base import BaseChatOpenAI

from app.agent.adapters.graph.factory.langgraph_factory import LangGraphAgentFactory
from app.agent.adapters.telemetry.usage import chat_model_label


def test_validate_chat_model_accepts_base() -> None:
    m = BaseChatOpenAI(model="x", api_key="k")
    LangGraphAgentFactory._validate_chat_model(m)


def test_validate_chat_model_accepts_runnable_binding() -> None:
    m = BaseChatOpenAI(model="x", api_key="k")
    b = m.bind(response_format={"type": "json_object"})
    assert isinstance(b, RunnableBinding)
    LangGraphAgentFactory._validate_chat_model(b)


def test_chat_model_label_unwraps_binding() -> None:
    b = BaseChatOpenAI(model="mymodel", api_key="k").bind(response_format={"type": "json_object"})
    assert chat_model_label(b) == "mymodel"
