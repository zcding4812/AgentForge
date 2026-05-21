"""请求体字符串在 Schema 层 ``field_validator`` 规范化。"""

from app.agent.kernel.spec import AgentKind
from app.schemas.agent import AgentCreateBody, AgentInvokeRequest


def test_invoke_request_strips_ids_and_tool_names() -> None:
    r = AgentInvokeRequest.model_validate(
        {
            "agent_kind": AgentKind.SIMPLE_CHAT,
            "user_message": "  hello  ",
            "conversation_session_id": "  sid1  ",
            "request_id": "  rid  ",
            "agent_id": 1,
            "tool_names": ["  a ", "", "b"],
        }
    )
    assert r.user_message == "hello"
    assert r.conversation_session_id == "sid1"
    assert r.request_id == "rid"
    assert r.tool_names == ["a", "b"]


def test_agent_create_strips_name_and_optional() -> None:
    b = AgentCreateBody.model_validate(
        {
            "name": "  n  ",
            "agent_kind": AgentKind.SIMPLE_CHAT,
            "description": "   ",
            "system_prompt": "  x  ",
        }
    )
    assert b.name == "n"
    assert b.description is None
    assert b.system_prompt == "x"


def test_agent_create_accepts_prompt_system_in_config_json() -> None:
    b = AgentCreateBody.model_validate(
        {
            "name": "a",
            "agent_kind": AgentKind.REACT,
            "config_json": {"v": 1, "prompt_system": "hi"},
        }
    )
    assert b.system_prompt is None
    assert b.config_json is not None
    assert b.config_json["prompt_system"] == "hi"


def test_agent_create_accepts_missing_prompt() -> None:
    b = AgentCreateBody.model_validate(
        {
            "name": "a",
            "agent_kind": AgentKind.SIMPLE_CHAT,
            "config_json": {"v": 1, "temperature": 0.7},
        }
    )
    assert b.system_prompt is None
