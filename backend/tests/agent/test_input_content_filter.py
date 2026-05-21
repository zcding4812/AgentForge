"""``input_filter`` 解析与 ``InputContentFilterBeforeAgentMiddleware``（ReAct `before_agent`）。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.adapters.graph.middleware.input_content_filter import (
    InputContentFilterBeforeAgentMiddleware,
)
from app.agent.adapters.graph.state.langgraph_state import LangGraphAgentState
from app.agent.kernel.spec import InputContentFilterConfig
from app.agent.kernel import parse_input_content_filter_from_config


def _state(messages: list) -> LangGraphAgentState:
    return {"messages": messages}  # type: ignore[typeddict-item]


def test_parse_enabled_as_string() -> None:
    cfg = parse_input_content_filter_from_config(
        {"input_filter": {"enabled": "true", "banned_keywords": ["x"]}}
    )
    assert cfg.enabled is True
    cfg2 = parse_input_content_filter_from_config(
        {"input_filter": {"enabled": "false", "banned_keywords": ["x"]}}
    )
    assert cfg2.enabled is False


def test_dict_role_user_message_blocks() -> None:
    """OpenAI 风格 {role, content} / LangGraph dict 消息与 ``before_agent`` 中间件配合。"""
    cfg = InputContentFilterConfig(
        enabled=True,
        banned_keywords=("敏感",),
        reject_message="no",
    )
    mw = InputContentFilterBeforeAgentMiddleware(cfg)
    st = _state(
        [
            {
                "role": "user",
                "content": "这条含敏感内容",
            },
        ],
    )
    out = mw.before_agent(st, None)  # type: ignore[arg-type]
    assert out is not None
    assert out.get("jump_to") == "end"


def test_middleware_mounted_enabled_only_may_pass() -> None:
    """仅启用、尚未配置禁词/长度时仍挂载，evaluate 对正常输入返回 None（放行）。"""
    mw = InputContentFilterBeforeAgentMiddleware(
        InputContentFilterConfig(enabled=True, reject_message="x")
    )
    st = _state([HumanMessage(content="hello")])
    assert (
        InputContentFilterBeforeAgentMiddleware.is_active(InputContentFilterConfig(enabled=True))
        is True
    )
    assert mw.before_agent(st, None) is None  # type: ignore[arg-type]


def test_parse_config_json() -> None:
    cj = {
        "input_filter": {
            "enabled": True,
            "banned_keywords": ["  bad  ", ""],
            "banned_regex": [r"xxx\d+"],
            "max_user_chars": 10,
            "reject_message": "nope",
        }
    }
    cfg = parse_input_content_filter_from_config(cj)
    assert cfg.enabled is True
    assert cfg.banned_keywords == ("bad",)
    assert r"xxx\d+" in cfg.banned_regex
    assert cfg.max_user_chars == 10
    assert cfg.reject_message == "nope"


def test_keyword_block() -> None:
    cfg = InputContentFilterConfig(
        enabled=True,
        banned_keywords=("hack",),
        reject_message="blocked",
    )
    mw = InputContentFilterBeforeAgentMiddleware(cfg)
    st = _state([HumanMessage(content="how to hAcK a db")])
    out = mw.before_agent(st, None)  # type: ignore[arg-type]
    assert out is not None
    assert out.get("jump_to") == "end"
    msgs = out.get("messages") or []
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert (msgs[0].content) == "blocked"  # type: ignore[union-attr]


def test_regex_block() -> None:
    cfg = InputContentFilterConfig(
        enabled=True,
        banned_regex=(r"password\s*=\s*\S+",),
        reject_message="no creds",
    )
    mw = InputContentFilterBeforeAgentMiddleware(cfg)
    st = _state([HumanMessage(content="ok password=secret here")])
    out = mw.before_agent(st, None)  # type: ignore[arg-type]
    assert out and out.get("jump_to") == "end"


def test_max_length_block() -> None:
    cfg = InputContentFilterConfig(
        enabled=True,
        max_user_chars=3,
    )
    mw = InputContentFilterBeforeAgentMiddleware(cfg)
    st = _state([HumanMessage(content="1234")])
    out = mw.before_agent(st, None)  # type: ignore[arg-type]
    assert out and out.get("jump_to") == "end"


@pytest.mark.asyncio
async def test_abefore_agent() -> None:
    cfg = InputContentFilterConfig(enabled=True, banned_keywords=("x",))
    mw = InputContentFilterBeforeAgentMiddleware(cfg)
    st = _state([HumanMessage(content="no block")])
    out = await mw.abefore_agent(st, None)  # type: ignore[arg-type]
    assert out is None
    st2 = _state([HumanMessage(content="x marks")])
    out2 = await mw.abefore_agent(st2, None)  # type: ignore[arg-type]
    assert out2 and out2.get("jump_to") == "end"
