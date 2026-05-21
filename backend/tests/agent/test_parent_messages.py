import pytest

from app.agent.adapters.tools.workbench.parent_messages import parse_parent_messages_json


def test_parse_parent_messages_empty() -> None:
    assert parse_parent_messages_json(None) == []
    assert parse_parent_messages_json("") == []
    assert parse_parent_messages_json("   ") == []


def test_parse_parent_messages_valid_minimal() -> None:
    raw = '[{"user": "hi", "assistant": "hello"}]'
    out = parse_parent_messages_json(raw)
    assert len(out) == 1
    assert out[0].user == "hi"
    assert out[0].assistant == "hello"


def test_parse_parent_messages_openai_style_pair() -> None:
    raw = '[{"role":"user","content":"父问题"},{"role":"assistant","content":"父助手摘要"}]'
    out = parse_parent_messages_json(raw)
    assert len(out) == 1
    assert out[0].user == "父问题"
    assert out[0].assistant == "父助手摘要"


def test_parse_parent_messages_openai_style_skips_system() -> None:
    raw = (
        '[{"role":"system","content":"忽略"},{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"ok"}]'
    )
    out = parse_parent_messages_json(raw)
    assert len(out) == 1
    assert out[0].user == "hi"
    assert out[0].assistant == "ok"


def test_parse_parent_messages_invalid_json() -> None:
    with pytest.raises(ValueError, match="合法 JSON"):
        parse_parent_messages_json("not json")


def test_parse_parent_messages_not_array() -> None:
    with pytest.raises(ValueError, match="须为 JSON 数组"):
        parse_parent_messages_json('{"user": "x"}')


def test_parse_parent_messages_truncates_to_50() -> None:
    arr = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(60)]
    import json

    raw = json.dumps(arr, ensure_ascii=False)
    out = parse_parent_messages_json(raw)
    assert len(out) == 50
    assert out[0].user == "u10"
    assert out[-1].user == "u59"
