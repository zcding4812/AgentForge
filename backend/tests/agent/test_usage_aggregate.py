"""``aggregate_token_usage_from_messages`` 对多步末态消息的累加。"""

from langchain_core.messages import AIMessage

from app.agent.adapters.telemetry.usage import (
    aggregate_token_usage_from_messages,
    extract_token_usage_from_message,
)


def test_extract_prefers_usage_metadata_langchain_openai_shape() -> None:
    m = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
    )
    pt, ct, tt = extract_token_usage_from_message(m)
    assert pt == 100
    assert ct == 40
    assert tt == 140


def test_aggregate_sums_multiple_ai_messages() -> None:
    m1 = AIMessage(
        content="a",
        response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    m2 = AIMessage(
        content="b",
        response_metadata={"usage": {"input_tokens": 20, "output_tokens": 8}},
    )
    pt, ct, tt = aggregate_token_usage_from_messages([m1, m2])
    assert pt == 30
    assert ct == 13
    assert tt == 43


def test_aggregate_empty_returns_none() -> None:
    assert aggregate_token_usage_from_messages([]) == (None, None, None)


def test_aggregate_total_only_messages() -> None:
    m = AIMessage(
        content="x",
        response_metadata={"token_usage": {"total_tokens": 42}},
    )
    _pt, _ct, tt = aggregate_token_usage_from_messages([m])
    assert tt == 42


def test_extract_additional_kwargs_usage() -> None:
    m = AIMessage(
        content="x", additional_kwargs={"usage": {"prompt_tokens": 3, "completion_tokens": 7}}
    )
    pt, ct, tt = extract_token_usage_from_message(m)
    assert pt == 3
    assert ct == 7
    assert tt == 10
