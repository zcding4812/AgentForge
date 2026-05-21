"""W3C trace_id / span_id 校验（与落库列宽及 OpenAI tool_call id 等非链路 id 区分）。"""

from app.core.context import RequestTraceContext


def test_is_w3c_trace_id_accepts_normalized_hex() -> None:
    tid = "a" * 32
    assert RequestTraceContext.is_w3c_trace_id(tid) is True
    assert RequestTraceContext.is_w3c_trace_id(tid.upper()) is True


def test_is_w3c_trace_id_rejects_tool_call_style_id() -> None:
    # 与 LangChain/OpenAI tool_call id 常见形态类似：非 32 位纯 hex
    bad = "call_189899a14e0945d58cf6ff_task_1"
    assert RequestTraceContext.is_w3c_trace_id(bad) is False


def test_is_w3c_span_id_length_and_hex() -> None:
    assert RequestTraceContext.is_w3c_span_id("13045899bb3c054b") is True
    assert RequestTraceContext.is_w3c_span_id("13045899bb3c054") is False
    assert RequestTraceContext.is_w3c_span_id("gggggggggggggggg") is False
