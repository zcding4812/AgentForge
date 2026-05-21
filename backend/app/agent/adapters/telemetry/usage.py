"""从 LangChain 消息元数据提取 token 用量（适配器侧，无框架无关保证）。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableBinding


def chat_model_label(model: object) -> str | None:
    """用于遥测日志的模型展示名。支持 ``BaseChatModel`` 及 ``bind`` 后的 ``RunnableBinding``。"""
    m: object = model
    if isinstance(m, RunnableBinding) and getattr(m, "bound", None) is not None:
        m = m.bound
    if not isinstance(m, BaseChatModel):
        return None
    name = getattr(m, "model_name", None)
    return str(name).strip() if name else None


def _parse_usage_mapping(usage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """从 OpenAI/LangChain 常见的 usage 字典解析 ``(prompt/input, completion/output, total)``。"""

    def _int(key: str) -> int | None:
        v = usage.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    pt = _int("prompt_tokens") or _int("input_tokens")
    ct = _int("completion_tokens") or _int("output_tokens")
    tt = _int("total_tokens")
    if tt is None and pt is not None and ct is not None:
        tt = pt + ct
    return pt, ct, tt


def _usage_mapping_from_any(raw: Any) -> dict[str, Any] | None:
    """将 ``usage_metadata`` / ``usage`` 等规整为可解析的 ``dict``。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, Mapping):
        return dict(raw)
    md = getattr(raw, "model_dump", None)
    if callable(md):
        try:
            dumped = md()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dumped
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def extract_token_usage_from_message(msg: BaseMessage) -> tuple[int | None, int | None, int | None]:
    """返回 ``(prompt_tokens, completion_tokens, total_tokens)``；无法解析时为 ``None``。

    LangChain 1.x / ``langchain-openai`` 通常在 ``AIMessage.usage_metadata`` 写入标准用量；
    旧路径为 ``response_metadata["token_usage"]`` / ``["usage"]``，二者均解析；
    部分网关仅把用量放在 ``additional_kwargs["usage"]``。
    """
    um = _usage_mapping_from_any(getattr(msg, "usage_metadata", None))
    if um:
        pt, ct, tt = _parse_usage_mapping(um)
        if pt is not None or ct is not None or tt is not None:
            return pt, ct, tt

    meta_raw = getattr(msg, "response_metadata", None)
    if isinstance(meta_raw, dict):
        meta = meta_raw
    elif isinstance(meta_raw, Mapping):
        meta = dict(meta_raw)
    else:
        meta = {}
    usage = _usage_mapping_from_any(meta.get("token_usage"))
    if usage is None:
        usage = _usage_mapping_from_any(meta.get("usage"))
    if usage:
        return _parse_usage_mapping(usage)

    add_kw = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(add_kw, dict):
        ak_usage = _usage_mapping_from_any(add_kw.get("usage"))
        if ak_usage:
            return _parse_usage_mapping(ak_usage)

    return None, None, None


def aggregate_token_usage_from_messages(
    messages: list[BaseMessage],
) -> tuple[int | None, int | None, int | None]:
    """对末态消息列表中各条可解析的 ``usage`` 做累加（多轮工具/ReAct/Plan 等多步调用）。

    返回 ``(prompt_tokens, completion_tokens, total_tokens)``；第三项为与网关对齐的 **总**
    token（分项之和；若仅有 ``total_tokens`` 而无分项则累加 ``total_tokens``）。全程无用量则为 ``None``。
    """
    pt_acc: int | None = None
    ct_acc: int | None = None
    tt_only_sum = 0
    tt_only_n = 0
    for m in messages:
        pt, ct, tt = extract_token_usage_from_message(m)
        if pt is not None:
            pt_acc = (pt_acc or 0) + pt
        if ct is not None:
            ct_acc = (ct_acc or 0) + ct
        if pt is None and ct is None and tt is not None:
            tt_only_sum += tt
            tt_only_n += 1
    if pt_acc is None and ct_acc is None:
        if tt_only_n:
            return None, None, tt_only_sum
        return None, None, None
    total = (pt_acc or 0) + (ct_acc or 0)
    return pt_acc, ct_acc, total
