"""与 :mod:`input_content_filter` 中间件配套的规则求值（依赖 LangChain 消息类型；可单测）。"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

from app.agent.kernel.spec import InputContentFilterConfig

logger = logging.getLogger(__name__)


def compile_banned_regexes(config: InputContentFilterConfig) -> list[re.Pattern[str]]:
    out: list[re.Pattern[str]] = []
    for p in config.banned_regex:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            logger.warning("input_filter.banned_regex 无效，已忽略 | pat=%r err=%s", p, e)
    return out


def _message_text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for b in c:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
        return "".join(parts)
    return str(c) if c is not None else ""


def _text_from_dict_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "".join(parts)


def _text_from_message_like(m: object) -> str:
    if isinstance(m, BaseMessage):
        return _message_text(m)
    if isinstance(m, dict) and "content" in m:
        return _text_from_dict_content(m.get("content"))
    return ""


def _is_user_or_human(m: object) -> bool:
    if isinstance(m, HumanMessage):
        return True
    if isinstance(m, BaseMessage) and getattr(m, "type", None) == "human":
        return True
    if isinstance(m, dict):
        t = m.get("type")
        if t is not None and str(t).lower() in ("human", "user"):
            return True
        r = m.get("role")
        if r is not None and str(r).strip().lower() in ("user", "human"):
            return True
    return False


def _clone_human_content(original: HumanMessage, new_str: str) -> str | list:
    c = original.content
    if isinstance(c, str):
        return new_str
    if isinstance(c, list):
        for i, b in enumerate(c):
            if isinstance(b, str):
                n = list(c)
                n[i] = new_str
                return n
            if isinstance(b, dict) and b.get("type") == "text":
                d = {**b, "text": new_str}
                n = list(c)
                n[i] = d
                return n
        return new_str
    return new_str


def input_content_filter_evaluate(
    state: Mapping[str, Any],
    config: InputContentFilterConfig,
    compiled_regex: list[re.Pattern[str]],
) -> dict[str, Any] | None:
    if not config.enabled:
        return None
    items = list(state.get("messages") or [])
    last_human: object | None = None
    for m in reversed(items):
        if _is_user_or_human(m):
            last_human = m
            break
    if last_human is None:
        return None

    text = _text_from_message_like(last_human)
    if not (text and text.strip()):
        return None

    lower = text.lower()
    for kw in config.banned_keywords:
        if not kw:
            continue
        if kw.lower() in lower:
            return {
                "messages": [AIMessage(content=config.reject_message)],
                "jump_to": "end",
            }
    for pat in compiled_regex:
        if pat.search(text):
            return {
                "messages": [AIMessage(content=config.reject_message)],
                "jump_to": "end",
            }
    mxc = config.max_user_chars
    if mxc is not None and len(text) > mxc:
        if config.truncate_on_max:
            new_text = text[:mxc]
            if isinstance(last_human, dict):
                raw_id = last_human.get("id")
            else:
                raw_id = getattr(last_human, "id", None)
            msg_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None
            if msg_id:
                new_body = (
                    _clone_human_content(last_human, new_text)
                    if isinstance(last_human, HumanMessage)
                    else new_text
                )
                return {
                    "messages": [
                        RemoveMessage(id=msg_id),
                        HumanMessage(content=new_body),
                    ],
                }
        return {
            "messages": [AIMessage(content=config.reject_message)],
            "jump_to": "end",
        }
    return None
