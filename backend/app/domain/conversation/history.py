"""会话历史：消息行协议 → ``HistoryTurnSlot`` 组装、去重、协调编排，及 Redis 缓存 JSON 载荷。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from app.agent.kernel.spec import HistoryTurnSlot, ToolCallSlot, ToolResultSlot

# ---------------------------------------------------------------------------
# 协议与文本
# ---------------------------------------------------------------------------


class ConversationMessageLike(Protocol):
    """``append_message`` 落库行在映射为 ``HistoryTurnSlot`` 时所需字段。"""

    @property
    def role(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def message_metadata(self) -> dict[str, Any] | None: ...


def strip_optional(text: str | None) -> str | None:
    """首尾去空白；空串视为 ``None``。"""
    if text is None:
        return None
    t = text.strip()
    return t if t else None


# ---------------------------------------------------------------------------
# 工具元数据解析
# ---------------------------------------------------------------------------


class AssistantToolMetadataParser:
    """解析落库/HTTP 对齐的 ``tool_calls`` / ``tool_results`` JSON 列表。"""

    def extract_tool_calls(self, meta: dict[str, Any] | None) -> tuple[ToolCallSlot, ...]:
        if not meta:
            return ()
        raw = meta.get("tool_calls")
        if not isinstance(raw, list):
            return ()
        out: list[ToolCallSlot] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not tid or not name:
                continue
            args = item.get("arguments")
            if args is None:
                arg_s = "{}"
            elif isinstance(args, str):
                arg_s = args.strip() or "{}"
            else:
                arg_s = json.dumps(args, ensure_ascii=False)
            out.append(ToolCallSlot(id=tid, name=name, arguments=arg_s))
        return tuple(out)

    def extract_tool_results(self, meta: dict[str, Any] | None) -> tuple[ToolResultSlot, ...]:
        if not meta:
            return ()
        raw = meta.get("tool_results")
        if not isinstance(raw, list):
            return ()
        out: list[ToolResultSlot] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            tcid = str(item.get("tool_call_id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not tcid or not name:
                continue
            content = item.get("content")
            c = "" if content is None else str(content)
            out.append(ToolResultSlot(tool_call_id=tcid, name=name, content=c))
        return tuple(out)


# ---------------------------------------------------------------------------
# 槽位 JSON（缓存等）
# ---------------------------------------------------------------------------

_PAYLOAD_VERSION = 1


def history_turn_slots_to_json_str(turns: list[HistoryTurnSlot]) -> str:
    payload: dict[str, Any] = {
        "v": _PAYLOAD_VERSION,
        "turns": [_turn_to_obj(t) for t in turns],
    }
    return json.dumps(payload, ensure_ascii=False)


def history_turn_slots_from_json_str(raw: str) -> list[HistoryTurnSlot]:
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("v") != _PAYLOAD_VERSION:
        raise ValueError("invalid conversation history cache payload")
    turns_raw = data.get("turns")
    if not isinstance(turns_raw, list):
        raise ValueError("invalid turns list")
    return [_turn_from_obj(x) for x in turns_raw]


def _turn_to_obj(t: HistoryTurnSlot) -> dict[str, Any]:
    return {
        "user": t.user,
        "assistant": t.assistant,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in t.tool_calls
        ],
        "tool_results": [
            {
                "tool_call_id": tr.tool_call_id,
                "name": tr.name,
                "content": tr.content,
            }
            for tr in t.tool_results
        ],
    }


def _turn_from_obj(o: Any) -> HistoryTurnSlot:
    if not isinstance(o, dict):
        raise ValueError("invalid turn object")
    u = str(o.get("user") or "")
    a = str(o.get("assistant") or "")
    tcs = tuple(
        ToolCallSlot(
            id=str(x.get("id") or "").strip(),
            name=str(x.get("name") or "").strip(),
            arguments=str(x.get("arguments") if x.get("arguments") is not None else "{}"),
        )
        for x in (o.get("tool_calls") or [])
        if isinstance(x, dict)
        and str(x.get("id") or "").strip()
        and str(x.get("name") or "").strip()
    )
    trs = tuple(
        ToolResultSlot(
            tool_call_id=str(x.get("tool_call_id") or "").strip(),
            name=str(x.get("name") or "").strip(),
            content=str(x.get("content") if x.get("content") is not None else ""),
        )
        for x in (o.get("tool_results") or [])
        if isinstance(x, dict)
        and str(x.get("tool_call_id") or "").strip()
        and str(x.get("name") or "").strip()
    )
    return HistoryTurnSlot(user=u, assistant=a, tool_calls=tcs, tool_results=trs)


# ---------------------------------------------------------------------------
# 去重与组装
# ---------------------------------------------------------------------------


class HistoryTurnDeduper:
    """去掉与本轮用户输入重复的末轮，避免 Human 重复注入。"""

    def dedupe_last_if_same_as_current(
        self,
        turns: list[HistoryTurnSlot],
        user_message: str,
    ) -> list[HistoryTurnSlot]:
        u = user_message.strip()
        if not turns or not u:
            return turns
        last = turns[-1]
        if last.user.strip() == u:
            return turns[:-1]
        return turns


class HistoryTurnAssembler:
    """将会话行序列组装为可注入模型的历史轮次。"""

    def __init__(self, metadata_parser: AssistantToolMetadataParser | None = None) -> None:
        self._metadata = metadata_parser or AssistantToolMetadataParser()

    def assemble(self, messages: Sequence[ConversationMessageLike]) -> list[HistoryTurnSlot]:
        """
        按 ``created_at`` / ``id`` 升序遍历（调用方保证顺序）。

        - ``user`` → ``assistant`` 配成一轮；连续 ``user`` 时丢弃前一条未闭合的 user。
        - 以 **user 收尾且无 assistant**：不注入该 user。
        - 孤儿 ``assistant`` 跳过。
        """
        pending_user: str | None = None
        turns: list[HistoryTurnSlot] = []

        for msg in messages:
            role = (msg.role or "").strip().lower()
            if role == "user":
                pending_user = msg.content if msg.content is not None else ""
            elif role == "assistant":
                if pending_user is None:
                    continue
                u = strip_optional(pending_user) or ""
                pending_user = None
                if not u:
                    continue
                meta = msg.message_metadata if isinstance(msg.message_metadata, dict) else None
                a = strip_optional(msg.content) or ""
                tcs = self._metadata.extract_tool_calls(meta)
                trs = self._metadata.extract_tool_results(meta)
                if not a and not tcs:
                    continue
                turns.append(
                    HistoryTurnSlot(user=u, assistant=a, tool_calls=tcs, tool_results=trs),
                )
            else:
                continue

        return turns


class ConversationHistoryCoordinator:
    """
    会话历史领域入口：先 ``HistoryTurnAssembler``，再 ``HistoryTurnDeduper``。

    与 ``app.services`` 中「编排仓储 + 调领域」的分工对应；本类仅表达领域内的步骤组合。
    """

    def __init__(
        self,
        assembler: HistoryTurnAssembler | None = None,
        deduper: HistoryTurnDeduper | None = None,
    ) -> None:
        self._assembler = assembler or HistoryTurnAssembler()
        self._deduper = deduper or HistoryTurnDeduper()

    def assemble_turns_only(
        self,
        messages: Sequence[ConversationMessageLike],
    ) -> list[HistoryTurnSlot]:
        """仅 DB → 轮次，不做与当前句的去重（例如列表回放、调试）。"""
        return self._assembler.assemble(messages)

    def prepare_turns_for_model_invoke(
        self,
        messages: Sequence[ConversationMessageLike],
        current_user_message: str,
    ) -> list[HistoryTurnSlot]:
        """供本轮模型调用：组装历史 + 去掉与 ``current_user_message`` 重复的末轮。"""
        turns = self._assembler.assemble(messages)
        return self._deduper.dedupe_last_if_same_as_current(turns, current_user_message)
