"""从图末态 ``messages`` 提取本轮工具调用历史，供入库 ``metadata.tool_history``。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.kernel.workbench_orchestration import (
    WORKBENCH_INVOKE_SUB_AGENT_TOOL,
    WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
)
from app.core.constants.conversation import TOOL_HISTORY_METADATA_MAX_RESULT_CHARS

WORKBENCH_SUB_AGENT_TOOLS: frozenset[str] = frozenset(
    {
        WORKBENCH_INVOKE_SUB_AGENT_TOOL,
        WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
    }
)


def _parse_wb_tool_envelope(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict) and "code" in content:
        return content
    if not isinstance(content, str):
        return None
    s = content.strip()
    if not s:
        return None
    try:
        p = json.loads(s)
    except json.JSONDecodeError:
        return None
    return p if isinstance(p, dict) else None


def _iter_nested_trace_rows_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    """从子 Agent 工具返回的 ``data`` 段展开嵌套观测行（插入在父级 ToolMessage 结果之前）。"""
    rows: list[dict[str, Any]] = []
    sid_raw = data.get("sub_agent_id")
    try:
        sid = int(sid_raw) if sid_raw is not None else None
    except (TypeError, ValueError):
        sid = None

    sub_th = data.get("sub_tool_history")
    if isinstance(sub_th, list):
        for item in sub_th:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["sub_agent_id"] = sid
            row["nesting"] = "sub_agent"
            nm = row.get("name")
            if sid is not None and isinstance(nm, str) and not nm.startswith("[子Agent"):
                row["name"] = f"[子Agent #{sid}] {nm}"
            rows.append(row)

    sub_pt = data.get("sub_process_trace")
    if isinstance(sub_pt, list):
        for item in sub_pt:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", ""))
            ph = str(item.get("phase", ""))
            label = (
                f"[子Agent #{sid}] 模型输出 ({ph})"
                if sid is not None
                else f"[子Agent] 模型输出 ({ph})"
            )
            rows.append(
                {
                    "phase": "result",
                    "name": label,
                    "content": text,
                    "tool_call_id": None,
                    "sub_agent_id": sid,
                    "nesting": "sub_agent",
                }
            )
    return rows


def _nested_rows_from_parent_tool_result(result_row: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(result_row.get("name") or "")
    if name not in WORKBENCH_SUB_AGENT_TOOLS:
        return []
    payload = _parse_wb_tool_envelope(result_row.get("content"))
    if not isinstance(payload, dict) or str(payload.get("code")) != "OK":
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []

    if data.get("parallel") is True and isinstance(data.get("items"), list):
        acc: list[dict[str, Any]] = []
        for it in data["items"]:
            if not isinstance(it, dict) or str(it.get("code")) != "OK":
                continue
            inner = it.get("data")
            if not isinstance(inner, dict):
                continue
            acc.extend(_iter_nested_trace_rows_from_data(inner))
        return acc
    return _iter_nested_trace_rows_from_data(data)


def expand_nested_workbench_traces(
    entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """在父级 ``workbench_invoke_sub_*`` 工具结果前插入子 Agent 的 ``sub_tool_history`` / ``sub_process_trace``。"""
    if not entries:
        return None
    out: list[dict[str, Any]] = []
    for e in entries:
        if e.get("phase") == "result" and str(e.get("name") or "") in WORKBENCH_SUB_AGENT_TOOLS:
            out.extend(_nested_rows_from_parent_tool_result(e))
        out.append(dict(e))
    for i, row in enumerate(out):
        row["seq"] = i
    return out


def _slice_after_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    last_human = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human = i
    if last_human < 0:
        return list(messages)
    return list(messages[last_human + 1 :])


def _normalize_tool_arguments(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            p = json.loads(s)
            return p if isinstance(p, (dict, list)) else {"value": p}
        except json.JSONDecodeError:
            return {"raw": s}
    if isinstance(raw, list):
        return raw
    return {"value": raw}


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def extract_current_turn_tool_history(messages: list[BaseMessage]) -> list[dict[str, Any]] | None:
    """按时间序产出 ``call`` / ``result`` 记录，与 LangGraph 末态消息对齐。"""
    chunk = _slice_after_last_human(messages)
    out: list[dict[str, Any]] = []
    seq = 0
    max_chars = TOOL_HISTORY_METADATA_MAX_RESULT_CHARS
    trunc_suffix = "\n…(truncated)"

    for m in chunk:
        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None) or []
            if not tcs:
                continue
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tid = tc.get("id")
                if tid is None:
                    tid = tc.get("tool_call_id")
                name = tc.get("name")
                args = _normalize_tool_arguments(tc.get("args", tc.get("arguments")))
                out.append(
                    {
                        "seq": seq,
                        "phase": "call",
                        "tool_call_id": tid,
                        "name": name,
                        "arguments": args,
                    }
                )
                seq += 1
        elif isinstance(m, ToolMessage):
            body = _stringify_tool_content(m.content)
            if len(body) > max_chars:
                body = body[: max_chars - len(trunc_suffix)] + trunc_suffix
            out.append(
                {
                    "seq": seq,
                    "phase": "result",
                    "tool_call_id": getattr(m, "tool_call_id", None),
                    "name": getattr(m, "name", None),
                    "content": body,
                }
            )
            seq += 1

    return out if out else None
