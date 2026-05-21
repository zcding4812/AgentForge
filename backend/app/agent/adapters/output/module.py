from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.adapters.output.tool_history import (
    _normalize_tool_arguments,
    _stringify_tool_content,
    expand_nested_workbench_traces,
    extract_current_turn_tool_history,
)
from app.agent.adapters.telemetry.usage import aggregate_token_usage_from_messages
from app.agent.kernel.spec import (
    FinalAgentOutput,
    OutputControl,
    ProcessTraceStep,
    ResponseConstraints,
)
from app.agent.kernel.workbench_orchestration import extract_workbench_orchestration_edges
from app.core.constants.conversation import TOOL_HISTORY_METADATA_MAX_RESULT_CHARS

_THINKING_BLOCK = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_INNER = re.compile(
    r"<redacted_thinking>([\s\S]*?)</redacted_thinking>|<thinking>([\s\S]*?)</thinking>",
    re.DOTALL | re.IGNORECASE,
)
# 工作台 aggregate 偶发以「任务已完成」式元话语收束，丢掉前文实质；用于正文回退判定。
_WORKBENCH_META_TAIL_HINT = re.compile(
    r"(本答复严格|已完成精准交付|如您后续需要|请随时提出|不增不减|不偏不倚|"
    r"步骤\s*\d+\s*·\s*模型输出\s*（最终）|对齐用户字面诉求)",
    re.IGNORECASE,
)


class AgentOutputModule:
    """将图末态消息转为领域 `FinalAgentOutput`（与 HTTP 传输形状解耦）。"""

    def finalize(
        self,
        messages: list[BaseMessage],
        response: ResponseConstraints,
        control: OutputControl,
        *,
        workbench_parent_agent_id: int | None = None,
        workbench_fan_in: dict | None = None,
        observability_messages: list[BaseMessage] | None = None,
    ) -> FinalAgentOutput:
        """``observability_messages`` 若非 ``None``，则 ``process_trace`` / ``tool_history`` / ``orchestration_edges``
        优先由其推导。空列表表示门面切片未命中，回退为「完整 ``messages`` 中末条 Human 之后」，
        避免子调用误传空段导致正文与 trace 全空；``tokens`` 仍对完整 ``messages`` 累加。
        """
        obs = observability_messages if observability_messages is not None else messages
        if observability_messages is not None and not observability_messages:
            obs = self._messages_after_last_human(messages)
        ai_turn = self._list_current_turn_ai_messages(obs)
        if (
            workbench_fan_in is not None
            and len(ai_turn) >= 2
            and response.response_format == "text"
        ):
            thinking_joined, text = self._workbench_resolve_visible_text(ai_turn, control)
        else:
            raws = [self._normalize_ai_content(m) for m in ai_turn]
            raw_last = raws[-1] if raws else ""
            thinking_joined, text = self.extract_thinking_and_body(raw_last)
        thinking_out: str | None = (thinking_joined or "").strip() or None
        if thinking_out == "":
            thinking_out = None
        structured: dict | None = None
        if response.response_format in ("json_object", "json_schema"):
            structured = self._try_parse_json(text)
            if structured is None and response.response_format == "json_object":
                text = "JSON 解析失败"
        process_trace = self._build_interleaved_process_trace(obs, control, final_model_body=text)
        _pt, _ct, tt = aggregate_token_usage_from_messages(messages)
        orch: tuple[tuple[int, int, int], ...] | None = None
        if workbench_parent_agent_id is not None:
            edges = extract_workbench_orchestration_edges(
                obs,
                parent_agent_id=workbench_parent_agent_id,
            )
            if edges:
                orch = tuple((e.order, e.parent_agent_id, e.child_agent_id) for e in edges)
        th_list = extract_current_turn_tool_history(obs)
        th_expanded = expand_nested_workbench_traces(th_list) if th_list else None
        tool_history: tuple[dict[str, Any], ...] | None = (
            tuple(th_expanded) if th_expanded else None
        )
        return FinalAgentOutput(
            text=text,
            structured=structured,
            tokens=tt,
            thinking_text=thinking_out,
            orchestration_edges=orch,
            workbench_fan_in=workbench_fan_in,
            process_trace=process_trace,
            tool_history=tool_history,
        )

    @staticmethod
    def _messages_after_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
        last_h = -1
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                last_h = i
        if last_h < 0:
            return list(messages)
        return list(messages[last_h + 1 :])

    def _last_assistant_text(self, messages: list[BaseMessage], control: OutputControl) -> str:
        for m in reversed(messages):
            if isinstance(m, ToolMessage) and not control.include_tool_messages_in_raw:
                continue
            if isinstance(m, AIMessage):
                return self._normalize_ai_content(m)
        return ""

    def _list_current_turn_ai_messages(self, messages: list[BaseMessage]) -> list[AIMessage]:
        last_human = -1
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                last_human = i
        if last_human < 0:
            return [m for m in messages if isinstance(m, AIMessage)]
        return [m for m in messages[last_human + 1 :] if isinstance(m, AIMessage)]

    def _workbench_resolve_visible_text(
        self,
        ai_turn: list[AIMessage],
        control: OutputControl,
    ) -> tuple[str, str]:
        """末条为 workbench_aggregate 时，若明显薄于前文或呈元话语收束，则以前文最详实正文为对外可见答复。"""
        bodies: list[tuple[str, str]] = []
        for m in ai_turn:
            raw = self._normalize_ai_content(m)
            th, body = self.extract_thinking_and_body(raw)
            visible = self.strip_thinking_markers(body, control.strip_thinking_blocks).strip()
            bodies.append((th, visible))
        if len(bodies) < 2:
            th0, vis0 = bodies[-1]
            return th0, vis0
        *prev, last = bodies
        L_prev_max = max(len(p[1]) for p in prev)
        last_th, last_vis = last
        prev_best = max((p[1] for p in prev), key=len)
        meta_tail = bool(_WORKBENCH_META_TAIL_HINT.search(last_vis))
        ratio_short = L_prev_max >= 500 and len(last_vis) <= min(
            420,
            max(200, int(0.36 * L_prev_max)),
        )
        if (L_prev_max >= 400 and meta_tail) or ratio_short:
            merged = self._workbench_merge_fallback_primary_with_aggregate(prev_best, last_vis)
            return "", merged
        return last_th, last_vis

    @staticmethod
    def _workbench_merge_fallback_primary_with_aggregate(primary: str, aggregate_tail: str) -> str:
        """以前文最详实质为主；末条 aggregate 里仍有可独立阅读的补充时拼接到后，避免误丢可用句。"""
        base = primary.rstrip()
        tail = aggregate_tail.strip()
        if not tail or tail in base:
            return base
        if len(tail) < 80:
            return base
        has_meta_fluff = bool(_WORKBENCH_META_TAIL_HINT.search(tail))
        if has_meta_fluff and len(tail) < 360:
            return base
        return f"{base}\n\n---\n\n{tail}"

    def _model_row_display_text(
        self,
        message: AIMessage,
        raw: str,
        control: OutputControl,
        *,
        is_final: bool,
        final_body: str,
    ) -> str:
        if is_final and final_body.strip():
            return final_body.strip()
        _thinking, body = self.extract_thinking_and_body(raw or "")
        body = self.strip_thinking_markers(body, control.strip_thinking_blocks)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if body:
            return body
        tcs = getattr(message, "tool_calls", None) or []
        if tcs:
            return "（本段无可见正文，见下列工具步骤。）"
        return "（本段无模型正文，可能仅下发工具调用。参数与返回见下方「工具」。）"

    def _build_interleaved_process_trace(
        self,
        obs: list[BaseMessage],
        control: OutputControl,
        *,
        final_model_body: str,
    ) -> tuple[ProcessTraceStep, ...] | None:
        chunk = self._messages_after_last_human(obs)
        last_ai_index = -1
        for i, m in enumerate(chunk):
            if isinstance(m, AIMessage):
                last_ai_index = i
        if last_ai_index < 0:
            return None

        steps: list[ProcessTraceStep] = []
        seq = 0
        max_chars = TOOL_HISTORY_METADATA_MAX_RESULT_CHARS
        trunc_suffix = "\n…(truncated)"
        arg_cap = 8000

        for i, m in enumerate(chunk):
            if isinstance(m, AIMessage):
                raw = self._normalize_ai_content(m)
                is_last_ai = i == last_ai_index
                visible = self._model_row_display_text(
                    m,
                    raw,
                    control,
                    is_final=is_last_ai,
                    final_body=final_model_body,
                )
                phase = "final" if is_last_ai else "step"
                steps.append(ProcessTraceStep(seq=seq, text=visible, phase=phase, kind="model"))
                seq += 1
                tcs = getattr(m, "tool_calls", None) or []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tid = tc.get("id")
                    if tid is None:
                        tid = tc.get("tool_call_id")
                    name = tc.get("name")
                    args = _normalize_tool_arguments(tc.get("args", tc.get("arguments")))
                    try:
                        arg_str = json.dumps(args, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        arg_str = str(args)
                    if len(arg_str) > arg_cap:
                        arg_str = arg_str[: arg_cap - 3] + "..."
                    nm = str(name) if name is not None else "tool"
                    disp = f"**{nm}**\n\n```json\n{arg_str}\n```"
                    steps.append(
                        ProcessTraceStep(
                            seq=seq,
                            text=disp,
                            phase="step",
                            kind="tool_call",
                            tool_call_id=str(tid) if tid is not None else None,
                            name=str(name) if name is not None else None,
                            arguments=args,
                        )
                    )
                    seq += 1
            elif isinstance(m, ToolMessage):
                full_body = _stringify_tool_content(m.content)
                display_body = full_body
                if len(display_body) > max_chars:
                    display_body = display_body[: max_chars - len(trunc_suffix)] + trunc_suffix
                nm = getattr(m, "name", None) or "tool"
                tid = getattr(m, "tool_call_id", None)
                disp = f"**{nm}** · 返回\n\n```\n{display_body}\n```"
                steps.append(
                    ProcessTraceStep(
                        seq=seq,
                        text=disp,
                        phase="step",
                        kind="tool_result",
                        tool_call_id=str(tid) if tid is not None else None,
                        name=str(nm) if nm else None,
                        content=full_body if len(full_body) <= max_chars else display_body,
                    )
                )
                seq += 1

        return tuple(steps) if steps else None

    def _normalize_ai_content(self, message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            text = "".join(parts)
        else:
            text = str(content)
        return text

    def _try_parse_json(self, text: str) -> dict | None:
        raw = text.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def strip_thinking_markers(text: str, enabled: bool) -> str:
        if not enabled or not text:
            return text
        return _THINKING_BLOCK.sub("", text).strip()

    @staticmethod
    def extract_thinking_and_body(raw: str) -> tuple[str, str]:
        """返回 (拼接后的 thinking 正文, 去掉 thinking 块后的正文)。"""
        parts: list[str] = []
        for m in _THINKING_INNER.finditer(raw or ""):
            inner = (m.group(1) or m.group(2) or "").strip()
            if inner:
                parts.append(inner)
        thinking = "\n\n".join(parts)
        stripped = _THINKING_BLOCK.sub("", raw or "").strip()
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return thinking, stripped
