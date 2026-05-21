"""工作台编排边：从末态对话消息列表解析对子 Agent 的调用顺序（MVP，不依赖 Trace Span）。

仅依赖消息对象上的 **``tool_calls``** 约定（与 LangGraph / LangChain 末态结构对齐），**不** import LangChain/LangGraph，符合 ``kernel`` 分层约束。

实现以 :class:`WorkbenchOrchestrationExtractor` 为主；:func:`extract_workbench_orchestration_edges` 为便捷 Facade。

模块级 ``WORKBENCH_INVOKE_SUB_AGENT_TOOL`` / ``WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL`` 为内置工具名字面量的 **唯一事实来源**；其它 Python 模块须 import 二者，勿再复制同名字符串。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

WORKBENCH_INVOKE_SUB_AGENT_TOOL = "workbench_invoke_sub_agent"
WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL = "workbench_invoke_sub_agents_parallel"


@dataclass(frozen=True, slots=True)
class OrchestrationEdge:
    """编排边：父为当前工作台入库 ``agent_id``，子为目标 ``agent_id``（含单次调用与并行批量内的每个子任务）。"""

    order: int
    parent_agent_id: int
    child_agent_id: int


class WorkbenchOrchestrationExtractor:
    """从末态 ``messages`` 收集 ``OrchestrationEdge``（顺序与并行工具）。"""

    __slots__ = ("_parent_agent_id",)

    def __init__(self, *, parent_agent_id: int) -> None:
        self._parent_agent_id = parent_agent_id

    @staticmethod
    def _tool_arguments_as_dict(args: Any) -> dict[str, Any] | None:
        """将 ``tool_calls[].args`` 规范为 ``dict``（支持 JSON 字符串或已是 dict）。"""
        if args is None:
            return None
        if isinstance(args, str):
            s = args.strip()
            if not s:
                return None
            try:
                parsed: Any = json.loads(s)
            except json.JSONDecodeError:
                return None
        elif isinstance(args, dict):
            parsed = args
        else:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _tool_call_name_and_args(tc: Any) -> tuple[str | None, Any]:
        if isinstance(tc, dict):
            return tc.get("name"), tc.get("args")
        return getattr(tc, "name", None), getattr(tc, "args", None)

    @staticmethod
    def _parse_agent_id_from_tool_args(args: Any) -> int | None:
        d = WorkbenchOrchestrationExtractor._tool_arguments_as_dict(args)
        if d is None:
            return None
        raw = d.get("agent_id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_tasks_json_parallel_agent_ids(args: Any) -> list[int]:
        """从 ``workbench_invoke_sub_agents_parallel`` 的 ``tasks_json`` 解析子 ``agent_id`` 列表（保持数组顺序）。"""
        d = WorkbenchOrchestrationExtractor._tool_arguments_as_dict(args)
        if d is None:
            return []
        raw_tasks = d.get("tasks_json")
        if raw_tasks is None:
            return []
        if isinstance(raw_tasks, str):
            try:
                arr: Any = json.loads(raw_tasks)
            except json.JSONDecodeError:
                return []
        elif isinstance(raw_tasks, list):
            arr = raw_tasks
        else:
            return []
        if not isinstance(arr, list):
            return []
        out: list[int] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            cid = item.get("agent_id")
            if cid is None:
                continue
            try:
                out.append(int(cid))
            except (TypeError, ValueError):
                continue
        return out

    def extract_from_messages(self, messages: list[Any]) -> tuple[OrchestrationEdge, ...]:
        order = 0
        out: list[OrchestrationEdge] = []
        parent = self._parent_agent_id
        for m in messages:
            tool_calls = getattr(m, "tool_calls", None) or []
            if not tool_calls:
                continue
            for tc in tool_calls:
                name, args = self._tool_call_name_and_args(tc)
                if name == WORKBENCH_INVOKE_SUB_AGENT_TOOL:
                    child_id = self._parse_agent_id_from_tool_args(args)
                    if child_id is None:
                        continue
                    out.append(
                        OrchestrationEdge(
                            order=order,
                            parent_agent_id=parent,
                            child_agent_id=child_id,
                        )
                    )
                    order += 1
                elif name == WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL:
                    for child_id in self._parse_tasks_json_parallel_agent_ids(args):
                        out.append(
                            OrchestrationEdge(
                                order=order,
                                parent_agent_id=parent,
                                child_agent_id=child_id,
                            )
                        )
                        order += 1
        return tuple(out)


def extract_workbench_orchestration_edges(
    messages: list[Any],
    *,
    parent_agent_id: int,
) -> tuple[OrchestrationEdge, ...]:
    """遍历末态消息中带 ``tool_calls`` 的条目，收集对子 Agent 的编排边（顺序调用与并行批量）。"""
    return WorkbenchOrchestrationExtractor(parent_agent_id=parent_agent_id).extract_from_messages(
        messages
    )
