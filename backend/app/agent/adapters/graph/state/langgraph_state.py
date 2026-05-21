from __future__ import annotations

from typing import Annotated, NotRequired

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class WorkbenchFanInItem(TypedDict):
    """单次子调用汇总项（含并行批内拆分后的每一项）。"""

    tool: str
    code: str
    ok: bool
    message: str
    child_agent_id: int | None


class WorkbenchFanInChildStat(TypedDict):
    """按子 Agent 聚合的执行统计。"""

    child_agent_id: int
    total_calls: int
    success_calls: int
    failed_calls: int
    last_code: str | None


class WorkbenchFanInSummary(TypedDict):
    """workbench fan-in 结构化汇总。"""

    total_calls: int
    success_calls: int
    failed_calls: int
    unique_child_agent_ids: list[int]
    items: list[WorkbenchFanInItem]
    child_stats: list[WorkbenchFanInChildStat]


class LangGraphAgentState(TypedDict):
    """全策略共用的 LangGraph 状态：messages 使用官方 reducer；可扩展 plan 等字段。"""

    messages: Annotated[list, add_messages]
    plan: NotRequired[str]
    workbench_fan_in: NotRequired[WorkbenchFanInSummary]
    #: ReAct 图首节点：输入过滤命中 ``jump_to`` 时为 True，条件边直达 ``END``
    react_input_filter_stop: NotRequired[bool]
