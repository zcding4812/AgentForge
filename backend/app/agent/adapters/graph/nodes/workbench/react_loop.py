from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.agent.adapters.graph.node_span import agent_graph_child_span
from app.agent.adapters.graph.state.langgraph_state import (
    LangGraphAgentState,
    WorkbenchFanInChildStat,
    WorkbenchFanInItem,
    WorkbenchFanInSummary,
)
from app.agent.adapters.telemetry.usage import (
    chat_model_label,
    extract_token_usage_from_message,
)
from app.agent.adapters.tools.tool_choice_arg import openai_tool_choice_arg
from app.agent.kernel.default_prompts import WORKBENCH_ORCHESTRATOR_PROMPT
from app.agent.kernel.ports import AgentGraphTelemetry
from app.agent.kernel.spec import ChatModelLike, ToolChoicePolicy
from app.agent.kernel.workbench_orchestration import (
    WORKBENCH_INVOKE_SUB_AGENT_TOOL,
    WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
)
from app.core.constants.agent import WORKBENCH_PARALLEL_MAX_TASKS


def _parse_parallel_tasks_from_args(args: Any) -> list[tuple[int, str, str | None]] | None:
    """解析并规范化并行子任务参数，失败返回 ``None``（走旧工具兼容逻辑）。"""
    if not isinstance(args, dict):
        return None
    raw_tasks = args.get("tasks_json")
    if isinstance(raw_tasks, str):
        try:
            parsed_tasks: Any = json.loads(raw_tasks)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw_tasks, list):
        parsed_tasks = raw_tasks
    else:
        return None
    if not isinstance(parsed_tasks, list):
        return None
    if len(parsed_tasks) < 1 or len(parsed_tasks) > WORKBENCH_PARALLEL_MAX_TASKS:
        return None
    tasks: list[tuple[int, str, str | None]] = []
    for item in parsed_tasks:
        if not isinstance(item, dict):
            return None
        try:
            aid = int(item["agent_id"])
        except (KeyError, TypeError, ValueError):
            return None
        um = item.get("user_message")
        if not isinstance(um, str) or not um.strip():
            return None
        pm = item.get("parent_messages")
        pm_out: str | None = None
        if pm is not None and str(pm).strip():
            if not isinstance(pm, str):
                return None
            pm_out = pm
        tasks.append((aid, um.strip(), pm_out))
    deduped: list[tuple[int, str, str | None]] = []
    seen: set[tuple[int, str]] = set()
    for spec in tasks:
        k = (spec[0], spec[1])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(spec)
    return deduped


def _expand_parallel_tool_calls(reply: AIMessage) -> AIMessage:
    """将 ``workbench_invoke_sub_agents_parallel`` 改写为多个子调用，交由 ToolNode 并发执行。"""
    tool_calls = getattr(reply, "tool_calls", None)
    if not isinstance(tool_calls, list) or not tool_calls:
        return reply
    changed = False
    rewritten: list[Any] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            rewritten.append(tc)
            continue
        if tc.get("name") != WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL:
            rewritten.append(tc)
            continue
        tasks = _parse_parallel_tasks_from_args(tc.get("args"))
        if tasks is None:
            rewritten.append(tc)
            continue
        changed = True
        base_id = str(tc.get("id") or "wb_parallel")
        for idx, (aid, um, pm) in enumerate(tasks):
            args: dict[str, Any] = {"agent_id": aid, "user_message": um}
            if pm is not None:
                args["parent_messages"] = pm
            rewritten.append(
                {
                    "name": WORKBENCH_INVOKE_SUB_AGENT_TOOL,
                    "args": args,
                    "id": f"{base_id}_task_{idx}",
                    "type": "tool_call",
                }
            )
    if not changed:
        return reply
    cloned = reply.model_copy(deep=True)
    cloned.tool_calls = rewritten
    return cloned


def _parse_tool_payload(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    s = content.strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    try:
        literal = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    return literal if isinstance(literal, dict) else None


def _collect_fan_in_summary(messages: list[Any]) -> WorkbenchFanInSummary:
    items: list[WorkbenchFanInItem] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = str(getattr(m, "name", "") or "").strip()
        if name not in {WORKBENCH_INVOKE_SUB_AGENT_TOOL, WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL}:
            continue
        payload = _parse_tool_payload(getattr(m, "content", None))
        if payload is None:
            items.append(
                WorkbenchFanInItem(
                    tool=name,
                    code="PARSE_FAILED",
                    ok=False,
                    message="tool payload parse failed",
                    child_agent_id=None,
                )
            )
            continue
        code = str(payload.get("code", "UNKNOWN"))
        message = str(payload.get("message", ""))
        data = payload.get("data")
        if (
            name == WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL
            and isinstance(data, dict)
            and isinstance(data.get("items"), list)
        ):
            for it in data["items"]:
                if not isinstance(it, dict):
                    continue
                icode = str(it.get("code", "UNKNOWN"))
                raw_aid = it.get("agent_id")
                aid: int | None
                try:
                    aid = int(raw_aid) if raw_aid is not None else None
                except (TypeError, ValueError):
                    aid = None
                items.append(
                    WorkbenchFanInItem(
                        tool=name,
                        code=icode,
                        ok=icode == "OK",
                        message=str(it.get("message", "")),
                        child_agent_id=aid,
                    )
                )
            continue
        child_id: int | None = None
        if isinstance(data, dict):
            raw_child = data.get("sub_agent_id", data.get("agent_id"))
            try:
                child_id = int(raw_child) if raw_child is not None else None
            except (TypeError, ValueError):
                child_id = None
        items.append(
            WorkbenchFanInItem(
                tool=name,
                code=code,
                ok=code == "OK",
                message=message,
                child_agent_id=child_id,
            )
        )
    success = sum(1 for it in items if it["ok"])
    failed = len(items) - success
    unique_agents = sorted(
        {int(it["child_agent_id"]) for it in items if isinstance(it.get("child_agent_id"), int)}
    )
    child_stat_map: dict[int, WorkbenchFanInChildStat] = {}
    for it in items:
        aid = it.get("child_agent_id")
        if not isinstance(aid, int):
            continue
        cur = child_stat_map.get(aid)
        if cur is None:
            cur = WorkbenchFanInChildStat(
                child_agent_id=aid,
                total_calls=0,
                success_calls=0,
                failed_calls=0,
                last_code=None,
            )
            child_stat_map[aid] = cur
        cur["total_calls"] += 1
        if it["ok"]:
            cur["success_calls"] += 1
        else:
            cur["failed_calls"] += 1
        cur["last_code"] = it["code"]
    child_stats = [child_stat_map[k] for k in sorted(child_stat_map)]
    return WorkbenchFanInSummary(
        total_calls=len(items),
        success_calls=success,
        failed_calls=failed,
        unique_child_agent_ids=unique_agents,
        items=items,
        child_stats=child_stats,
    )


def _fan_in_summary_for_prompt(summary: WorkbenchFanInSummary) -> str:
    """将 fan-in 结构化摘要压缩为稳定文本，降低提示词噪声。"""
    base = {
        "total_calls": summary["total_calls"],
        "success_calls": summary["success_calls"],
        "failed_calls": summary["failed_calls"],
        "unique_child_agent_ids": summary["unique_child_agent_ids"],
        "child_stats": summary["child_stats"],
    }
    return json.dumps(base, ensure_ascii=False)


def _as_fan_in_summary(raw: Any) -> WorkbenchFanInSummary | None:
    if not isinstance(raw, dict):
        return None
    required = {
        "total_calls",
        "success_calls",
        "failed_calls",
        "unique_child_agent_ids",
        "items",
        "child_stats",
    }
    if not required.issubset(raw.keys()):
        return None
    return raw  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class WorkbenchPlanPromptConfig:
    """工作台前置规划提示（简短计划，供下一节点执行）。"""

    system_prompt: str = (
        "你是工作台 Planner。请根据当前对话产出一个简短执行计划。\n"
        "要求：\n"
        "0. **目标与同构**：用户在本轮提出的**原始问题/任务**（见对话中对应本轮的用户 Human 正文）是唯一最终目标；"
        "先判断该问题合理的**交付类型与深度**，计划中每一步须服务该目标并与之一致。"
        "勿引入用户未要求的支线或替用户「升级」任务范围；也勿在用户已明确要求深度时把计划写成「只做轻描淡写」。\n"
        "1. 输出 2-5 条要点，每条一行。\n"
        "2. 聚焦子 Agent 调度顺序与并行机会；**不同领域/不同工具或知识依赖**的步骤应绑定**不同** `agent_id`（或先 `workbench_create_agent` 再委派），禁止把多条独立交付线都写进「反复派同一 agent」。\n"
        "3. 若上下文已有 Agent 一览或前序消息已列出资源，计划中**勿**再安排无新意的重复列举；优先写清「委派谁、完成什么」。\n"
        "4. 计划须**可收敛**：应在少数步骤内到达「可汇总答复或可向用户说明阻塞」的状态，避免无限探索式步骤。\n"
        "5. 仅输出计划正文，不要额外解释。"
    )


@dataclass(frozen=True, slots=True)
class WorkbenchReplanPromptConfig:
    """工具/子 Agent 返回后修订计划（覆盖 ``state.plan``）。"""

    system_prompt: str = (
        "你是工作台 Planner。当前对话已包含编排工具与子 Agent 的**最新返回**（含 ToolMessage）。\n"
        "请根据这些事实**修订后续执行计划**（2-6 条要点，每条一行）：\n"
        "0. **不得偏离用户原始问题**：修订只调整**如何达成**用户已提出的目标，不得更换目标、"
        "不得把子 Agent 顺带发挥或未询问的主题升格为主线；若中间结果诱人超范围扩写，计划中须拉回与用户问题**同构**的交付范围。"
        "若发现先前计划与用户合理期望深度不一致（过细或过粗），在此纠正后续步骤。\n"
        "1. 已完成、已否定或不再适用的步骤请删除或改写，不必拘泥于上一轮计划。\n"
        "2. 聚焦**接下来**仍须调用的子 Agent、顺序与可并行机会；若上一轮计划把多域工作都压在**同一** `agent_id` 上，须在此**拆分为多子 Agent 或多步改派**。\n"
        "3. 若需补充列举资源或改派，在计划中写明意图。\n"
        "4. **反重复**：若最近 ToolMessage 显示**同类调用失败两次**或与上轮**无新信息**，必须删除「再来一次相同调用」类步骤，改为换子 Agent、改参、或写清「停止工具循环、由根 Agent 向用户说明原因与缺口」。\n"
        "5. **下一步须递进**：修订后紧随其后的关键一步须与前序失败路径**实质不同**，禁止仅复述将重试同一工具同名同参。\n"
        "6. 仅输出计划正文，不要额外解释。"
    )


async def _run_workbench_planner_llm(
    chat_model: ChatModelLike,
    telemetry: AgentGraphTelemetry,
    *,
    system_prompt: str,
    messages: list[BaseMessage],
    span_name: str,
    trace_phase: str,
) -> str:
    async with agent_graph_child_span(
        span_name,
        trace_id=telemetry.trace_id,
        trace_parent_span_id=telemetry.trace_parent_span_id,
    ):
        plan_prompt: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            *messages,
        ]
        t0 = time.perf_counter()
        plan_msg = await chat_model.ainvoke(plan_prompt)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        pt, ct, tt = extract_token_usage_from_message(plan_msg)
        telemetry.tracer.record_llm_call(
            phase=trace_phase,
            duration_ms=duration_ms,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            model_name=chat_model_label(chat_model),
            request_id=telemetry.request_id,
            trace_id=telemetry.trace_id,
            trace_parent_span_id=telemetry.trace_parent_span_id,
        )
        return plan_msg.content if isinstance(plan_msg.content, str) else str(plan_msg.content)


def _plan_as_trace_message(heading: str, plan_text: str) -> AIMessage | None:
    """将计划写入 ``messages``，供入库 ``process_trace`` / 前端「执行过程」展示（与 ``state.plan`` 同源）。"""
    body = str(plan_text or "").strip()
    if not body:
        return None
    return AIMessage(content=f"【{heading}】\n{body}")


class WorkbenchPlanNode:
    """工作台图节点：首轮生成短计划，存入 ``state.plan``。"""

    __slots__ = ("_chat_model", "_telemetry", "_prompt_config")

    def __init__(
        self,
        chat_model: ChatModelLike,
        *,
        telemetry: AgentGraphTelemetry | None = None,
        prompt_config: WorkbenchPlanPromptConfig | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._telemetry = telemetry or AgentGraphTelemetry()
        self._prompt_config = prompt_config or WorkbenchPlanPromptConfig()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            raise ValueError("工作台规划节点执行前 messages 不能为空")
        plan_text = await _run_workbench_planner_llm(
            self._chat_model,
            self._telemetry,
            system_prompt=self._prompt_config.system_prompt,
            messages=messages,
            span_name="agent.node.workbench.plan",
            trace_phase="workbench.plan",
        )
        plan_stripped = str(plan_text or "").strip()
        out: dict[str, Any] = {"plan": plan_stripped}
        trace_ai = _plan_as_trace_message("编排计划", plan_stripped)
        if trace_ai is not None:
            out["messages"] = [trace_ai]
        return out


class WorkbenchReplanNode:
    """工具执行后：根据最新 ToolMessage 等修订 ``state.plan``，再进入下一轮 ReAct。"""

    __slots__ = ("_chat_model", "_telemetry", "_prompt_config")

    def __init__(
        self,
        chat_model: ChatModelLike,
        *,
        telemetry: AgentGraphTelemetry | None = None,
        prompt_config: WorkbenchReplanPromptConfig | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._telemetry = telemetry or AgentGraphTelemetry()
        self._prompt_config = prompt_config or WorkbenchReplanPromptConfig()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            raise ValueError("工作台修订计划节点执行前 messages 不能为空")
        plan_text = await _run_workbench_planner_llm(
            self._chat_model,
            self._telemetry,
            system_prompt=self._prompt_config.system_prompt,
            messages=messages,
            span_name="agent.node.workbench.replan",
            trace_phase="workbench.replan",
        )
        plan_stripped = str(plan_text or "").strip()
        out: dict[str, Any] = {"plan": plan_stripped}
        trace_ai = _plan_as_trace_message("修订计划", plan_stripped)
        if trace_ai is not None:
            out["messages"] = [trace_ai]
        return out


class WorkbenchToolAwareAgentNode:
    """工作台图节点：基于计划执行 ReAct 循环（可调工具）。"""

    __slots__ = ("_chat_model", "_bound_model", "_telemetry")

    def __init__(
        self,
        chat_model: ChatModelLike,
        tools: tuple[BaseTool | dict, ...],
        *,
        telemetry: AgentGraphTelemetry | None = None,
        tool_choice_policy: ToolChoicePolicy | None = None,
    ) -> None:
        self._chat_model = chat_model
        tc = openai_tool_choice_arg(tool_choice_policy)
        tool_list = list(tools)
        if tc is not None and tool_list:
            self._bound_model = chat_model.bind_tools(tool_list, tool_choice=tc)
        else:
            self._bound_model = chat_model.bind_tools(tool_list)
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            raise ValueError("工作台执行节点执行前 messages 不能为空")
        # 计划由 workbench_plan / workbench_replan 以 AIMessage 写入 messages，避免与 process_trace 脱节；
        # 不在此重复注入 state.plan，防止与消息历史重复占用上下文。
        sys_text = WORKBENCH_ORCHESTRATOR_PROMPT
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.workbench.react_loop",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            reply = await self._bound_model.ainvoke([SystemMessage(content=sys_text), *messages])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(reply)
            tel.tracer.record_llm_call(
                phase="workbench.react_loop",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._chat_model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            reply = _expand_parallel_tool_calls(reply)
            return {"messages": [reply]}


class WorkbenchAggregateNode:
    """工作台图节点：将计划与执行轨迹汇总为最终答复。"""

    __slots__ = ("_chat_model", "_telemetry")

    def __init__(
        self,
        chat_model: ChatModelLike,
        *,
        telemetry: AgentGraphTelemetry | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._telemetry = telemetry or AgentGraphTelemetry()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            raise ValueError("工作台汇总节点执行前 messages 不能为空")
        plan = str(state.get("plan") or "").strip()
        fan_in = _as_fan_in_summary(state.get("workbench_fan_in"))
        aggregate_prompt = (
            "你是工作台汇总器。请基于上文完整对话与工具执行结果给出最终答复。\n"
            "要求：\n"
            "0. **对齐用户原始问题**：先明确用户本轮提出的**字面问题/任务**，最终答复须直接完成该问题；"
            "若中间编排或子 Agent 正文曾偏题，须在成稿时收敛回用户所问，勿以工具侧发挥代替用户目标。\n"
            "1. 优先回答用户问题本身；可对重复表述合并改写，但**最终正文的信息量与结构层次不得明显薄于**对话中已产出的关键步骤（分条、表格、字段说明、注意事项等须保留或等价展开）。\n"
            "   禁止用少量「收尾引导/CTA」替代实质内容。\n"
            "2. 「避免复读」指避免机械重复同段原文，而非删除用户需要的流程、字段与避坑细节。\n"
            "3. 若存在多个子 Agent 结果，先对齐结论再组织答复；**同构成稿**：按用户问题的合理范围决定主干是概念、流程、对比、实现或其它形态；"
            "勿用固定模板套所有题。超范围发挥删或移附录，用户已要的深度不得克扣。\n"
            "4. 若计划与执行有偏差，简要说明并给出最终采用方案。\n"
            "5. 输出使用与用户一致的语言。\n"
            "6. **禁止**用「本答复已完成/严格对齐/请随时提出/不增不减」等**元话语独占终稿**；"
            "若上文已有分节、表格、步骤与字段说明，终稿必须**全文纳入或同构重组**，不得用一段收束话术顶替实质内容。"
            "（省 token 指删重复句，不是删用户需要的要点。）"
        )
        if plan:
            aggregate_prompt = f"{aggregate_prompt}\n\n【规划摘要】\n{plan}"
        if fan_in is not None:
            fan_in_text = _fan_in_summary_for_prompt(fan_in)
            aggregate_prompt = f"{aggregate_prompt}\n\n【fan-in 汇总】\n{fan_in_text}"
        tel = self._telemetry
        async with agent_graph_child_span(
            "agent.node.workbench.aggregate",
            trace_id=tel.trace_id,
            trace_parent_span_id=tel.trace_parent_span_id,
        ):
            t0 = time.perf_counter()
            prompt_msgs: list[BaseMessage] = [
                SystemMessage(content=aggregate_prompt),
                *messages,
            ]
            merged: AIMessageChunk | None = None
            async for chunk in self._chat_model.astream(prompt_msgs):
                if isinstance(chunk, AIMessageChunk):
                    merged = chunk if merged is None else merged + chunk
            if merged is None:
                raise ValueError("工作台汇总节点模型流式输出为空")
            reply = AIMessage(
                content=merged.content,
                tool_calls=list(merged.tool_calls) if merged.tool_calls else [],
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            pt, ct, tt = extract_token_usage_from_message(reply)
            tel.tracer.record_llm_call(
                phase="workbench.aggregate",
                duration_ms=duration_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
                model_name=chat_model_label(self._chat_model),
                request_id=tel.request_id,
                trace_id=tel.trace_id,
                trace_parent_span_id=tel.trace_parent_span_id,
            )
            return {"messages": [reply]}


class WorkbenchFanInNode:
    """工作台图节点：显式汇总子 Agent 调用结果，写入 ``state.workbench_fan_in``。"""

    __slots__ = ()

    async def __call__(self, state: LangGraphAgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        return {"workbench_fan_in": _collect_fan_in_summary(messages)}
