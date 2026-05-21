"""子 Agent 单次 / 并行调用、流式 progress 与返回体压缩（同模块）。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agent.adapters.tools.workbench.context import (
    WorkbenchConstants,
    WorkbenchRuntime,
    WorkbenchRuntimeBuilder,
    workbench_runtime_scope,
)
from app.agent.adapters.tools.workbench.parent_messages import ParentMessagesParseStrategy
from app.agent.adapters.tools.workbench.tool_support import (
    tool_names_from_agent_config_json,
    wb_err,
    wb_ok,
)
from app.agent.kernel import AgentKind, AgentProgressStage, AgentSseEventType
from app.agent.kernel.workbench_orchestration import (
    WORKBENCH_INVOKE_SUB_AGENT_TOOL,
    WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
)
from app.core.constants.agent import WORKBENCH_SUB_AGENT_TOOL_RESULT_MAX_CHARS
from app.core.constants.conversation import TOOL_HISTORY_METADATA_MAX_RESULT_CHARS
from app.repositories.agent_repo import AgentRepository
from app.schemas.agent import AgentInvokeRequest, ChatHistoryTurn, PromptEngineeringBody

_NESTED_TOOL_RESULT_CAP = min(8_000, TOOL_HISTORY_METADATA_MAX_RESULT_CHARS // 6)


def _clip_text(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    suf = "\n…(truncated)"
    return s[: max(0, cap - len(suf))].rstrip() + suf


class SubAgentPayloadCompactor:
    """子调用结果压缩：截断正文 / structured，并裁剪嵌套 ``tool_history`` / ``process_trace``。"""

    @staticmethod
    def compact(
        *,
        assistant_text: str | None,
        structured: Any,
        tokens: Any,
        sub_agent_id: int,
        sub_agent_kind: str,
        sub_tool_history: list[dict[str, Any]] | None = None,
        sub_process_trace: list[dict[str, Any]] | None = None,
        max_chars: int = WORKBENCH_SUB_AGENT_TOOL_RESULT_MAX_CHARS,
    ) -> dict[str, Any]:
        truncated = False
        text = (assistant_text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n…[assistant_text 已截断]"
            truncated = True

        out_struct: Any = structured
        structured_omitted = False
        if structured is not None:
            try:
                ser = json.dumps(structured, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                ser = str(structured)
            struct_cap = max(1024, max_chars // 2)
            if len(ser) > struct_cap:
                out_struct = None
                structured_omitted = True
                truncated = True

        payload: dict[str, Any] = {
            "assistant_text": text,
            "structured": out_struct,
            "tokens": tokens,
            "sub_agent_id": sub_agent_id,
            "sub_agent_kind": sub_agent_kind,
        }
        if structured_omitted:
            payload["structured_omitted"] = True
        if truncated:
            payload["result_truncated"] = True

        budget = WorkbenchConstants.NESTED_OBSERVABILITY_MAX_CHARS
        th = [dict(x) for x in sub_tool_history] if sub_tool_history else None
        pt = [dict(x) for x in sub_process_trace] if sub_process_trace else None
        nest_trunc = False

        def _nested_ser_size() -> int:
            try:
                blob = json.dumps(
                    {"sub_tool_history": th, "sub_process_trace": pt},
                    ensure_ascii=False,
                    default=str,
                )
            except (TypeError, ValueError):
                return budget + 1
            return len(blob)

        if th:
            for row in th:
                if row.get("phase") == "result" and isinstance(row.get("content"), str):
                    row["content"] = _clip_text(row["content"], _NESTED_TOOL_RESULT_CAP)
        if pt:
            for row in pt:
                t = row.get("text")
                if isinstance(t, str):
                    row["text"] = _clip_text(t, _NESTED_TOOL_RESULT_CAP)

        while th and _nested_ser_size() > budget:
            th.pop()
            nest_trunc = True
        while pt and _nested_ser_size() > budget:
            pt.pop()
            nest_trunc = True

        if th:
            payload["sub_tool_history"] = th
        if pt:
            payload["sub_process_trace"] = pt
        if nest_trunc:
            payload["sub_observability_truncated"] = True
        return payload


async def stream_pixel_agent_status(
    parent_ctx: WorkbenchRuntime,
    agent_id: int,
    status: str,
) -> None:
    """根层流式：像素小人忙闲（``active`` / ``waiting``）。"""
    q = parent_ctx.stream_event_queue
    if q is None:
        return
    await q.put(
        {
            "type": AgentSseEventType.PROGRESS.value,
            "stage": AgentProgressStage.PIXEL_AGENT_STATUS.value,
            "agent_id": agent_id,
            "status": status,
        }
    )


async def stream_sub_agent_progress(
    parent_ctx: WorkbenchRuntime,
    agent_id: int,
    sub_phase: str,
    *,
    invoke_tool: str = WORKBENCH_INVOKE_SUB_AGENT_TOOL,
) -> None:
    """根层流式时向 side queue 发 ``sub_agent`` progress。"""
    q = parent_ctx.stream_event_queue
    if q is None:
        return
    wba = parent_ctx.parent_invoke.agent_id
    if wba is None:
        return
    text = (
        f"子 Agent #{agent_id} · 执行中…"
        if sub_phase == "start"
        else f"子 Agent #{agent_id} · 已完成"
    )
    await q.put(
        {
            "type": AgentSseEventType.PROGRESS.value,
            "stage": AgentProgressStage.SUB_AGENT.value,
            "tool": invoke_tool,
            "text": text,
            "sub_phase": sub_phase,
            "active_agent_id": agent_id,
            "workbench_agent_id": wba,
        }
    )


async def run_sub_agent_invocation(
    parent_ctx: WorkbenchRuntime,
    agent_id: int,
    user_message: str,
    parent_messages: str | None,
    *,
    stream_invoke_tool: str = WORKBENCH_INVOKE_SUB_AGENT_TOOL,
) -> dict[str, Any]:
    """在「当前父工作台」上下文中执行一次子 Agent 调用。"""
    if parent_ctx.parent_invoke.workspace_namespace != parent_ctx.namespace:
        return wb_err(
            "NAMESPACE_INCONSISTENT",
            "运行时命名空间与请求体不一致",
            data={},
        )
    if parent_ctx.depth >= parent_ctx.max_depth:
        return wb_err(
            "MAX_DEPTH",
            f"嵌套深度超过上限（max={parent_ctx.max_depth}）",
            data={"max": parent_ctx.max_depth},
        )
    row = await AgentRepository.get_by_id(agent_id, db_manager=parent_ctx.db_manager)
    if row is None:
        return wb_err("NOT_FOUND", "子 Agent 不存在", data={"agent_id": agent_id})
    root_aid = parent_ctx.parent_invoke.agent_id
    if root_aid is not None and int(agent_id) == int(root_aid):
        return wb_err(
            "INVALID_TARGET",
            "不得将工作台根 Agent（当前编排自身）作为子调用目标；请为不同领域选用或创建其它子 Agent",
            data={"agent_id": agent_id},
        )
    if row.workspace_namespace != parent_ctx.namespace:
        return wb_err("NAMESPACE_MISMATCH", "子 Agent 不在当前工作区", data={"agent_id": agent_id})
    if str(row.status).strip().lower() != "active":
        return wb_err(
            "UNAVAILABLE",
            f"子 Agent 状态为 {row.status!r}，非 active，无法调用",
            data={"agent_id": agent_id, "status": row.status},
        )

    hist_turns: list[ChatHistoryTurn] = []
    if parent_messages and str(parent_messages).strip():
        try:
            hist_turns = ParentMessagesParseStrategy.default().parse(parent_messages)
        except ValueError as e:
            return wb_err("VALIDATION_ERROR", str(e), data={"field": "parent_messages"})

    child_kind = AgentKind(row.agent_kind)
    sp = str(row.system_prompt).strip() if row.system_prompt else None
    prompts: PromptEngineeringBody | None = None
    force_client_hist = False
    if hist_turns:
        prompts = PromptEngineeringBody(
            system_prompt=sp,
            chat_history=hist_turns,
            max_history_rounds=min(
                WorkbenchConstants.MAX_PARENT_MESSAGE_TURNS,
                max(10, len(hist_turns) + 2),
            ),
        )
        force_client_hist = True
    elif sp:
        prompts = PromptEngineeringBody(system_prompt=sp)

    cfg = row.config_json if isinstance(row.config_json, dict) else None
    child_tool_names = tool_names_from_agent_config_json(cfg)

    root_sid = parent_ctx.root_conversation_session_id
    root_wb_aid = parent_ctx.parent_invoke.agent_id
    share_root_session = bool(root_sid and root_wb_aid is not None)
    child_req = AgentInvokeRequest(
        agent_kind=child_kind,
        user_message=user_message.strip(),
        config_id=row.sys_model_id,
        tool_names=child_tool_names,
        strict_tool_names=False,
        model_identity=parent_ctx.parent_invoke.model_identity,
        hyperparameters=parent_ctx.parent_invoke.hyperparameters,
        response=parent_ctx.parent_invoke.response,
        output=parent_ctx.parent_invoke.output,
        workspace_namespace=parent_ctx.namespace,
        agent_id=row.id,
        prompts=prompts,
        workbench_force_client_history=force_client_hist,
        conversation_session_id=root_sid if share_root_session else None,
        conversation_owner_agent_id=root_wb_aid if share_root_session else None,
    )

    inner_rt = WorkbenchRuntimeBuilder.nested(parent_ctx)
    agent_svc = parent_ctx.agent_service

    wba = parent_ctx.parent_invoke.agent_id
    emit_pixel_agent_status = stream_invoke_tool == WORKBENCH_INVOKE_SUB_AGENT_TOOL and wba is not None
    if emit_pixel_agent_status:
        await stream_pixel_agent_status(parent_ctx, wba, "waiting")
        await stream_pixel_agent_status(parent_ctx, agent_id, "active")
    await stream_sub_agent_progress(parent_ctx, agent_id, "start", invoke_tool=stream_invoke_tool)
    try:
        async with workbench_runtime_scope(inner_rt):
            try:
                q = parent_ctx.stream_event_queue
                data = await agent_svc.invoke(
                    child_req,
                    request_id=parent_ctx.effective_request_id,
                    forward_pixel_progress_to=q,
                )
            except Exception as e:
                return wb_err(
                    "INVOKE_FAILED",
                    str(e),
                    data={"agent_id": agent_id},
                )
    finally:
        await stream_sub_agent_progress(parent_ctx, agent_id, "end", invoke_tool=stream_invoke_tool)
        if emit_pixel_agent_status and wba is not None:
            await stream_pixel_agent_status(parent_ctx, agent_id, "waiting")
            await stream_pixel_agent_status(parent_ctx, wba, "active")

    sub_th = [dict(h) for h in data.tool_history] if data.tool_history else None
    sub_pt = [p.model_dump(mode="json") for p in data.process_trace] if data.process_trace else None
    payload = SubAgentPayloadCompactor.compact(
        assistant_text=data.assistant_text,
        structured=data.structured,
        tokens=data.tokens,
        sub_agent_id=agent_id,
        sub_agent_kind=row.agent_kind,
        sub_tool_history=sub_th,
        sub_process_trace=sub_pt,
    )
    return wb_ok(payload, message="sub_agent_completed")


async def run_parallel_sub_agent_invocations(
    ctx: WorkbenchRuntime,
    tasks_json: str,
) -> dict[str, Any]:
    """并发调用多个子 Agent；``tasks_json`` 为 JSON 数组字符串。"""
    if ctx.parent_invoke.workspace_namespace != ctx.namespace:
        return wb_err(
            "NAMESPACE_INCONSISTENT",
            "运行时命名空间与请求体不一致",
            data={},
        )
    if ctx.depth >= ctx.max_depth:
        return wb_err(
            "MAX_DEPTH",
            f"嵌套深度超过上限（max={ctx.max_depth}）",
            data={"max": ctx.max_depth},
        )
    try:
        raw = json.loads(tasks_json)
    except json.JSONDecodeError as e:
        return wb_err(
            "VALIDATION_ERROR",
            f"tasks_json 不是合法 JSON: {e}",
            data={"field": "tasks_json"},
        )
    if not isinstance(raw, list):
        return wb_err("VALIDATION_ERROR", "tasks_json 须为 JSON 数组", data={"field": "tasks_json"})
    if len(raw) < 1:
        return wb_err("VALIDATION_ERROR", "至少包含 1 个子任务", data={"field": "tasks_json"})
    if len(raw) > WorkbenchConstants.MAX_PARALLEL_TASKS:
        return wb_err(
            "VALIDATION_ERROR",
            f"并行任务数不得超过 {WorkbenchConstants.MAX_PARALLEL_TASKS}",
            data={"field": "tasks_json", "max": WorkbenchConstants.MAX_PARALLEL_TASKS},
        )

    tasks: list[tuple[int, str, str | None]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return wb_err("VALIDATION_ERROR", f"tasks[{i}] 须为 JSON 对象", data={"index": i})
        try:
            agent_id_i = int(item["agent_id"])
        except (KeyError, TypeError, ValueError):
            return wb_err("VALIDATION_ERROR", f"tasks[{i}].agent_id 无效", data={"index": i})
        um = item.get("user_message")
        if not isinstance(um, str) or not um.strip():
            return wb_err(
                "VALIDATION_ERROR",
                f"tasks[{i}].user_message 须为非空字符串",
                data={"index": i},
            )
        pm_raw = item.get("parent_messages")
        pm_out: str | None = None
        if pm_raw is not None and str(pm_raw).strip():
            if not isinstance(pm_raw, str):
                return wb_err(
                    "VALIDATION_ERROR",
                    f"tasks[{i}].parent_messages 须为字符串或省略",
                    data={"index": i},
                )
            pm_out = pm_raw
        tasks.append((agent_id_i, um.strip(), pm_out))

    deduped: list[tuple[int, str, str | None]] = []
    seen_keys: set[tuple[int, str]] = set()
    for spec in tasks:
        k = (spec[0], spec[1])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(spec)
    tasks = deduped

    async def run_one(spec: tuple[int, str, str | None]) -> dict[str, Any]:
        return await run_sub_agent_invocation(
            ctx,
            spec[0],
            spec[1],
            spec[2],
            stream_invoke_tool=WORKBENCH_INVOKE_SUB_AGENTS_PARALLEL_TOOL,
        )

    outcomes = await asyncio.gather(*(run_one(t) for t in tasks), return_exceptions=True)

    items: list[dict[str, Any]] = []
    for i, res in enumerate(outcomes):
        aid = tasks[i][0]
        if isinstance(res, Exception):
            items.append(
                {
                    "index": i,
                    "agent_id": aid,
                    "code": "INVOKE_FAILED",
                    "message": str(res),
                    "data": {},
                }
            )
            continue
        if isinstance(res, dict):
            items.append(
                {
                    "index": i,
                    "agent_id": aid,
                    "code": res.get("code", "UNKNOWN"),
                    "message": res.get("message", ""),
                    "data": res.get("data"),
                }
            )
        else:
            items.append(
                {
                    "index": i,
                    "agent_id": aid,
                    "code": "INVOKE_FAILED",
                    "message": "unexpected_result_type",
                    "data": {"type": type(res).__name__},
                }
            )

    return wb_ok({"parallel": True, "items": items}, message="parallel_sub_agent_completed")
