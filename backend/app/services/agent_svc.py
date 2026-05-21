from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import redis.asyncio as redis
from langchain_core.tools import BaseTool
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from app.agent.adapters.tools.registry import (
    ToolResolutionStrictError,
    resolve_tool_names_with_report,
)
from app.agent.adapters.tools.workbench import WORKBENCH_TOOL_INSTANCES
from app.agent.adapters.tools.workbench.context import (
    WorkbenchRuntimeBuilder,
    get_workbench_runtime,
    workbench_runtime_scope,
)
from app.agent.application.facade import AgentExecutionError, AgentRunFacade
from app.agent.kernel import (
    AgentKind,
    AgentProgressStage,
    AgentSseEventType,
    FinalAgentOutput,
    ModelConfigSnapshot,
    OutputControl,
    PromptSlots,
    parse_input_content_filter_from_config,
)
from app.agent.kernel.default_prompts import get_default_prompt_entry
from app.agent.kernel.spec import ProcessTraceStep
from app.core.constants.agent import AgentSseErrorCode
from app.core.constants.tracing import TRACE_SPAN_AGENT_WORK_TIMEOUT_MS
from app.core.context import RequestContext, TraceRuntimeContext
from app.core.logger import get_logger
from app.core.tracing import (
    SpanParentCapture,
    capture_span_parent,
    child_span_async,
)
from app.infrastructure.cache import get_agent_memory_cache
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.models.agent_mod import AgentEntity
from app.models.knowledge_mod import KnowledgeBase
from app.models.workspace_mod import WorkspaceNamespace
from app.repositories.agent_repo import AgentRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.namespace_repo import WorkspaceNamespaceRepository
from app.repositories.provider_repo import ProviderRepository
from app.schemas.agent import (
    AgentCreateBody,
    AgentDetailOut,
    AgentInvokeData,
    AgentInvokeRequest,
    AgentListData,
    AgentOut,
    AgentProcessTraceStep,
    AgentUpdateBody,
    DefaultPromptDetailData,
    KnowledgeInvokeCitation,
    OrchestrationEdge,
    WorkbenchWorkspaceCreateBody,
)
from app.schemas.conversation import ConversationMessageAppendBody
from app.services.agent_invoke import (
    InvokeRequestContextBuilder,
    InvokeSnapshotBuilder,
    PreparedInvoke,
    merge_workbench_stream,
)
from app.services.agent_invoke.invoke_memory import (
    AgentMemorySettingsLoader,
    PrepareInvokeMemoryPipeline,
)
from app.services.agent_invoke.knowledge_inject import enrich_prompt_slots_with_knowledge
from app.services.conversation_svc import ConversationService

logger = get_logger(__name__)

_AGENT_CREATE_ALLOWED_SYS_MODEL_TYPES: frozenset[str] = frozenset({"llm", "tts", "stt"})


class AgentEntityService:
    """`agent_entity` 表：列表、创建、详情、更新。"""

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager,
        redis: redis.Redis | None = None,
    ) -> None:
        self._db = db_manager
        self._redis = redis

    async def list_agents(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        q: str | None = None,
        agent_kinds: list[str] | None = None,
        exclude_agent_kinds: list[str] | None = None,
        workspace_namespace: str | None = None,
    ) -> AgentListData:
        page_res = await AgentRepository.list_page(
            page=page,
            page_size=page_size,
            q=q,
            agent_kinds=agent_kinds,
            exclude_agent_kinds=exclude_agent_kinds,
            workspace_namespace=workspace_namespace,
            db_manager=self._db,
        )
        items = [AgentOut.model_validate(r) for r in page_res.items]
        return AgentListData(
            items=items,
            total=page_res.total,
            page=page_res.page,
            page_size=page_res.page_size,
        )

    async def _validate_sys_model_for_new_agent(self, sys_model_id: int | None) -> int | None:
        if sys_model_id is None:
            return None
        pair = await ProviderRepository.get_model_with_provider(sys_model_id, db_manager=self._db)
        if pair is None:
            raise ValueError("挂载的对话模型不存在（请检查 sys_model_id）")
        m, p = pair
        if m.is_enabled != 1:
            raise ValueError("挂载的模型未启用")
        if p.status != 1:
            raise ValueError("模型所属提供商未启用")
        mt = (m.model_type or "").strip().lower()
        if mt not in _AGENT_CREATE_ALLOWED_SYS_MODEL_TYPES:
            raise ValueError(
                "该模型类型不宜作为 Agent 对话挂载；请选用对话类模型（可先列出 model_type=chat 的 sys_model）"
            )
        return sys_model_id

    async def create_agent(self, body: AgentCreateBody) -> AgentOut:
        if body.agent_kind == AgentKind.WORKBENCH:
            raise ValueError(
                "请使用 POST /api/agents/workbench 在指定命名空间创建工作台，或沿用运维种子数据"
            )
        resolved_model_id = await self._validate_sys_model_for_new_agent(body.sys_model_id)
        resolved_system_prompt = body.system_prompt
        if resolved_system_prompt is None and isinstance(body.config_json, dict):
            ps = body.config_json.get("prompt_system")
            if isinstance(ps, str) and ps.strip():
                resolved_system_prompt = ps.strip()
        wn = await WorkspaceNamespaceRepository.get_or_create_by_slug(
            body.workspace_namespace,
            db_manager=self._db,
        )
        try:
            row = await AgentRepository.create(
                db_manager=self._db,
                namespace_id=wn.id,
                name=body.name,
                description=body.description,
                agent_kind=body.agent_kind.value,
                status="active",
                sys_model_id=resolved_model_id,
                system_prompt=resolved_system_prompt,
                config_json=body.config_json,
            )
        except IntegrityError as e:
            orig = str(getattr(e, "orig", e))
            if (
                "uk_agent_entity_namespace_name" in orig
                or "uk_agent_entity_workspace_name" in orig
                or "uk_agents_name" in orig
                or "Duplicate entry" in orig
            ):
                raise ValueError("名称已存在，请更换") from e
            raise
        fresh = await AgentRepository.get_by_id(row.id, db_manager=self._db)
        if fresh is None:
            raise RuntimeError("Agent 创建后读取失败")
        return AgentOut.model_validate(fresh)

    async def create_workbench_workspace(self, body: WorkbenchWorkspaceCreateBody) -> AgentOut:
        """在指定命名空间创建唯一 ``workbench`` Agent；该命名空间已存在工作台时拒绝。"""
        ns = body.workspace_namespace.strip()
        wn = await WorkspaceNamespaceRepository.get_or_create_by_slug(ns, db_manager=self._db)
        if wn.workbench_agent_id is not None:
            raise ValueError("该命名空间已存在工作台，请切换至该工作区或更换命名空间")
        dup = await AgentRepository.list_page(
            page=1,
            page_size=1,
            agent_kinds=[AgentKind.WORKBENCH.value],
            namespace_id=wn.id,
            db_manager=self._db,
        )
        if dup.total > 0:
            raise ValueError("该命名空间已存在工作台，请切换至该工作区或更换命名空间")
        try:
            row = await AgentRepository.create(
                db_manager=self._db,
                namespace_id=wn.id,
                name=body.name,
                description=body.description,
                agent_kind=AgentKind.WORKBENCH.value,
                status="active",
                sys_model_id=None,
                system_prompt=None,
                config_json=None,
            )
            await WorkspaceNamespaceRepository.patch_by_id(
                wn.id,
                db_manager=self._db,
                workbench_agent_id=row.id,
            )
        except IntegrityError as e:
            orig = str(getattr(e, "orig", e))
            if (
                "uk_agent_entity_namespace_name" in orig
                or "uk_agent_entity_workspace_name" in orig
                or "uk_agents_name" in orig
                or "Duplicate entry" in orig
            ):
                raise ValueError("名称已存在，请更换工作台名称或命名空间") from e
            raise
        fresh = await AgentRepository.get_by_id(row.id, db_manager=self._db)
        if fresh is None:
            raise RuntimeError("工作台创建后读取失败")
        return AgentOut.model_validate(fresh)

    async def get_agent(self, agent_id: int) -> AgentDetailOut | None:
        row = await AgentRepository.get_by_id(agent_id, db_manager=self._db)
        if row is None:
            return None
        return AgentDetailOut.model_validate(row)

    async def update_agent(self, agent_id: int, body: AgentUpdateBody) -> AgentDetailOut:
        existing = await AgentRepository.get_by_id(agent_id, db_manager=self._db)
        if existing is None:
            raise LookupError("Agent 不存在")
        payload = body.model_dump(exclude_unset=True, mode="python")
        if "agent_kind" in payload and payload["agent_kind"] is not None:
            ak = payload["agent_kind"]
            new_kind = ak.value if isinstance(ak, AgentKind) else str(ak)
            if (
                existing.agent_kind == AgentKind.WORKBENCH.value
                and new_kind != AgentKind.WORKBENCH.value
            ):
                raise ValueError("系统工作台 Agent 的编排类型不可修改")
            if (
                existing.agent_kind != AgentKind.WORKBENCH.value
                and new_kind == AgentKind.WORKBENCH.value
            ):
                raise ValueError("不可将普通 Agent 改为工作台类型")
            payload["agent_kind"] = new_kind
        if not payload:
            raise ValueError("至少提供一个更新字段")
        try:
            patched = await AgentRepository.patch_by_id(agent_id, db_manager=self._db, **payload)
        except IntegrityError as e:
            orig = str(getattr(e, "orig", e))
            if (
                "uk_agent_entity_namespace_name" in orig
                or "uk_agent_entity_workspace_name" in orig
                or "uk_agents_name" in orig
                or "Duplicate entry" in orig
            ):
                raise ValueError("名称已存在，请更换") from e
            raise
        if patched is None:
            raise LookupError("Agent 不存在")
        if self._redis is not None:
            await get_agent_memory_cache().invalidate(self._redis, agent_id)
        fresh = await AgentRepository.get_by_id(agent_id, db_manager=self._db)
        if fresh is None:
            raise LookupError("Agent 不存在")
        return AgentDetailOut.model_validate(fresh)

    async def _purge_workspace_namespace(self, namespace_id: int) -> list[int]:
        """同一事务：解除工作台外键、硬删知识库行、删尽该空间 Agent、最后删命名空间行。

        ``namespace.workbench_agent_id`` 指向 Agent，须先置空再删 Agent 行。"""

        async def _write(session: AsyncSession) -> list[int]:
            res = await session.scalars(
                select(AgentEntity.id).where(AgentEntity.namespace_id == namespace_id)
            )
            agent_ids = [int(i) for i in res.all()]
            wn = await session.get(WorkspaceNamespace, namespace_id)
            if wn is not None:
                wn.workbench_agent_id = None
                await session.flush()
            await session.execute(
                delete(KnowledgeBase).where(KnowledgeBase.namespace_id == namespace_id)
            )
            await session.execute(
                delete(AgentEntity).where(AgentEntity.namespace_id == namespace_id)
            )
            if wn is not None:
                await session.delete(wn)
            return agent_ids

        return await self._db.run_write(_write)

    async def delete_agent(self, agent_id: int) -> None:
        row = await AgentRepository.get_by_id(agent_id, db_manager=self._db)
        if row is None:
            raise LookupError("Agent 不存在")
        if row.agent_kind == AgentKind.WORKBENCH.value:
            agent_ids = await self._purge_workspace_namespace(row.namespace_id)
            if self._redis is not None:
                cache = get_agent_memory_cache()
                for aid in agent_ids:
                    await cache.invalidate(self._redis, aid)
            return
        deleted = await AgentRepository.delete_by_id(agent_id, db_manager=self._db)
        if not deleted:
            raise LookupError("Agent 不存在")
        if self._redis is not None:
            await get_agent_memory_cache().invalidate(self._redis, agent_id)


_PIXEL_PROGRESS_STAGES_FWD = frozenset(
    {
        AgentProgressStage.PIXEL_TOOL_START.value,
        AgentProgressStage.PIXEL_TOOL_DONE.value,
        AgentProgressStage.PIXEL_TOOLS_CLEAR.value,
        AgentProgressStage.PIXEL_AGENT_STATUS.value,
    }
)


def _final_agent_output_from_stream_done(done: dict[str, Any]) -> FinalAgentOutput:
    """将流式末帧 ``done`` 字典还原为 ``FinalAgentOutput``（与 ``AgentRunFacade.stream_chat_turn`` 收口一致）。"""
    pt_raw = done.get("process_trace")
    proc: tuple[ProcessTraceStep, ...] | None = None
    if isinstance(pt_raw, list) and pt_raw:
        steps: list[ProcessTraceStep] = []
        for d in pt_raw:
            if not isinstance(d, dict):
                continue
            ph = d.get("phase")
            kd = d.get("kind")
            steps.append(
                ProcessTraceStep(
                    seq=int(d.get("seq", 0)),
                    text=str(d.get("text") or ""),
                    phase=ph if ph in ("step", "final") else "step",
                    kind=kd if kd in ("model", "tool_call", "tool_result") else "model",
                    tool_call_id=d.get("tool_call_id"),
                    name=d.get("name"),
                    arguments=d.get("arguments"),
                    content=d.get("content"),
                )
            )
        proc = tuple(steps) if steps else None

    orch_raw = done.get("orchestration_edges")
    orch: tuple[tuple[int, int, int], ...] | None = None
    if isinstance(orch_raw, list) and orch_raw:
        parsed: list[tuple[int, int, int]] = []
        for d in orch_raw:
            if not isinstance(d, dict):
                continue
            try:
                parsed.append(
                    (
                        int(d["order"]),
                        int(d["parent_agent_id"]),
                        int(d["child_agent_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        orch = tuple(parsed) if parsed else None

    th_raw = done.get("tool_history")
    th: tuple[dict[str, Any], ...] | None = None
    if isinstance(th_raw, list) and th_raw:
        th = tuple(dict(x) for x in th_raw if isinstance(x, dict))

    wf = done.get("workbench_fan_in")
    wf_out = wf if isinstance(wf, dict) else None

    tok = done.get("tokens")
    tok_i: int | None = int(tok) if tok is not None else None

    return FinalAgentOutput(
        text=str(done.get("assistant_text") or ""),
        structured=done.get("structured") if isinstance(done.get("structured"), dict) else None,
        tokens=tok_i,
        thinking_text=done.get("thinking_text") if isinstance(done.get("thinking_text"), str) else None,
        orchestration_edges=orch,
        workbench_fan_in=wf_out,
        process_trace=proc,
        tool_history=th,
    )


class AgentService:
    """薄封装：HTTP 体 → 领域快照与输出控制，再调用 `agent` 门面。

    **workbench 路径（摘要）**：``prepare_invoke`` 对账命名空间；根层挂载内置 ``workbench_*``（子 Agent 编排与同命名空间知识库/工具目录，不合并客户端
    ``tool_names``、不注入对话级知识检索）；``invoke`` / 流式入口注入
    :class:`~app.agent.adapters.tools.workbench.context.WorkbenchRuntime`；子 Agent 仅通过
    ``workbench_invoke_sub_agent`` 进入，嵌套时不再追加工作台工具集。详见 ``WorkbenchRuntime`` 模块文档字符串。
    """

    def __init__(
        self,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        redis: redis.Redis | None = None,
        chunk_repository: ChunkRepository | None = None,
    ) -> None:
        self._db = db_manager
        self._redis = redis
        self._chunk_repository = chunk_repository
        self._conversation: ConversationService | None = (
            ConversationService(db_manager, redis) if db_manager is not None else None
        )
        self._memory_loader = AgentMemorySettingsLoader(db_manager, redis)
        self._memory_pipeline = PrepareInvokeMemoryPipeline()

    @staticmethod
    def _format_workbench_workspace_agent_catalog(slug: str, rows: list[AgentEntity]) -> str:
        """将同命名空间 Agent 行格式化为 Markdown，注入 ``PromptSlots.workbench_workspace_agent_catalog``。"""
        lines = [
            f"以下为命名空间 `{slug}` 在 **本次 API 请求开始时** 由服务端从数据库查询得到的 Agent 列表（"
            f"每次请求都会重新查询；**同一请求内**若新建/删除了 Agent，请以 `workbench_list_agents` 为准）。",
            "",
        ]
        for r in rows:
            kind = (r.agent_kind or "").strip() or "?"
            name = (r.name or "").strip() or "(未命名)"
            desc = (r.description or "").strip()
            tail = f" — {desc[:200]}" if desc else ""
            if kind == AgentKind.WORKBENCH.value:
                lines.append(
                    f"- `agent_id={r.id}` **{name}** · `{kind}` · 工作台编排入口（**勿**作为 "
                    f"`workbench_invoke_sub_agent` 的 target）{tail}"
                )
            else:
                lines.append(
                    f"- `agent_id={r.id}` **{name}** · `{kind}` · 可作为子任务委派目标{tail}"
                )
        lines.append("")
        lines.append(
            "编排时请优先使用上表中的 `agent_id` 调用 `workbench_invoke_sub_agent` / 并行工具。"
        )
        return "\n".join(lines)

    @staticmethod
    def _process_trace_to_schema(out: FinalAgentOutput) -> list[AgentProcessTraceStep] | None:
        if not out.process_trace:
            return None
        return [
            AgentProcessTraceStep(
                seq=s.seq,
                text=s.text,
                phase=s.phase,
                kind=s.kind,
                tool_call_id=s.tool_call_id,
                name=s.name,
                arguments=s.arguments,
                content=s.content,
            )
            for s in out.process_trace
        ]

    def _build_invoke_conversation_extras(
        self,
        result: FinalAgentOutput,
        knowledge_citations: tuple[KnowledgeInvokeCitation, ...] | tuple[()],
    ) -> dict[str, Any] | None:
        extra: dict[str, Any] = {}
        if result.structured is not None:
            extra["structured"] = result.structured
        if knowledge_citations:
            extra["knowledge_citations"] = [c.model_dump(mode="json") for c in knowledge_citations]
        if result.orchestration_edges:
            extra["orchestration_edges"] = [
                {"order": o, "parent_agent_id": p, "child_agent_id": c}
                for o, p, c in result.orchestration_edges
            ]
        if result.workbench_fan_in:
            extra["workbench_fan_in"] = result.workbench_fan_in
        if result.thinking_text:
            extra["thinking_text"] = result.thinking_text
        pt_schema = self._process_trace_to_schema(result)
        if pt_schema:
            extra["process_trace"] = [p.model_dump(mode="json") for p in pt_schema]
        if result.tool_history:
            extra["tool_history"] = list(result.tool_history)
        return extra or None

    @staticmethod
    def _build_stream_conversation_extras(ev: dict[str, Any]) -> dict[str, Any] | None:
        _EXTRA_KEYS = (
            "structured", "knowledge_citations", "orchestration_edges",
            "workbench_fan_in", "thinking_text", "process_trace", "tool_history",
        )
        extra: dict[str, Any] = {k: ev[k] for k in _EXTRA_KEYS if ev.get(k) is not None}
        return extra or None

    async def _workbench_workspace_agent_catalog_text(self, workspace_namespace: str) -> str:
        """工作台根编排：每次 prepare_invoke 查库生成目录 Markdown（无跨请求缓存）。"""
        db = self._db
        if db is None:
            raise RuntimeError("workbench agent catalog requires database")
        slug = str(workspace_namespace).strip() or "default"
        rows = await AgentRepository.list_in_workspace_namespace(
            workspace_namespace=slug,
            db_manager=db,
        )
        if not rows:
            return (
                f"命名空间 `{slug}` 在本次请求查询时**暂无**入库 Agent 记录（以库为准，动态变化时可 `workbench_list_agents` 对账）。\n"
                "需要子 Agent 时可直接 `workbench_create_agent`（`agent_kind` 不可为 workbench）。"
            )
        return self._format_workbench_workspace_agent_catalog(slug, rows)

    async def prepare_invoke(
        self,
        body: AgentInvokeRequest,
        *,
        request_id: str | None,
        span_parent: SpanParentCapture | None = None,
    ) -> PreparedInvoke:
        async with child_span_async(
            "agent.prepare_invoke",
            span_parent,
            span_type="agent",
            component="agent-service",
            depth=2,
        ):
            return await self._prepare_invoke_impl(body, request_id=request_id)

    async def _normalize_workbench_invoke_request(
        self, body: AgentInvokeRequest
    ) -> AgentInvokeRequest:
        """workbench：须带 agent_id，且请求体 workspace_namespace 与入库实体一致。"""
        if body.agent_kind != AgentKind.WORKBENCH:
            return body
        if self._db is None:
            raise ValueError("workbench 编排需要数据库连接")
        if body.agent_id is None:
            raise ValueError(
                "workbench 编排须传入 agent_id，以便校验 workspace_namespace 与入库 Agent 一致"
            )
        row = await AgentRepository.get_by_id(body.agent_id, db_manager=self._db)
        if row is None:
            raise LookupError("Agent 不存在")
        if row.agent_kind != AgentKind.WORKBENCH.value:
            raise ValueError("入库 Agent 的 agent_kind 须为 workbench")
        if row.workspace_namespace != body.workspace_namespace:
            raise ValueError("workspace_namespace 与入库 Agent 所属命名空间不一致")
        return body

    async def _prepare_invoke_impl(
        self,
        body: AgentInvokeRequest,
        *,
        request_id: str | None,
    ) -> PreparedInvoke:
        body = await self._normalize_workbench_invoke_request(body)
        has_sys_model = body.config_id is not None
        if has_sys_model:
            if self._db is None:
                raise ValueError("使用 config_id 时数据库不可用")
            snap_b = InvokeSnapshotBuilder(self._db)
            base = await snap_b.from_sys_model(body.config_id)
            snapshot = InvokeSnapshotBuilder.merge_request(base, body)
        else:
            snapshot = InvokeSnapshotBuilder.from_request_body(body)

        if body.agent_id is not None and self._db is not None:
            row = await AgentRepository.get_by_id(body.agent_id, db_manager=self._db)
            cj = getattr(row, "config_json", None) if row is not None else None
            if isinstance(cj, dict):
                snapshot = replace(
                    snapshot,
                    input_content_filter=parse_input_content_filter_from_config(cj),
                )

        control = InvokeRequestContextBuilder.output_control(body)
        # 工作台根层：仅允许内置 workbench_*，不合并 HTTP/MCP 等 tool_names，避免误解析 strict
        resolve_names = body.tool_names or []
        resolve_strict = body.strict_tool_names
        if body.agent_kind == AgentKind.WORKBENCH:
            wr = get_workbench_runtime()
            if wr is None or wr.depth == 0:
                resolve_names = []
                resolve_strict = False
        try:
            resolved = resolve_tool_names_with_report(
                resolve_names,
                strict=resolve_strict,
            )
        except ToolResolutionStrictError as e:
            raise ValueError(str(e)) from e
        tool_warnings: tuple[str, ...] = ()
        if body.agent_kind == AgentKind.WORKBENCH:
            wr = get_workbench_runtime()
            if wr is None or wr.depth == 0:
                tools = tuple(WORKBENCH_TOOL_INSTANCES)
                warn_parts: list[str] = []
                if resolved.missing_names:
                    warn_parts.append(
                        f"以下工具名未注册，已忽略: {', '.join(resolved.missing_names)}",
                    )
                if warn_parts:
                    tool_warnings = ("\n".join(warn_parts),)
            else:
                tools = resolved.tools
                if resolved.missing_names:
                    tool_warnings = (
                        f"以下工具名未注册，已忽略: {', '.join(resolved.missing_names)}",
                    )
        else:
            tools = resolved.tools
            if resolved.missing_names:
                tool_warnings = (f"以下工具名未注册，已忽略: {', '.join(resolved.missing_names)}",)
        effective_request_id = body.request_id or request_id

        effective_sid = await self._resolve_effective_conversation_session(body)

        memory_settings, knowledge_binding = await self._memory_loader.load_for_invoke(body)

        prompt_slots = await self._memory_pipeline.build_prompt_slots(
            body=body,
            effective_sid=effective_sid,
            memory_settings=memory_settings,
            conversation=self._conversation,
            has_db=self._db is not None,
        )
        # 工作台编排不注入知识库检索（子 Agent 可自行绑定知识库与工具）
        if body.agent_kind == AgentKind.WORKBENCH:
            knowledge_citations = ()
        else:
            prompt_slots, knowledge_citations = await enrich_prompt_slots_with_knowledge(
                self._db,
                self._chunk_repository,
                body,
                prompt_slots,
                knowledge_binding,
            )

        # workbench：每次 HTTP invoke/stream 在 prepare 阶段查库拉取当前命名空间 Agent 列表（无缓存）
        if body.agent_kind == AgentKind.WORKBENCH and self._db is not None:
            cat = await self._workbench_workspace_agent_catalog_text(body.workspace_namespace)
            prompt_slots = (
                replace(
                    prompt_slots,
                    workbench_workspace_agent_catalog=cat,
                    omit_platform_default_system=True,
                )
                if prompt_slots
                else PromptSlots(
                    workbench_workspace_agent_catalog=cat,
                    omit_platform_default_system=True,
                )
            )

        return PreparedInvoke(
            snapshot=snapshot,
            output_control=control,
            tools=tools,
            prompt_slots=prompt_slots,
            effective_request_id=effective_request_id,
            conversation_session_id=effective_sid,
            invoke_body=body,
            tool_warnings=tool_warnings,
            knowledge_citations=knowledge_citations,
        )

    async def _assert_agents_share_workspace_namespace(
        self,
        owner_agent_id: int,
        invoke_agent_id: int,
    ) -> None:
        """跨 Agent 写入同一会话时：两实体须存在且 ``workspace_namespace`` 一致。"""
        if owner_agent_id == invoke_agent_id:
            return
        if self._db is None:
            raise ValueError("数据库不可用，无法校验跨 Agent 会话归属")
        row_o = await AgentRepository.get_by_id(owner_agent_id, db_manager=self._db)
        row_i = await AgentRepository.get_by_id(invoke_agent_id, db_manager=self._db)
        if row_o is None or row_i is None:
            raise ValueError("会话归属或执行方 Agent 不存在")
        ns_o = (str(row_o.workspace_namespace or "").strip() or "default").lower()
        ns_i = (str(row_i.workspace_namespace or "").strip() or "default").lower()
        if ns_o != ns_i:
            raise ValueError(
                "conversation_owner_agent_id 与 agent_id 须属同一 workspace_namespace 方可共享会话"
            )

    async def _resolve_effective_conversation_session(
        self,
        body: AgentInvokeRequest,
    ) -> str | None:
        sid = InvokeRequestContextBuilder.conversation_session_id(body)
        if sid:
            if self._db is None:
                raise ValueError("传入 conversation_session_id 时须配置数据库")
            assert body.agent_id is not None
            assert self._conversation is not None
            owner_aid: int
            if body.conversation_owner_agent_id is not None:
                wr = get_workbench_runtime()
                if wr is not None:
                    root_wb = wr.parent_invoke.agent_id
                    if root_wb is None or body.conversation_owner_agent_id != root_wb:
                        raise ValueError("conversation_owner_agent_id 须为根工作台的 agent_id")
                    if wr.root_conversation_session_id and sid != wr.root_conversation_session_id:
                        raise ValueError("conversation_session_id 与根工作台当前会话不一致")
                    owner_aid = body.conversation_owner_agent_id
                else:
                    owner_aid = body.conversation_owner_agent_id
                    await self._conversation.require_active_session_for_agent(
                        sid,
                        agent_id=owner_aid,
                    )
                    if body.agent_id != owner_aid:
                        await self._assert_agents_share_workspace_namespace(
                            owner_aid,
                            body.agent_id,
                        )
                    return sid
            else:
                owner_aid = body.agent_id
            await self._conversation.require_active_session_for_agent(
                sid,
                agent_id=owner_aid,
            )
            return sid
        if body.agent_id is None:
            return None
        if self._db is None:
            raise ValueError("传入 agent_id 以持久化对话时须配置数据库")
        assert self._conversation is not None
        created = await self._conversation.create_session_for_invoke(
            agent_id=body.agent_id,
            user_message=body.user_message,
        )
        return created.session_id

    async def persist_conversation_user_for_stream(
        self,
        body: AgentInvokeRequest,
        *,
        conversation_session_id: str | None,
    ) -> None:
        """在建立 SSE 响应前写入本轮用户消息（与 ``prepare_invoke`` 中的会话解析配套）。"""
        if not conversation_session_id:
            return
        assert self._conversation is not None
        um = await self._conversation.append_message(
            conversation_session_id,
            ConversationMessageAppendBody(
                role="user",
                content=body.user_message,
                agent_id=body.agent_id,
                metadata=InvokeRequestContextBuilder.conversation_metadata(body),
            ),
        )
        if um is None:
            raise ValueError("无法写入用户消息（会话不可用）")

    async def _run_turn_via_stream_pixel_bridge(
        self,
        *,
        prepared: PreparedInvoke,
        trace_id: str | None,
        trace_parent_span_id: str | None,
        forward_pixel_progress_to: asyncio.Queue[dict[str, Any]],
    ) -> FinalAgentOutput:
        """与 ``stream_events_after_prepare`` 同源执行一轮图，并把 ``pixel_*`` 进度写入队列（``agent_id`` 为请求体中的执行方）。"""
        ib = prepared.invoke_body
        done_ev: dict[str, Any] | None = None
        async for ev in AgentService.stream_events_after_prepare(
            ib,
            prepared.snapshot,
            prepared.output_control,
            prepared.tools,
            prepared.prompt_slots,
            prepared.effective_request_id,
            trace_id=trace_id,
            trace_parent_span_id=trace_parent_span_id,
            conversation_session_id=prepared.conversation_session_id,
        ):
            et = ev.get("type")
            if et == AgentSseEventType.PROGRESS.value:
                st = ev.get("stage")
                if st in _PIXEL_PROGRESS_STAGES_FWD:
                    await forward_pixel_progress_to.put(ev)
            elif et == AgentSseEventType.DONE.value:
                done_ev = ev
            elif et == AgentSseEventType.ERROR.value:
                raise AgentExecutionError(
                    str(ev.get("message") or ev.get("detail") or "Agent 流式执行失败")
                )
        if done_ev is None:
            raise AgentExecutionError("Agent 流式执行未收到 done")
        return _final_agent_output_from_stream_done(done_ev)

    async def invoke(
        self,
        body: AgentInvokeRequest,
        *,
        request_id: str | None = None,
        forward_pixel_progress_to: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> AgentInvokeData:
        span_parent = capture_span_parent(RequestContext.get_request())
        prepared = await self.prepare_invoke(
            body,
            request_id=request_id,
            span_parent=span_parent,
        )
        ib = prepared.invoke_body
        if prepared.conversation_session_id and self._db and self._conversation:
            um = await self._conversation.append_message(
                prepared.conversation_session_id,
                ConversationMessageAppendBody(
                    role="user",
                    content=ib.user_message,
                    agent_id=ib.agent_id,
                    metadata=InvokeRequestContextBuilder.conversation_metadata(ib),
                ),
            )
            if um is None:
                raise ValueError("无法写入用户消息（会话不可用）")

        facade = AgentRunFacade()
        async with child_span_async(
            "agent.run_turn",
            span_parent,
            span_type="agent",
            component="agent-service",
            depth=2,
            timeout_ms=TRACE_SPAN_AGENT_WORK_TIMEOUT_MS,
        ):
            tid = span_parent.trace_id if span_parent else None
            pid = TraceRuntimeContext.get_current_span_id()
            if forward_pixel_progress_to is not None:
                if ib.agent_kind == AgentKind.WORKBENCH:
                    if self._db is None:
                        raise ValueError("workbench 编排需要数据库连接")
                    wb = WorkbenchRuntimeBuilder.top_level(
                        db_manager=self._db,
                        namespace=ib.workspace_namespace,
                        agent_service=self,
                        parent_invoke=ib,
                        effective_request_id=prepared.effective_request_id,
                        redis=self._redis,
                        chunk_repository=self._chunk_repository,
                        root_conversation_session_id=prepared.conversation_session_id,
                    )
                    async with workbench_runtime_scope(wb):
                        result = await self._run_turn_via_stream_pixel_bridge(
                            prepared=prepared,
                            trace_id=tid,
                            trace_parent_span_id=pid,
                            forward_pixel_progress_to=forward_pixel_progress_to,
                        )
                else:
                    result = await self._run_turn_via_stream_pixel_bridge(
                        prepared=prepared,
                        trace_id=tid,
                        trace_parent_span_id=pid,
                        forward_pixel_progress_to=forward_pixel_progress_to,
                    )
            elif ib.agent_kind == AgentKind.WORKBENCH:
                if self._db is None:
                    raise ValueError("workbench 编排需要数据库连接")
                wb = WorkbenchRuntimeBuilder.top_level(
                    db_manager=self._db,
                    namespace=ib.workspace_namespace,
                    agent_service=self,
                    parent_invoke=ib,
                    effective_request_id=prepared.effective_request_id,
                    redis=self._redis,
                    chunk_repository=self._chunk_repository,
                    root_conversation_session_id=prepared.conversation_session_id,
                )
                async with workbench_runtime_scope(wb):
                    result = await facade.run_chat_turn(
                        ib.agent_kind,
                        prepared.snapshot,
                        ib.user_message,
                        prepared.output_control,
                        request_id=prepared.effective_request_id,
                        tools=prepared.tools,
                        prompt_slots=prepared.prompt_slots,
                        trace_id=tid,
                        trace_parent_span_id=pid,
                        conversation_session_id=prepared.conversation_session_id,
                        workbench_parent_agent_id=ib.agent_id,
                    )
            else:
                result = await facade.run_chat_turn(
                    ib.agent_kind,
                    prepared.snapshot,
                    ib.user_message,
                    prepared.output_control,
                    request_id=prepared.effective_request_id,
                    tools=prepared.tools,
                    prompt_slots=prepared.prompt_slots,
                    trace_id=tid,
                    trace_parent_span_id=pid,
                    conversation_session_id=prepared.conversation_session_id,
                )

        if prepared.conversation_session_id and self._db and self._conversation:
            try:
                extra = self._build_invoke_conversation_extras(
                    result, prepared.knowledge_citations,
                )
                am = await self._conversation.append_message(
                    prepared.conversation_session_id,
                    ConversationMessageAppendBody(
                        role="assistant",
                        content=result.text,
                        agent_id=ib.agent_id,
                        metadata=InvokeRequestContextBuilder.conversation_metadata(ib, extra=extra),
                        tokens=result.tokens,
                    ),
                )
                if am is None:
                    logger.warning(
                        "助手消息未写入对话库 | session_id=%s",
                        prepared.conversation_session_id,
                    )
            except Exception:
                logger.exception(
                    "写入助手消息失败 | session_id=%s",
                    prepared.conversation_session_id,
                )

        kc_out: list[KnowledgeInvokeCitation] | None = (
            list(prepared.knowledge_citations) if prepared.knowledge_citations else None
        )
        orch_out: list[OrchestrationEdge] | None = None
        if result.orchestration_edges:
            orch_out = [
                OrchestrationEdge(order=o, parent_agent_id=p, child_agent_id=c)
                for o, p, c in result.orchestration_edges
            ]

        return AgentInvokeData(
            assistant_text=result.text,
            thinking_text=result.thinking_text,
            structured=result.structured,
            tokens=result.tokens,
            conversation_session_id=prepared.conversation_session_id,
            warnings=list(prepared.tool_warnings) if prepared.tool_warnings else None,
            knowledge_citations=kc_out,
            orchestration_edges=orch_out,
            workbench_fan_in=result.workbench_fan_in,
            process_trace=self._process_trace_to_schema(result),
            tool_history=list(result.tool_history) if result.tool_history else None,
        )

    @staticmethod
    def get_default_prompt(kind: str) -> DefaultPromptDetailData | None:
        ent = get_default_prompt_entry(kind)
        if ent is None:
            return None
        return DefaultPromptDetailData(
            id=ent.id,
            label=ent.label,
            description=ent.description,
            text=ent.text,
        )

    @staticmethod
    async def stream_events_after_prepare(
        body: AgentInvokeRequest,
        snapshot: ModelConfigSnapshot,
        control: OutputControl,
        tools: tuple[BaseTool | dict, ...],
        prompt_slots: PromptSlots | None,
        effective_request_id: str | None,
        *,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
        conversation_session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式事件迭代。勿在此处包 ``child_span_async``（与 ``ContextVar`` 不兼容）；span 见 ``iter_invoke_stream_sse_bytes``。"""
        facade = AgentRunFacade()
        tool_names = tuple(body.tool_names or [])
        wb_pid = body.agent_id if body.agent_kind == AgentKind.WORKBENCH else None
        async for ev in facade.stream_chat_turn(
            body.agent_kind,
            snapshot,
            body.user_message,
            control,
            request_id=effective_request_id,
            tools=tools,
            tool_names=tool_names,
            prompt_slots=prompt_slots,
            trace_id=trace_id,
            trace_parent_span_id=trace_parent_span_id,
            conversation_session_id=conversation_session_id,
            workbench_parent_agent_id=wb_pid,
            stream_agent_id=body.agent_id,
        ):
            yield ev

    async def iter_invoke_stream_sse_bytes(
        self,
        body: AgentInvokeRequest,
        prepared: PreparedInvoke,
        _request: Request,
        *,
        span_parent: SpanParentCapture | None = None,
    ) -> AsyncIterator[bytes]:
        """编排 SSE：逐帧写出。

        不与 ``request.is_disconnected()`` 做竞态：在部分 ASGI/代理环境下断连探测可能早于末帧就绪，
        导致提前结束响应、**不发 ``done``**。客户端真正断开时，后续 ``yield`` 会失败，由运行时结束流。
        """

        async def raw_events(
            trace_id: str | None,
            trace_parent_span_id: str | None,
        ) -> AsyncIterator[dict[str, Any]]:
            ib = prepared.invoke_body
            base_iter = AgentService.stream_events_after_prepare(
                ib,
                prepared.snapshot,
                prepared.output_control,
                prepared.tools,
                prepared.prompt_slots,
                prepared.effective_request_id,
                trace_id=trace_id,
                trace_parent_span_id=trace_parent_span_id,
                conversation_session_id=prepared.conversation_session_id,
            )
            if ib.agent_kind == AgentKind.WORKBENCH and self._db is not None:
                wq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                stream_iter = merge_workbench_stream(base_iter, wq)
                wb = WorkbenchRuntimeBuilder.top_level(
                    db_manager=self._db,
                    namespace=ib.workspace_namespace,
                    agent_service=self,
                    parent_invoke=ib,
                    effective_request_id=prepared.effective_request_id,
                    redis=self._redis,
                    chunk_repository=self._chunk_repository,
                    stream_event_queue=wq,
                    root_conversation_session_id=prepared.conversation_session_id,
                )
                async with workbench_runtime_scope(wb):
                    async for ev in stream_iter:
                        yield ev
            else:
                async for ev in base_iter:
                    yield ev

        def _error_frame(message: str, detail: str) -> bytes:
            err = json.dumps(
                {
                    "type": AgentSseEventType.ERROR,
                    "code": AgentSseErrorCode.EXECUTION_FAILED,
                    "message": message,
                    "detail": detail,
                },
                ensure_ascii=False,
            )
            return f"data: {err}\n\n".encode()

        effective_sid = prepared.conversation_session_id

        async def stream_with_knowledge_hint() -> AsyncIterator[dict[str, Any]]:
            if (
                len(prepared.knowledge_citations) > 0
                and prepared.invoke_body.agent_kind != AgentKind.WORKBENCH
            ):
                kq: dict[str, Any] = {
                    "type": AgentSseEventType.PROGRESS,
                    "stage": AgentProgressStage.KNOWLEDGE_QUERY.value,
                    "text": "已检索知识库，准备生成…",
                    "tool": "knowledge_retrieval",
                }
                if prepared.invoke_body.agent_id is not None:
                    kq["agent_id"] = prepared.invoke_body.agent_id
                yield kq
            async for ev in raw_events(
                span_parent.trace_id if span_parent else None,
                TraceRuntimeContext.get_current_span_id(),
            ):
                yield ev

        try:
            async with child_span_async(
                "agent.stream_turn",
                span_parent,
                span_type="agent",
                component="agent-service",
                depth=2,
                timeout_ms=TRACE_SPAN_AGENT_WORK_TIMEOUT_MS,
            ):
                async for ev in stream_with_knowledge_hint():
                    if effective_sid and ev.get("type") == AgentSseEventType.START:
                        ev = {**ev, "conversation_session_id": effective_sid}
                    if ev.get("type") == AgentSseEventType.DONE:
                        done_patch: dict[str, Any] = {}
                        if prepared.tool_warnings:
                            done_patch["warnings"] = list(prepared.tool_warnings)
                        if prepared.knowledge_citations:
                            done_patch["knowledge_citations"] = [
                                c.model_dump(mode="json") for c in prepared.knowledge_citations
                            ]
                        if done_patch:
                            ev = {**ev, **done_patch}
                    try:
                        line = json.dumps(ev, ensure_ascii=False)
                    except TypeError as e:
                        yield _error_frame("流式序列化失败", str(e))
                        return
                    yield f"data: {line}\n\n".encode()
                    if (
                        effective_sid
                        and self._db
                        and self._conversation
                        and ev.get("type") == AgentSseEventType.DONE
                    ):
                        try:
                            extra_assist = self._build_stream_conversation_extras(ev)
                            await self._conversation.append_message(
                                effective_sid,
                                ConversationMessageAppendBody(
                                    role="assistant",
                                    content=str(ev.get("assistant_text") or ""),
                                    agent_id=prepared.invoke_body.agent_id,
                                    metadata=InvokeRequestContextBuilder.conversation_metadata(
                                        prepared.invoke_body, extra=extra_assist
                                    ),
                                    tokens=ev.get("tokens"),
                                ),
                            )
                        except Exception:
                            logger.exception(
                                "流式结束写入助手消息失败 | session_id=%s", effective_sid
                            )
        except AgentExecutionError as e:
            yield _error_frame(str(e), str(e))
