from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Final

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.agent.adapters.graph.factory import LangGraphAgentFactory, get_agent_checkpointer
from app.agent.adapters.output.module import AgentOutputModule
from app.agent.adapters.telemetry import get_app_agent_tracer
from app.agent.adapters.message_builder import MessageBuilder
from app.agent.kernel import (
    AGENT_PROGRESS_TEXT_GENERATING,
    AgentKind,
    AgentProgressStage,
    AgentSseEventType,
    AgentState,
    FinalAgentOutput,
    ModelConfigSnapshot,
    OutputControl,
    PromptSlots,
)
from app.agent.kernel.exceptions import AgentExecutionError
from app.agent.kernel.ports import (
    AgentGraphFactoryPort,
    AgentGraphTelemetry,
    AgentTracerPort,
    GraphBuildContext,
    ToolInvocationContext,
)
from app.agent.kernel.tool_runtime import tool_invocation_context
from app.config import get_settings
from app.core.context import RequestTraceContext


def _pixel_tool_call_id_name(tc: Any) -> tuple[str | None, str]:
    """从 LangChain ``tool_calls`` 单条解析 ``(id, name)``。"""
    if isinstance(tc, dict):
        raw_id = tc.get("id")
        name = tc.get("name")
        tid = raw_id if isinstance(raw_id, str) else str(raw_id) if raw_id is not None else None
        return tid, str(name or "tool")
    tid = getattr(tc, "id", None)
    name = getattr(tc, "name", None)
    tid_s = tid if isinstance(tid, str) else str(tid) if tid is not None else None
    return tid_s, str(name or "tool")


def _collect_pixel_tool_sse_from_stream_message(
    msg: BaseMessage | None,
    *,
    stream_agent_id: int,
    pixel_started_tool_ids: set[str],
    pixel_finished_tool_ids: set[str],
) -> list[dict[str, Any]]:
    """从 ``messages`` 流式单帧解析工具起止（与 ``values`` 全量态互补，ReAct 等常在 chunk 上先出现 tool_calls）。"""
    out: list[dict[str, Any]] = []
    if msg is None:
        return out
    if isinstance(msg, ToolMessage):
        raw_tid = getattr(msg, "tool_call_id", None)
        tid = raw_tid if isinstance(raw_tid, str) else str(raw_tid) if raw_tid is not None else None
        if not tid or tid in pixel_finished_tool_ids:
            return out
        pixel_finished_tool_ids.add(tid)
        out.append(
            {
                "type": AgentSseEventType.PROGRESS,
                "stage": AgentProgressStage.PIXEL_TOOL_DONE.value,
                "agent_id": stream_agent_id,
                "tool_id": tid,
            }
        )
        return out
    if isinstance(msg, AIMessage | AIMessageChunk):
        for tc in getattr(msg, "tool_calls", None) or []:
            tid, name = _pixel_tool_call_id_name(tc)
            if not tid or tid in pixel_started_tool_ids:
                continue
            pixel_started_tool_ids.add(tid)
            out.append(
                {
                    "type": AgentSseEventType.PROGRESS,
                    "stage": AgentProgressStage.PIXEL_TOOL_START.value,
                    "agent_id": stream_agent_id,
                    "tool_id": tid,
                    "tool_name": name,
                    "status": f"调用 {name}…",
                    "permission_active": False,
                    "run_in_background": True,
                }
            )
    return out


class AgentRunFacade:
    """应用门面：装配工厂、构建初始消息、执行图、输出收口；异常统一为 ``AgentExecutionError``。"""

    def __init__(self) -> None:
        self._graph_factory: Final[AgentGraphFactoryPort] = LangGraphAgentFactory()
        self._output_module: Final[AgentOutputModule] = AgentOutputModule()
        self._message_builder: Final[MessageBuilder] = MessageBuilder()
        self._tracer: Final[AgentTracerPort] = get_app_agent_tracer()

    @staticmethod
    def _resolve_checkpoint_thread_id(
        conversation_session_id: str | None,
        request_id: str | None,
    ) -> str:
        """本轮图 ``thread_id``（每轮唯一，与并发/子调用隔离）。"""
        rid = (request_id or "").strip() or uuid.uuid4().hex
        token = uuid.uuid4().hex[:12]
        sid = (conversation_session_id or "").strip()
        if sid:
            return f"conv:{sid}:req:{rid}:t:{token}"
        return f"ephemeral:req:{rid}:t:{token}"

    @staticmethod
    def _build_graph_run_config(*, recursion_limit: int, thread_id: str) -> dict[str, Any]:
        """与 ``CompiledStateGraph.ainvoke`` / ``astream`` 对齐的 ``config``。"""
        return {
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": thread_id},
        }

    @staticmethod
    async def _cleanup_checkpoint_thread(thread_id: str) -> None:
        """单轮结束后丢弃该线程的 checkpoint 条目（缓存回收）。"""
        cp = get_agent_checkpointer()
        await cp.adelete_thread(thread_id)

    @staticmethod
    def _observability_suffix_after_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
        """与 ``MessageBuilder`` 一致：本轮用户句为末条 ``HumanMessage``，其后的消息均为图在本轮追加。

        子 Agent 等场景下用「前缀长度」切片易与 LangGraph 归约后的列表失配，导致观测段为空或误混入主会话。
        """
        last_h = -1
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                last_h = i
        if last_h < 0:
            return []
        return list(messages[last_h + 1 :])

    @staticmethod
    def _resolve_observability_messages(
        final_messages: list[BaseMessage],
        *,
        initial_len: int,
    ) -> list[BaseMessage]:
        """process_trace / tool_history 专用片段：优先末条 Human 之后；空则回退 ``initial_len`` 后缀（不回退全表，避免主 Agent 历史渗入子调用）。"""
        suffix = AgentRunFacade._observability_suffix_after_last_human(final_messages)
        if suffix:
            return suffix
        if len(final_messages) >= initial_len > 0:
            return final_messages[initial_len:]
        return []

    async def run_chat_turn(
        self,
        kind: AgentKind,
        snapshot: ModelConfigSnapshot,
        user_message: str,
        output_control: OutputControl,
        *,
        request_id: str | None = None,
        tools: tuple[Any, ...] = (),
        prompt_slots: PromptSlots | None = None,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
        conversation_session_id: str | None = None,
        workbench_parent_agent_id: int | None = None,
    ) -> FinalAgentOutput:
        ctx = self._build_context(
            snapshot,
            request_id,
            tools,
            trace_id=trace_id,
            trace_parent_span_id=trace_parent_span_id,
            conversation_session_id=conversation_session_id,
        )
        with tool_invocation_context(ctx.tool_invocation):
            thread_id = self._resolve_checkpoint_thread_id(conversation_session_id, request_id)
            graph_cfg = self._build_graph_run_config(
                recursion_limit=get_settings().agent_graph_recursion_limit,
                thread_id=thread_id,
            )
            try:
                initial_messages = self._message_builder.build_initial_messages(
                    user_message, prompt_slots
                )
                initial_len = len(initial_messages)
                graph = self._graph_factory.build(kind, ctx)
                result = await graph.ainvoke({"messages": initial_messages}, config=graph_cfg)
                final_messages = list(result.get("messages") or [])
                obs_slice = self._resolve_observability_messages(
                    final_messages,
                    initial_len=initial_len,
                )
                wb_parent = workbench_parent_agent_id if kind == AgentKind.WORKBENCH else None
                wb_fan_in = result.get("workbench_fan_in") if kind == AgentKind.WORKBENCH else None
                return self._output_module.finalize(
                    final_messages,
                    snapshot.response,
                    output_control,
                    workbench_parent_agent_id=wb_parent,
                    workbench_fan_in=wb_fan_in if isinstance(wb_fan_in, dict) else None,
                    observability_messages=obs_slice,
                )
            except AgentExecutionError:
                raise
            except Exception as e:
                raise AgentExecutionError(f"Agent 执行失败: {e}") from e
            finally:
                await self._cleanup_checkpoint_thread(thread_id)

    async def stream_chat_turn(
        self,
        kind: AgentKind,
        snapshot: ModelConfigSnapshot,
        user_message: str,
        output_control: OutputControl,
        *,
        request_id: str | None = None,
        tools: tuple[BaseTool | dict, ...] = (),
        tool_names: tuple[str, ...] = (),
        prompt_slots: PromptSlots | None = None,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
        conversation_session_id: str | None = None,
        workbench_parent_agent_id: int | None = None,
        stream_agent_id: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式：start → progress → delta* → done；异常上抛由服务层序列化 error 帧。"""
        ctx = self._build_context(
            snapshot,
            request_id,
            tools,
            trace_id=trace_id,
            trace_parent_span_id=trace_parent_span_id,
            conversation_session_id=conversation_session_id,
        )
        with tool_invocation_context(ctx.tool_invocation):
            thread_id = self._resolve_checkpoint_thread_id(conversation_session_id, request_id)
            graph_cfg = self._build_graph_run_config(
                recursion_limit=get_settings().agent_graph_recursion_limit,
                thread_id=thread_id,
            )
            try:
                yield self._create_start_event(kind, snapshot, tool_names)
                yield self._create_progress_event()

                initial_messages = self._message_builder.build_initial_messages(
                    user_message, prompt_slots
                )
                initial_len = len(initial_messages)
                graph = self._graph_factory.build(kind, ctx)
                last_state: AgentState | None = None
                accumulated_stream_text = ""
                stream_source_node: str | None = None
                last_logged_graph_node: str | None = None
                pixel_started_tool_ids: set[str] = set()
                pixel_finished_tool_ids: set[str] = set()
                wb_agg = "workbench_aggregate"
                stream_modes: list[str] = (
                    ["messages", "values", "updates"]
                    if kind == AgentKind.WORKBENCH
                    else ["messages", "values"]
                )

                tid = ctx.telemetry.trace_id
                if tid:
                    # 延迟导入，避免 ``app.agent`` 包加载链与 ``tracing.engine`` 循环依赖
                    from app.core.tracing.engine import tracer as http_tracer

                    await http_tracer.log_event_async(
                        event_name="stream.astream.begin",
                        message=f"astream 开始 agent_kind={kind} modes={stream_modes!r}",
                        trace_id=tid,
                        payload={
                            "agent_kind": str(kind),
                            "stream_modes": list(stream_modes),
                            "workbench_aggregate_node": wb_agg
                            if kind == AgentKind.WORKBENCH
                            else None,
                        },
                    )

                async for event in graph.astream(
                    {"messages": initial_messages},
                    stream_mode=stream_modes,
                    config=graph_cfg,
                ):
                    if not isinstance(event, tuple) or len(event) != 2:
                        continue
                    mode, data = event
                    if mode == "updates" and isinstance(data, dict) and data:
                        stream_source_node = next(iter(data.keys()), None)
                        nk = stream_source_node
                        if (
                            tid
                            and isinstance(nk, str)
                            and nk.strip()
                            and nk != last_logged_graph_node
                        ):
                            last_logged_graph_node = nk
                            await http_tracer.log_event_async(
                                event_name="stream.graph_step",
                                message=f"图节点阶段完成: {nk}",
                                trace_id=tid,
                                payload={"langgraph_node": nk, "stream_mode": "updates"},
                            )
                    elif mode == "messages":
                        # LangGraph 的 messages 事件为 (token, metadata)；token 往往早于同节点的
                        # ``updates``，故 lane 必须以 metadata.langgraph_node 为准，否则会误判为 exploring。
                        stream_msg: BaseMessage | None = None
                        if isinstance(data, tuple) and len(data) >= 1:
                            stream_msg = data[0] if isinstance(data[0], BaseMessage) else None
                        elif isinstance(data, BaseMessage):
                            stream_msg = data
                        if stream_agent_id is not None:
                            for px_ev in _collect_pixel_tool_sse_from_stream_message(
                                stream_msg,
                                stream_agent_id=stream_agent_id,
                                pixel_started_tool_ids=pixel_started_tool_ids,
                                pixel_finished_tool_ids=pixel_finished_tool_ids,
                            ):
                                yield px_ev
                        meta_dict: dict[str, Any] | None = None
                        if isinstance(data, tuple) and len(data) >= 2 and isinstance(data[1], dict):
                            meta_dict = data[1]
                        raw_ln = meta_dict.get("langgraph_node") if meta_dict else None
                        lane_node: str | None = (
                            raw_ln
                            if isinstance(raw_ln, str) and raw_ln.strip()
                            else stream_source_node
                        )
                        if delta := self._extract_delta_text(data):
                            accumulated_stream_text += delta
                            delta_ev: dict[str, Any] = {
                                "type": AgentSseEventType.DELTA,
                                "text": delta,
                            }
                            if kind == AgentKind.WORKBENCH:
                                delta_ev["lane"] = "content" if lane_node == wb_agg else "exploring"
                            yield delta_ev
                    elif mode == "values" and isinstance(data, dict):
                        last_state = data
                        if stream_agent_id is not None:
                            msgs = list(data.get("messages") or [])
                            for m in msgs:
                                if isinstance(m, AIMessage):
                                    for tc in getattr(m, "tool_calls", None) or []:
                                        tid, name = _pixel_tool_call_id_name(tc)
                                        if not tid or tid in pixel_started_tool_ids:
                                            continue
                                        pixel_started_tool_ids.add(tid)
                                        yield {
                                            "type": AgentSseEventType.PROGRESS,
                                            "stage": AgentProgressStage.PIXEL_TOOL_START.value,
                                            "agent_id": stream_agent_id,
                                            "tool_id": tid,
                                            "tool_name": name,
                                            "status": f"调用 {name}…",
                                            "permission_active": False,
                                            "run_in_background": True,
                                        }
                                elif isinstance(m, ToolMessage):
                                    raw_tid = getattr(m, "tool_call_id", None)
                                    tid = (
                                        raw_tid
                                        if isinstance(raw_tid, str)
                                        else str(raw_tid)
                                        if raw_tid is not None
                                        else None
                                    )
                                    if not tid or tid in pixel_finished_tool_ids:
                                        continue
                                    pixel_finished_tool_ids.add(tid)
                                    yield {
                                        "type": AgentSseEventType.PROGRESS,
                                        "stage": AgentProgressStage.PIXEL_TOOL_DONE.value,
                                        "agent_id": stream_agent_id,
                                        "tool_id": tid,
                                    }

                final_messages = list((last_state or {}).get("messages") or [])
                if not final_messages:
                    raise AgentExecutionError("Agent 流式执行未得到末态消息")
                obs_slice = self._resolve_observability_messages(
                    final_messages,
                    initial_len=initial_len,
                )

                wb_parent = workbench_parent_agent_id if kind == AgentKind.WORKBENCH else None
                wb_fan_in: dict | None = None
                if kind == AgentKind.WORKBENCH and isinstance(last_state, dict):
                    raw_fan_in = last_state.get("workbench_fan_in")
                    if isinstance(raw_fan_in, dict):
                        wb_fan_in = raw_fan_in
                final_output = self._output_module.finalize(
                    final_messages,
                    snapshot.response,
                    output_control,
                    workbench_parent_agent_id=wb_parent,
                    workbench_fan_in=wb_fan_in,
                    observability_messages=obs_slice,
                )
                assistant_text = final_output.text
                if not (assistant_text or "").strip() and (accumulated_stream_text or "").strip():
                    # 末态 AIMessage 偶发为空串但 messages 流里已有 token（与图归约/厂商行为有关）
                    _, body = self._output_module.extract_thinking_and_body(accumulated_stream_text)
                    body = self._output_module.strip_thinking_markers(
                        body, output_control.strip_thinking_blocks
                    )
                    if body.strip():
                        assistant_text = body
                done_ev: dict[str, Any] = {
                    "type": AgentSseEventType.DONE,
                    "assistant_text": assistant_text,
                    "structured": final_output.structured,
                    "tokens": final_output.tokens,
                }
                if final_output.thinking_text:
                    done_ev["thinking_text"] = final_output.thinking_text
                if tid:
                    th = final_output.tool_history
                    await http_tracer.log_event_async(
                        event_name="stream.astream.done",
                        message=(
                            f"astream 结束 → done，末态消息 {len(final_messages)} 条，"
                            f"流式累计约 {len(accumulated_stream_text)} 字符"
                        ),
                        trace_id=tid,
                        payload={
                            "final_message_count": len(final_messages),
                            "stream_char_count": len(accumulated_stream_text),
                            "has_process_trace": bool(final_output.process_trace),
                            "tool_history_len": len(th) if th else 0,
                        },
                    )
                if final_output.orchestration_edges:
                    done_ev["orchestration_edges"] = [
                        {
                            "order": o,
                            "parent_agent_id": p,
                            "child_agent_id": c,
                        }
                        for o, p, c in final_output.orchestration_edges
                    ]
                if final_output.workbench_fan_in:
                    done_ev["workbench_fan_in"] = final_output.workbench_fan_in
                if final_output.process_trace:
                    done_ev["process_trace"] = [
                        s.as_metadata_dict() for s in final_output.process_trace
                    ]
                if final_output.tool_history:
                    done_ev["tool_history"] = list(final_output.tool_history)
                if stream_agent_id is not None:
                    yield {
                        "type": AgentSseEventType.PROGRESS,
                        "stage": AgentProgressStage.PIXEL_TOOLS_CLEAR.value,
                        "agent_id": stream_agent_id,
                    }
                yield done_ev
            except AgentExecutionError:
                raise
            except Exception as e:
                raise AgentExecutionError(f"Agent 执行失败: {e}") from e
            finally:
                await self._cleanup_checkpoint_thread(thread_id)

    def _build_context(
        self,
        snapshot: ModelConfigSnapshot,
        request_id: str | None,
        tools: tuple[BaseTool | dict, ...],
        *,
        trace_id: str | None = None,
        trace_parent_span_id: str | None = None,
        conversation_session_id: str | None = None,
    ) -> GraphBuildContext:
        safe_trace_id = trace_id if RequestTraceContext.is_w3c_trace_id(trace_id) else None
        tool_inv = ToolInvocationContext(
            session_id=conversation_session_id,
            request_id=request_id,
            trace_id=safe_trace_id,
        )
        return GraphBuildContext(
            model_snapshot=snapshot,
            tools=tools,
            telemetry=AgentGraphTelemetry(
                tracer=self._tracer,
                trace_id=safe_trace_id,
                trace_parent_span_id=trace_parent_span_id,
                request_id=request_id,
            ),
            tool_invocation=tool_inv,
        )

    def _create_start_event(
        self,
        kind: AgentKind,
        snapshot: ModelConfigSnapshot,
        tool_names: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "type": AgentSseEventType.START,
            "agent_kind": str(kind),
            "model": snapshot.identity.model_name,
            "provider": snapshot.identity.provider,
            "config_id": snapshot.config_id,
            "tool_names": list(tool_names),
        }

    def _create_progress_event(self) -> dict[str, Any]:
        return {
            "type": AgentSseEventType.PROGRESS,
            "stage": AgentProgressStage.GENERATING,
            "tool": None,
            "text": AGENT_PROGRESS_TEXT_GENERATING,
        }

    def _extract_delta_text(self, data: Any) -> str:
        msg: BaseMessage | None = None
        if isinstance(data, tuple) and len(data) >= 1:
            msg = data[0] if isinstance(data[0], BaseMessage) else None
        elif isinstance(data, BaseMessage):
            msg = data

        if not msg or not isinstance(msg, AIMessageChunk):
            return ""

        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return ""
