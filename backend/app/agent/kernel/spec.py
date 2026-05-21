"""内核规格：编排类型、推理快照、提示槽位、图状态形状、SSE 约定。

`AgentState.messages` 与 LangChain ``BaseMessage`` 对齐；``ChatModelLike`` 与 ``BaseChatModel`` / ``RunnableBinding`` 对齐。
二者类型仅在静态检查下引用 ``langchain_core``（``TYPE_CHECKING``），运行时无 LangChain 导入（``ChatModelLike`` 运行时为 ``object`` 占位）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableBinding

    #: 图编译侧聊天模型：``BaseChatModel`` 或 ``bind`` 后的 ``RunnableBinding``（与适配器 ``GraphCompileDeps`` 对齐）
    ChatModelLike: TypeAlias = BaseChatModel | RunnableBinding
else:
    ChatModelLike = object  # 运行时占位；静态检查使用 ``TYPE_CHECKING`` 分支

# ---------------------------------------------------------------------------
# 编排策略
# ---------------------------------------------------------------------------


class AgentKind(StrEnum):
    """不同 kind 对应不同 LangGraph 装配（实现位于 adapters）。"""

    SIMPLE_CHAT = "simple_chat"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    WORKBENCH = "workbench"


# ---------------------------------------------------------------------------
# 输出控制与最终产物（与 HTTP schema 映射）
# ---------------------------------------------------------------------------

ResponseFormat = Literal["text", "json_object", "json_schema"]


@dataclass(frozen=True, slots=True)
class OutputControl:
    """展示层控制；与 ResponseConstraints（模型 API）正交。"""

    strip_thinking_blocks: bool = True
    include_tool_messages_in_raw: bool = False


ProcessTracePhase = Literal["step", "final"]
ProcessTraceStepKind = Literal["model", "tool_call", "tool_result"]


@dataclass(frozen=True, slots=True)
class ProcessTraceStep:
    """执行过程时间线：模型片段与工具调用 / 返回交错（入库 ``metadata.process_trace`` 与 API 对齐）。"""

    seq: int
    text: str
    phase: ProcessTracePhase = "step"
    kind: ProcessTraceStepKind = "model"
    tool_call_id: str | None = None
    name: str | None = None
    arguments: Any | None = None
    content: str | None = None

    def as_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seq": self.seq,
            "text": self.text,
            "phase": self.phase,
            "kind": self.kind,
        }
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        if self.arguments is not None:
            d["arguments"] = self.arguments
        if self.content is not None:
            d["content"] = self.content
        return d


@dataclass(frozen=True, slots=True)
class FinalAgentOutput:
    """领域侧最终输出。"""

    text: str
    structured: dict | None = None
    #: 与网关 ``usage.total_tokens`` 一致（多步时为各次调用之和）；解析不到时为 ``None``
    tokens: int | None = None
    #: 从 ``<thinking>`` / ``<redacted_thinking>`` 中解析的思考正文；不含标签；无则 ``None``
    thinking_text: str | None = None
    #: 仅 workbench：自 ``messages`` 解析的 ``workbench_invoke_sub_agent`` 顺序；元组为 ``(order, parent_agent_id, child_agent_id)``
    orchestration_edges: tuple[tuple[int, int, int], ...] | None = None
    #: 仅 workbench：fan-in 聚合摘要（子调用总数/成功失败/涉及子 Agent 列表等）
    workbench_fan_in: dict | None = None
    #: 多段模型输出（ReAct 等）；仅当末态存在多于一条 AIMessage 时非空，供回看
    process_trace: tuple[ProcessTraceStep, ...] | None = None
    #: 本轮（末条 Human 之后）工具调用与 ToolMessage 结果的时间序列；写入 ``metadata.tool_history``
    tool_history: tuple[dict[str, Any], ...] | None = None


# ---------------------------------------------------------------------------
# 推理快照（一次 Run 冻结）
# ---------------------------------------------------------------------------

ToolChoiceMode = Literal["auto", "none", "required", "specific"]


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model_name: str
    deployment: str | None = None
    base_url_override: str | None = None
    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceHyperparameters:
    """采样与长度；``top_k`` 在默认 OpenAI 适配器中可能不下发。"""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ResponseConstraints:
    response_format: ResponseFormat = "text"
    json_schema_id: str | None = None
    #: OpenAI Chat Completions `response_format.type=json_schema` 时对应的 JSON Schema 对象；省略或空对象时由工厂层使用默认可变 object（不必预填模板）
    response_json_schema: dict | None = None
    parallel_tool_calls: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolChoicePolicy:
    mode: ToolChoiceMode = "auto"
    forced_tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class InputContentFilterConfig:
    """与入库 ``config_json.input_filter`` 对齐。ReAct 显式图首节点；Simple Chat 仍走 ``create_agent`` 的 ``before_agent``；Plan-Execute / Workbench 不处理。"""

    enabled: bool = False
    banned_keywords: tuple[str, ...] = ()
    banned_regex: tuple[str, ...] = ()
    max_user_chars: int | None = None
    #: 超长时 `True` 则截断末条 Human（需消息带 `id` 以便 `RemoveMessage`），否则与违禁一样拒绝
    truncate_on_max: bool = False
    reject_message: str = "该输入未通过安全策略，请修改后重试。"


@dataclass(frozen=True, slots=True)
class ModelConfigSnapshot:
    identity: ModelIdentity
    hyperparameters: InferenceHyperparameters
    response: ResponseConstraints
    tool_choice: ToolChoicePolicy | None = None
    config_id: str | int | None = None
    input_content_filter: InputContentFilterConfig = field(
        default_factory=InputContentFilterConfig,
    )


# ---------------------------------------------------------------------------
# 提示槽位（四类消息的数据来源）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCallSlot:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolResultSlot:
    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class HistoryTurnSlot:
    user: str
    assistant: str
    tool_calls: tuple[ToolCallSlot, ...] = ()
    tool_results: tuple[ToolResultSlot, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptSlots:
    system_prompt: str | None = None
    context: str | None = None
    #: 较早对话的滚动摘要（窗外历史）；注入顺序：System → 本段 → 窗内 ``history_turns`` → 当前 Human
    rolling_summary: str | None = None
    #: 仅 workbench 根编排：为 True 时不注入 ``MessageBuilder`` 的通用默认 System；``system_prompt`` 仅作可选「流程补充」
    omit_platform_default_system: bool = False
    #: 仅 workbench：每次请求 prepare 时由服务端查库生成并注入（Markdown）；置于摘要之后、历史之前
    workbench_workspace_agent_catalog: str | None = None
    history_turns: tuple[HistoryTurnSlot, ...] = ()
    max_history_rounds: int = 10
    #: 可选：窗内历史段（不含前缀 System 与末条当前用户）的 token 上限；``langchain_core.messages.trim_messages`` 近似计数
    max_history_tokens: int | None = None


# ---------------------------------------------------------------------------
# 图状态（与适配器 LangGraphAgentState 对齐）
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: list[BaseMessage]
    plan: NotRequired[str]
    workbench_fan_in: NotRequired[dict]
    react_input_filter_stop: NotRequired[bool]


# ---------------------------------------------------------------------------
# 流式 SSE（与 schemas.response 对齐）
# ---------------------------------------------------------------------------


class AgentSseEventType(StrEnum):
    START = "start"
    PROGRESS = "progress"
    PING = "ping"
    DELTA = "delta"
    DONE = "done"
    ERROR = "error"


class AgentProgressStage(StrEnum):
    """`type=progress` 时的 `stage` 取值（可扩展）。"""

    GENERATING = "generating"
    #: 工作台子 Agent 单次调用起止（与 ``sub_phase``、``active_agent_id`` 等字段联用）
    SUB_AGENT = "sub_agent"
    #: invoke 前已按绑定知识库完成检索并注入上下文（有命中分片时由 SSE 首段 progress 发出）
    KNOWLEDGE_QUERY = "knowledge_query"
    #: 像素画布：工具条开始（与 ``tool_id`` / ``tool_name`` / ``status`` / ``permission_active`` / ``run_in_background`` / ``agent_id``）
    PIXEL_TOOL_START = "pixel_tool_start"
    #: 像素画布：单条工具结束
    PIXEL_TOOL_DONE = "pixel_tool_done"
    #: 像素画布：清空该 Agent 工具条
    PIXEL_TOOLS_CLEAR = "pixel_tools_clear"
    #: 像素画布：小人忙闲（``status`` 为 ``active`` | ``waiting``，``agent_id`` 为目标角色）
    PIXEL_AGENT_STATUS = "pixel_agent_status"


AGENT_PROGRESS_TEXT_GENERATING = "正在生成回复…"
