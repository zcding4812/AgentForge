from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from app.agent.kernel.spec import AgentSseEventType
from app.core.constants.agent import AgentSseErrorCode
from app.schemas.agent import AgentProcessTraceStep, OrchestrationEdge, WorkbenchFanInSummary

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    message: str = Field(default="ok", description="接口级短提示")
    data: T


class PageMeta(BaseModel):
    page: int = Field(ge=1, description="当前页，从 1 开始")
    page_size: int = Field(ge=1, description="每页条数")
    total: int = Field(ge=0, description="总记录数")


class PagedData(BaseModel, Generic[T]):
    items: list[T]
    meta: PageMeta


class ErrorResponse(BaseModel):
    detail: str | list[dict] = Field(description="错误详情")


class AgentStreamEventStart(BaseModel):
    """首帧：元数据（agent / 模型 / 工具名列表）。"""

    type: Literal[AgentSseEventType.START] = AgentSseEventType.START
    agent_kind: str
    model: str
    provider: str
    config_id: str | int | None = None
    tool_names: list[str] = Field(default_factory=list)
    conversation_session_id: str | None = Field(
        default=None,
        description="本轮对话会话 ID（新对话时服务端创建；后续轮次请求头/体带上以续聊）",
    )


class AgentStreamEventProgress(BaseModel):
    """阶段进度（Plan/ReAct 等可在后续接 LangGraph 自定义流扩展）。"""

    type: Literal[AgentSseEventType.PROGRESS] = AgentSseEventType.PROGRESS
    stage: str
    tool: str | None = None
    text: str | None = None
    sub_phase: Literal["start", "end"] | None = Field(
        default=None,
        description="工作台子 Agent 进度：与 stage=sub_agent 联用",
    )
    active_agent_id: int | None = Field(default=None, description="当前执行的子 Agent id")
    workbench_agent_id: int | None = Field(
        default=None, description="工作台 orchestrator 的 agent id"
    )
    agent_id: int | None = Field(
        default=None,
        description="像素 progress：目标 agent_entity.id（工具条 / 状态）",
    )
    tool_id: str | None = Field(
        default=None,
        description="pixel_tool_start / pixel_tool_done：工具调用 id（与 LangChain tool_call id 对齐）",
    )
    tool_name: str | None = Field(default=None, description="pixel_tool_start：工具展示名")
    status: str | None = Field(
        default=None,
        description="pixel_tool_start：状态文案；pixel_agent_status：active 或 waiting",
    )
    permission_active: bool | None = Field(
        default=None,
        description="pixel_tool_start：是否处于权限确认态",
    )
    run_in_background: bool | None = Field(
        default=None,
        description="pixel_tool_start：是否后台工具条（避免主包额外 spawn 小人）",
    )


class AgentStreamEventPing(BaseModel):
    """心跳，防止网关空闲断开；前端可忽略。"""

    type: Literal[AgentSseEventType.PING] = AgentSseEventType.PING


class AgentStreamEventDelta(BaseModel):
    """每条 SSE 行 JSON：`type=delta`（增量文本，可多次）。"""

    type: Literal[AgentSseEventType.DELTA] = AgentSseEventType.DELTA
    text: str
    lane: Literal["exploring", "content"] | None = Field(
        default=None,
        description="工作台流式：`exploring` 写入顶部探索条，`content` 写入对语气泡；缺省等同 `content`",
    )


class AgentStreamEventDone(BaseModel):
    """`type=done`（与非流式 `AgentInvokeData` 收口一致）。"""

    type: Literal[AgentSseEventType.DONE] = AgentSseEventType.DONE
    assistant_text: str
    thinking_text: str | None = Field(
        default=None,
        description="与 AgentInvokeData.thinking_text 一致",
    )
    structured: dict | None = None
    tokens: int | None = None
    orchestration_edges: list[OrchestrationEdge] | None = Field(
        default=None,
        description="仅 workbench：与 AgentInvokeData.orchestration_edges 一致",
    )
    warnings: list[str] | None = Field(
        default=None,
        description="非致命提示（与 AgentInvokeData.warnings 一致）",
    )
    workbench_fan_in: WorkbenchFanInSummary | None = Field(
        default=None,
        description="仅 workbench：与 AgentInvokeData.workbench_fan_in 一致",
    )
    process_trace: list[AgentProcessTraceStep] | None = Field(
        default=None,
        description="与 AgentInvokeData.process_trace 一致",
    )
    tool_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="与 AgentInvokeData.tool_history 一致",
    )


class AgentStreamEventError(BaseModel):
    """`type=error`：结构化失败原因（前端展示 `message`，日志可用 `detail`）。"""

    type: Literal[AgentSseEventType.ERROR] = AgentSseEventType.ERROR
    code: str = Field(
        default=AgentSseErrorCode.EXECUTION_FAILED,
        description="稳定错误码，如 AGENT_EXECUTION_FAILED、LOOKUP_ERROR 等",
    )
    message: str
    detail: str | None = None


AgentStreamEvent = Annotated[
    AgentStreamEventStart
    | AgentStreamEventProgress
    | AgentStreamEventPing
    | AgentStreamEventDelta
    | AgentStreamEventDone
    | AgentStreamEventError,
    Field(discriminator="type"),
]
