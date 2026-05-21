from datetime import datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.response import PageMeta

ConversationMessageRole: TypeAlias = Literal["user", "assistant", "system"]


class ConversationSessionCreateBody(BaseModel):
    """创建会话；须绑定 ``agent_entity`` 主键。"""

    agent_id: int = Field(ge=1, description="工作区 Agent 主键 agent_entity.id")
    title: str | None = Field(default=None, max_length=255, description="可选；默认「新对话」")

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class ConversationSessionPatchBody(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    status: Literal[0, 1] | None = Field(
        default=None,
        description="1=活跃，0=关闭；关闭后仍可读历史，除非已软删除",
    )

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v


class ConversationMessageAppendBody(BaseModel):
    role: ConversationMessageRole = Field(description="user / assistant / system")
    content: str = Field(min_length=1)
    content_type: str = Field(default="text", max_length=32)
    agent_id: int | None = Field(
        default=None,
        ge=1,
        description="消息绑定的 agent_entity.id；省略时使用会话 conversation_session.agent_id",
    )
    metadata: dict | None = Field(default=None, description="模型、耗时、工具调用等扩展 JSON")
    tokens: int | None = Field(
        default=None,
        ge=0,
        description="可选覆盖：user 默认 tiktoken 估算；assistant 为网关 usage 总 token",
    )

    @field_validator("content", "content_type", mode="after")
    @classmethod
    def _strip_message_text(cls, v: str) -> str:
        return v.strip()


class ConversationSessionOut(BaseModel):
    """会话行输出；与 ``AgentConversationSession`` 列一致。"""

    session_id: str
    agent_id: int
    title: str
    status: int
    created_at: datetime
    updated_at: datetime
    summary: str | None = Field(default=None, description="滚动摘要（窗外历史压缩）")
    summary_version: int = Field(default=0, description="摘要乐观锁版本")
    summary_job_status: int = Field(
        default=0,
        description="摘要任务：0 idle，1 pending，2 running",
    )
    summary_anchor_turn_index: int = Field(
        default=0,
        description="上次成功摘要时的 user 轮次锚点，用于增量触发",
    )

    model_config = ConfigDict(from_attributes=True)


class ConversationMessageOut(BaseModel):
    """消息行输出；API 字段 ``metadata`` 对应 ORM 属性 ``message_metadata``。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    agent_id: int = Field(description="本条消息归属的 agent_entity.id")
    role: ConversationMessageRole
    content: str
    content_type: str
    created_at: datetime
    metadata: dict | None = Field(
        default=None,
        validation_alias="message_metadata",
        description="扩展 JSON；库列 metadata",
    )
    turn_index: int = Field(
        description="会话内轮次序号；0 表示非 user 轮次占位（如 system）或历史占位"
    )
    reply_message_id: int | None = Field(
        default=None,
        description="assistant 行指向本轮 user 消息 id；user 行为空",
    )
    tokens: int | None = Field(
        default=None,
        description="user：正文估算 token；assistant：网关 usage 总 token",
    )


class RollingSummaryEnqueueOut(BaseModel):
    """手动触发滚动摘要入队结果。"""

    queued: bool = Field(description="是否已入队（后台异步执行）")


class ConversationDetailData(BaseModel):
    session: ConversationSessionOut
    messages: list[ConversationMessageOut]
    messages_meta: PageMeta = Field(
        description="消息分页：第 1 页为时间轴上最近 page_size 条（页内按时间正序）；第 2 页为更早的一批"
    )


class ConversationMessageExecutionOut(BaseModel):
    """单条助手消息 metadata 中的执行过程快照（与 ``AgentInvokeData`` 中字段形状对齐）。"""

    message_id: int = Field(ge=1)
    session_id: str = Field(min_length=1, max_length=64)
    agent_id: int = Field(ge=1, description="本条消息归属的 agent_entity.id")
    role: ConversationMessageRole
    process_trace: list[Any] | None = Field(
        default=None,
        description="多段模型 / 工具时间线；无则 null",
    )
    tool_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="工具调用与结果序列；可与 process_trace 并存",
    )
    thinking_text: str | None = Field(default=None, description="metadata.thinking_text")


class ConversationSessionListData(BaseModel):
    items: list[ConversationSessionOut]
    meta: PageMeta
