"""多轮对话持久化 ORM（``conversation_session`` / ``conversation_message``）。"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db import Base, BigIntId, LongText


class AgentConversationSession(Base):
    """对话会话元数据。"""

    __tablename__ = "conversation_session"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: 创建该会话的 Agent（列表默认归属）；工作台共享会话下子 Agent 消息见 ``conversation_message.agent_id``，列表按参与者并集查询（见 ``ConversationRepository.list_sessions_page``）。
    agent_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("agent_entity.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    #: 窗外历史的滚动摘要正文；与滑动窗口内原文正交（阶段 B）
    summary: Mapped[str | None] = mapped_column(LongText, nullable=True)
    #: 乐观锁：摘要成功落库后递增
    summary_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    #: 0=idle，1=pending，2=running（见 ``app.core.constants.conversation``）
    summary_job_status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("0"),
    )
    #: 最近一次成功落库摘要时，会话内 user 的 ``turn_index`` 锚点（用于增量触发）
    summary_anchor_turn_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    messages: Mapped[list["AgentConversationMessage"]] = relationship(
        "AgentConversationMessage",
        back_populates="session",
    )


class AgentConversationMessage(Base):
    """会话内单条消息。"""

    __tablename__ = "conversation_message"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversation_session.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: 本条消息归属 / 产生的 Agent（同 session 可跨多个 agent_id）
    agent_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("agent_entity.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(LongText, nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
    )
    message_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    #: 会话内轮次：user 从 1 递增；同轮 assistant 与 user 同序号；system/占位为 0
    turn_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    #: assistant 行指向本轮 user 消息 ``id``；user 行为 ``None``
    reply_message_id: Mapped[int | None] = mapped_column(
        BigIntId,
        ForeignKey("conversation_message.id", ondelete="SET NULL"),
        nullable=True,
    )
    #: user：正文 tiktoken 估算；assistant：网关 ``usage`` 总 token（多步为累加）
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    session: Mapped["AgentConversationSession"] = relationship(
        "AgentConversationSession",
        back_populates="messages",
    )
