"""Agent 域 ORM：表 ``agent_entity``（入库配置与工作区 ``config_json``）。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db import Base, BigIntId

if TYPE_CHECKING:
    from app.models.workspace_mod import WorkspaceNamespace

_SysDateTime = DateTime(timezone=False)


class AgentEntity(Base):
    """用户可见的 Agent 配置资源（ORM 行）；与运行时编排/调用概念区分。"""

    __tablename__ = "agent_entity"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    namespace_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("namespace.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    agent_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active", index=True)

    sys_model_id: Mapped[int | None] = mapped_column(
        BigIntId,
        ForeignKey("sys_model.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    #: 第 1 类 System Prompt（与 ``PromptEngineeringBody.system_prompt`` 对齐）；可与 ``config_json`` 并存，保存时双写
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        _SysDateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        _SysDateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    namespace: Mapped[WorkspaceNamespace] = relationship(
        "WorkspaceNamespace",
        foreign_keys=[namespace_id],
    )
    #: 与 OpenAPI / 旧版字段对齐，等价于 ``namespace.slug``
    workspace_namespace = association_proxy("namespace", "slug")

    __table_args__ = (
        UniqueConstraint("namespace_id", "name", name="uk_agent_entity_namespace_name"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )
