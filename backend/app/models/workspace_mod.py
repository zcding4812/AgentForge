"""工作区命名空间 ORM：表 ``namespace``；Agent / 知识库通过 ``namespace_id`` 归属同一命名空间。

每个命名空间至多一条 ``agent_kind=workbench`` 的 Agent：由列 ``workbench_agent_id`` 指向该行；
亦可通过 ``agent_entity`` 按 ``namespace_id`` + ``agent_kind`` 查询，二者一致由应用层维护。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base, BigIntId

_SysDateTime = DateTime(timezone=False)


class WorkspaceNamespace(Base):
    """独立命名空间行；``slug`` 与对外 API / URL 中的 ``workspace_namespace`` 字符串一致。"""

    __tablename__ = "namespace"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    #: 该命名空间唯一的工作台 Agent（与 ``agent_entity`` 中同 ns 的 workbench 行应对齐）
    workbench_agent_id: Mapped[int | None] = mapped_column(
        BigIntId,
        ForeignKey("agent_entity.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        _SysDateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        _SysDateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}
