"""外部工具：类表继承（基表 + HTTP/MCP 子表），避免 kind 特有字段在同一行堆 NULL。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants.ext_tool import ToolKind
from app.infrastructure.db import Base, BigIntId, LongText

_SysDateTime = DateTime(timezone=False)


def _tool_kind_values(enum_cls: type[ToolKind]) -> list[str]:
    return [e.value for e in enum_cls]


class RuntimeExternalTool(Base):
    """通用元数据、Schema、启用状态与乐观锁；具体协议字段在子表。"""

    __tablename__ = "runtime_external_tool"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(512), nullable=False)
    kind: Mapped[ToolKind] = mapped_column(
        SQLEnum(ToolKind, values_callable=_tool_kind_values, native_enum=False, length=16),
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="1")
    input_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    created_at: Mapped[datetime] = mapped_column(
        _SysDateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        _SysDateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("name", name="uk_runtime_external_tool_name"),
        Index("idx_runtime_external_tool_kind_enabled", "kind", "enabled"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    __mapper_args__: dict[str, Any] = {
        "polymorphic_on": kind,
        "version_id_col": version,  # type: ignore[assignment]
    }


class RuntimeHttpTool(RuntimeExternalTool):
    """HTTP 工具：请求参数仅在子表，无 MCP 列 NULL。"""

    __tablename__ = "runtime_http_tool"

    id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("runtime_external_tool.id", ondelete="CASCADE"),
        primary_key=True,
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False, server_default="GET")
    headers_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    request_body_template: Mapped[str | None] = mapped_column(
        "request_body_template", LongText, nullable=True
    )
    timeout_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default="15000")

    __mapper_args__: dict[str, Any] = {"polymorphic_identity": ToolKind.HTTP}


class RuntimeMcpTool(RuntimeExternalTool):
    """MCP 工具：结构化查询字段 + 扩展 JSON。"""

    __tablename__ = "runtime_mcp_tool"

    id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("runtime_external_tool.id", ondelete="CASCADE"),
        primary_key=True,
    )
    transport_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default="http")
    server_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    connection_config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tools_config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("idx_runtime_mcp_tool_transport", "transport_type"),)

    __mapper_args__: dict[str, Any] = {"polymorphic_identity": ToolKind.MCP}
