"""链路追踪 ORM（``trace_runs`` / ``trace_spans`` / ``trace_span_logs``）。"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base, BigIntId

_TraceDateTime = DateTime(timezone=False).with_variant(mysql.DATETIME(fsp=6), "mysql")


class TraceRun(Base):
    """一次链路根记录：与 W3C trace_id 一一对应；不再单独存 request_id（与 trace_id 重复）。"""

    __tablename__ = "trace_runs"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trace_name: Mapped[str] = mapped_column(String(128))
    service_name: Mapped[str] = mapped_column(String(64), default="fastapi-backend")
    http_method: Mapped[str] = mapped_column(String(16), default="GET")
    http_path: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    started_at: Mapped[datetime] = mapped_column(_TraceDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(_TraceDateTime, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agent_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
    )


class TraceSpan(Base):
    __tablename__ = "trace_spans"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    span_id: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    span_name: Mapped[str] = mapped_column(String(128))
    span_type: Mapped[str] = mapped_column(String(32), default="custom")
    component: Mapped[str | None] = mapped_column(String(64), nullable=True)
    service_name: Mapped[str] = mapped_column(String(64), default="fastapi-backend")
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    started_at: Mapped[datetime] = mapped_column(_TraceDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(_TraceDateTime, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    is_slow: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    tags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
    )


class TraceSpanLog(Base):
    __tablename__ = "trace_span_logs"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    span_id: Mapped[str] = mapped_column(String(16), index=True)
    log_level: Mapped[str] = mapped_column(String(16), default="INFO")
    event_name: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(_TraceDateTime)
    seq_no: Mapped[int] = mapped_column(Integer, default=1)
