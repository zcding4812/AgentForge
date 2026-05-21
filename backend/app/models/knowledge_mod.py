"""知识库域 ORM：``knowledge_base``、``knowledge_document``。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db import Base, BigIntId

if TYPE_CHECKING:
    from app.models.workspace_mod import WorkspaceNamespace

_SysDateTime = DateTime(timezone=False)


class KnowledgeDocument(Base):
    """知识库内文档目录行（上传/解析状态）；正文与向量见派生存储。"""

    __tablename__ = "knowledge_document"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    #: 当前文档已持久化的分片数量（与 Mongo/Milvus 对齐后由索引任务更新；默认 0）
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    #: 内容版本（哈希或 profile 串）；与 Mongo/Milvus 幂等键 ``(kb_id, doc_id, content_version)`` 对齐
    content_version: Mapped[str] = mapped_column(
        String(128), nullable=False, default="", server_default=""
    )

    deleted_at: Mapped[datetime | None] = mapped_column(_SysDateTime, nullable=True, index=True)

    #: 文档级切块覆盖；均为空则 ingest 使用所属知识库 ``knowledge_base.chunk_*``
    chunk_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunk_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_overlap: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_separator: Mapped[str | None] = mapped_column(String(64), nullable=True)

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


class KnowledgeBase(Base):
    """知识库元数据（权威）；派生存储见设计说明 §3。"""

    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    namespace_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("namespace.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    storage_type: Mapped[str] = mapped_column(String(32), nullable=False, default="vector")
    retrieval_type: Mapped[str] = mapped_column(String(32), nullable=False, default="vector")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="empty", index=True)

    embedding_model_config_id: Mapped[int | None] = mapped_column(
        BigIntId,
        ForeignKey("sys_model.id", ondelete="SET NULL"),
        nullable=True,
    )
    milvus_collection: Mapped[str | None] = mapped_column(String(256), nullable=True)
    minio_prefix: Mapped[str | None] = mapped_column(String(512), nullable=True)

    chunk_method: Mapped[str] = mapped_column(String(32), nullable=False, default="length")
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    chunk_separator: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    deleted_at: Mapped[datetime | None] = mapped_column(_SysDateTime, nullable=True, index=True)

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
    workspace_namespace = association_proxy("namespace", "slug")

    __table_args__ = (
        UniqueConstraint("namespace_id", "slug", name="uk_knowledge_base_namespace_slug"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )


class KnowledgeTask(Base):
    """异步任务行：ingest / reindex / delete 等；与 Worker、幂等键对齐。"""

    __tablename__ = "knowledge_task"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    kb_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_id: Mapped[int | None] = mapped_column(
        BigIntId,
        ForeignKey("knowledge_document.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content_version: Mapped[str] = mapped_column(
        String(128), nullable=False, default="", server_default=""
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(_SysDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(_SysDateTime, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True, unique=True)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(_SysDateTime, nullable=True)

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
