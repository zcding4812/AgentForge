"""系统模型与提供商注册 ORM（表 ``sys_model_provider`` / ``sys_model``）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db import Base, BigIntId

_SysDateTime = DateTime(timezone=False)


class SysModelProvider(Base):
    """模型提供商：一套密钥 + 格式 + 地址供其下所有模型复用。"""

    __tablename__ = "sys_model_provider"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    provider_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    provider_name: Mapped[str] = mapped_column(String(64), nullable=False)

    base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    api_format: Mapped[str] = mapped_column(String(32), nullable=False, default="openai")
    auth_type: Mapped[str | None] = mapped_column(String(32), nullable=True, default="api_key")

    api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_secret: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    create_time: Mapped[datetime] = mapped_column(
        _SysDateTime, nullable=False, server_default=func.now()
    )
    update_time: Mapped[datetime] = mapped_column(
        _SysDateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    models: Mapped[list[SysModel]] = relationship(
        "SysModel",
        back_populates="provider",
        cascade="all, delete-orphan",
    )


class SysModel(Base):
    """具体模型实例，挂在某一提供商下。"""

    __tablename__ = "sys_model"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(
        BigIntId,
        ForeignKey("sys_model_provider.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_code: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)

    endpoint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timeout: Mapped[int | None] = mapped_column(Integer, nullable=True, default=30)
    is_enabled: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)

    create_time: Mapped[datetime] = mapped_column(
        _SysDateTime, nullable=False, server_default=func.now()
    )
    update_time: Mapped[datetime] = mapped_column(
        _SysDateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    provider: Mapped[SysModelProvider] = relationship("SysModelProvider", back_populates="models")

    __table_args__ = (
        UniqueConstraint("provider_id", "model_code", name="uk_provider_model"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )
