import logging
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from sqlalchemy import BigInteger, Integer, Text, event, text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# ====================== 类型 & 分页模型 ======================


class BaseModelProtocol(Protocol):
    __table__: Any
    __primary_key__: Any


T = TypeVar("T", bound=BaseModelProtocol)


@dataclass(slots=True, frozen=True)
class Page(Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
    total_pages: int

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1


Base = declarative_base()

# ====================== 字段类型（多数据库兼容） ======================

LongText = Text().with_variant(LONGTEXT, "mysql")
BigIntId = BigInteger().with_variant(Integer(), "sqlite")

# ====================== 数据库 URL 规范化 ======================


def normalize_mysql_database_url(
    url: str,
    *,
    target: Literal["sync", "async"],
) -> str:
    """
    在 MySQL 连接串的**同步**（``mysql+pymysql``，Alembic / 同步脚本）与**异步**
    （``mysql+aiomysql``，``create_async_engine``）之间做规范化。

    - ``target="sync"``：将 ``mysql+aiomysql`` / ``mysql+asyncmy`` 转为 ``mysql+pymysql``；
      已是 ``mysql+pymysql`` 则原样返回。
    - ``target="async"``：将 ``mysql+pymysql`` 转为 ``mysql+aiomysql``；已为异步驱动则原样返回。

    ``target="async"`` 且非 ``mysql+pymysql`` / 异步 mysql 时会报错。
    """
    u = url.strip()
    if not u:
        raise ValueError("database_url is empty")

    if target == "sync":
        if u.startswith("mysql+aiomysql://"):
            return u.replace("mysql+aiomysql://", "mysql+pymysql://", 1)
        if u.startswith("mysql+asyncmy://"):
            return u.replace("mysql+asyncmy://", "mysql+pymysql://", 1)
        return u

    if "+aiomysql" in u or "+asyncmy" in u:
        return u
    if u.startswith("mysql+pymysql://"):
        return u.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
    raise ValueError(
        "DATABASE_URL 须为 mysql+pymysql://... 以便推导异步 URL；"
        f"当前为: {u.split('://', 1)[0] if '://' in u else u}"
    )


def normalize_database_url_async(url: str) -> str:
    """异步引擎 URL：``sqlite+aiosqlite://`` 原样返回；MySQL 经 ``normalize_mysql_database_url``。"""
    u = url.strip()
    if not u:
        raise ValueError("database_url is empty")
    if u.startswith("sqlite+aiosqlite://"):
        return u
    return normalize_mysql_database_url(u, target="async")


# ====================== SQLite 外键 ======================


def _is_sqlite_async_url(database_url: str) -> bool:
    return database_url.strip().lower().startswith("sqlite+aiosqlite://")


def _register_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(
        dbapi_connection: sqlite3.Connection,
        _connection_record: object,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ====================== 数据库管理器 ======================


class SQLAlchemyDatabaseManager:
    """单例异步引擎 + ``AsyncSession``（全站统一）；引擎与会话工厂挂在类属性上。"""

    _instance: "SQLAlchemyDatabaseManager | None" = None
    _engine: AsyncEngine | None = None
    _session_factory: async_sessionmaker[AsyncSession] | None = None

    DEFAULT_POOL_CONFIG: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
    }

    def __init__(self, database_url: str, pool_config: dict | None = None) -> None:
        cls = type(self)
        if cls._engine is not None:
            return

        final_pool = {**cls.DEFAULT_POOL_CONFIG, **(pool_config or {})}

        if _is_sqlite_async_url(database_url):
            cls._engine = create_async_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
            _register_sqlite_foreign_keys(cls._engine)
        else:
            cls._engine = create_async_engine(database_url, **final_pool)

        cls._session_factory = async_sessionmaker(
            cls._engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )

    @classmethod
    def get_instance(
        cls,
        database_url: str | None = None,
        pool_config: dict | None = None,
    ) -> "SQLAlchemyDatabaseManager":
        if cls._instance is None:
            if not database_url:
                raise RuntimeError("首次初始化必须传入 database_url")
            cls._instance = cls(database_url, pool_config)
        return cls._instance

    @property
    def engine(self) -> AsyncEngine:
        cls = type(self)
        if not cls._engine:
            raise RuntimeError("数据库引擎未初始化")
        return cls._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        cls = type(self)
        if not cls._session_factory:
            raise RuntimeError("数据库引擎未初始化")
        return cls._session_factory

    async def run_read(
        self,
        handler: Callable[[AsyncSession], Awaitable[Any]],
    ) -> Any:
        async with self.session_factory() as session:
            return await handler(session)

    async def run_write(
        self,
        handler: Callable[[AsyncSession], Awaitable[Any]],
    ) -> Any:
        async with self.session_factory() as session:
            async with session.begin():
                return await handler(session)

    async def check_connection(self) -> bool:
        """
        执行 ``SELECT 1`` 探活。

        延迟受 ``pool_pre_ping``、异步驱动调度、与同库 Trace 负载等影响。
        可通过 ``DB_POOL_PRE_PING=false`` 略降探活耗时（需自行承担连接失效风险）。
        """
        cls = type(self)
        if not cls._engine:
            logger.error("数据库引擎未初始化，无法检查连接（技术日志）")
            return False
        try:
            async with cls._engine.connect() as conn:
                await conn.scalar(text("SELECT 1"))
            logger.debug("数据库连接检查通过（技术日志）")
            return True
        except SQLAlchemyError as e:
            logger.error(f"数据库连接检查失败（技术异常）：{str(e)}", exc_info=True)
            return False

    @classmethod
    async def close(cls) -> None:
        if cls._engine:
            await cls._engine.dispose()
        cls._instance = None
        cls._engine = None
        cls._session_factory = None
        logger.info("数据库引擎已关闭（技术日志）")


DatabaseManager = SQLAlchemyDatabaseManager
