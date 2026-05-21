"""仓储基类：统一 ``run_read`` / ``run_write`` 与可选 ``db_manager`` 解析。

约定采用 **类方法** + 显式传入 ``SQLAlchemyDatabaseManager``，与服务层构造注入并存：
服务持有 ``self._db``，调用 ``SomeRepository.method(..., db_manager=self._db)``。

若省略 ``db_manager``（如链路落库、后台任务），则通过 :meth:`_resolve_db_manager` 从
``RequestContext`` 与 ``app.state.db_manager`` 解析；无可用上下文时抛出 ``RuntimeError``。

子类继承 :class:`BaseRepository` 并设置 ``model``，再实现具体表访问；**不在**路由层直接调用仓储。
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import ceil
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.infrastructure.db import SQLAlchemyDatabaseManager

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=DeclarativeBase)


@dataclass(slots=True, frozen=True)
class Page(Generic[ModelType]):
    items: list[ModelType]
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


class BaseRepository(Generic[ModelType]):
    """泛型基类；``model`` 由子类绑定为具体 ORM 类型。"""

    model: type[ModelType]

    @classmethod
    def _resolve_db_manager(
        cls, db_manager: SQLAlchemyDatabaseManager | None = None
    ) -> SQLAlchemyDatabaseManager:
        if db_manager is not None:
            return db_manager

        try:
            from app.core.context import RequestContext

            request = RequestContext.get_request()
            if request is not None:
                dm = getattr(request.app.state, "db_manager", None)
                if dm is not None:
                    return dm
        except (ImportError, RuntimeError):
            pass

        raise RuntimeError(
            "无法解析数据库管理器：\n"
            "1. 非Web请求场景请手动传入 db_manager 参数；\n"
            "2. Web请求场景请检查 RequestContext 和 app.state.db_manager 是否初始化"
        )

    @classmethod
    def _handle_db_exception(cls, exc: SQLAlchemyError, operation: str) -> None:
        logger.error(
            "数据库操作失败 | 模型: %s | 操作: %s | 错误: %s",
            cls.model.__name__,
            operation,
            str(exc),
            exc_info=True,
        )
        raise RuntimeError(f"操作{cls.model.__name__}失败：{str(exc)}") from exc

    @classmethod
    async def run_read(
        cls,
        handler: Callable[[AsyncSession], Awaitable[Any]],
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Any:
        try:
            return await cls._resolve_db_manager(db_manager).run_read(handler)
        except IntegrityError:
            raise
        except SQLAlchemyError as e:
            cls._handle_db_exception(e, "read")

    @classmethod
    async def run_write(
        cls,
        handler: Callable[[AsyncSession], Awaitable[Any]],
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Any:
        try:
            return await cls._resolve_db_manager(db_manager).run_write(handler)
        except IntegrityError:
            raise
        except SQLAlchemyError as e:
            cls._handle_db_exception(e, "write")

    @classmethod
    async def get_one_by(cls, **filters: Any) -> ModelType | None:
        async def _read(session: AsyncSession) -> ModelType | None:
            return (await session.scalars(select(cls.model).filter_by(**filters))).first()

        return await cls.run_read(_read)

    @classmethod
    async def exists_by(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **filters: Any,
    ) -> bool:
        async def _read(session: AsyncSession) -> bool:
            q = select(func.count()).select_from(cls.model).filter_by(**filters)
            return (await session.scalar(q) or 0) > 0

        return await cls.run_read(_read, db_manager)

    @classmethod
    async def paginate(
        cls,
        *,
        page: int = 1,
        page_size: int = 20,
        filters: dict[str, Any] | None = None,
        db_manager: SQLAlchemyDatabaseManager | None = None,
    ) -> Page[ModelType]:
        filters = filters or {}
        normalized_page = max(1, page)
        normalized_page_size = max(1, min(page_size, 100))

        async def _read(session: AsyncSession) -> tuple[list[ModelType], int]:
            count_query = select(func.count()).select_from(cls.model).filter_by(**filters)
            total = await session.scalar(count_query) or 0
            offset = (normalized_page - 1) * normalized_page_size
            data_query = (
                select(cls.model).filter_by(**filters).offset(offset).limit(normalized_page_size)
            )
            items = list((await session.scalars(data_query)).all())
            return items, total

        items, total = await cls.run_read(_read, db_manager)
        total_pages = ceil(total / normalized_page_size) if total > 0 else 0

        return Page(
            items=items,
            page=normalized_page,
            page_size=normalized_page_size,
            total=total,
            total_pages=total_pages,
        )

    @classmethod
    async def create(
        cls,
        *,
        db_manager: SQLAlchemyDatabaseManager | None = None,
        **kwargs: Any,
    ) -> ModelType:
        async def _write(session: AsyncSession) -> ModelType:
            instance = cls.model(**kwargs)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await cls.run_write(_write, db_manager)
