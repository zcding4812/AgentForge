import time

from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.schemas.system import HealthData

_START_MONOTONIC = time.monotonic()


class SystemService:
    """系统健康检查；构造注入 ``db_manager`` 用于数据库探活。"""

    def __init__(self, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._db = db_manager

    async def _check_database_connection(self) -> bool:
        """探活数据库连接。"""
        return await self._db.check_connection()

    async def get_health_data(self) -> HealthData:
        """聚合 DB 状态与进程 uptime（不参与链路追踪，避免探活污染 trace 存储）。"""
        db_ok = await self._check_database_connection()
        system_status = "ok" if db_ok else "degraded"
        return HealthData(
            status=system_status,
            uptime_seconds=round(time.monotonic() - _START_MONOTONIC, 3),
            database_status="ok" if db_ok else "error",
        )
