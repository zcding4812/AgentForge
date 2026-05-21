"""系统监控看板：健康状态、中间件探活、对话 assistant token 按日统计。"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from urllib.parse import urlparse, urlunparse

from fastapi import Request

from app.config import Settings, get_settings
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.infrastructure.milvus import MilvusInfra
from app.infrastructure.minio import MinioObjectStore
from app.infrastructure.mongo import MongoDatabaseManager
from app.infrastructure.redis import RedisManager
from app.repositories.conversation_repo import ConversationRepository
from app.schemas.system import (
    HealthData,
    InfraComponentStatus,
    MonitorDashboardData,
    TokenUsageDayPoint,
)
from app.services.system_svc import SystemService

logger = logging.getLogger(__name__)


def redact_url_for_display(url: str, *, max_len: int = 512) -> str:
    """将 ``user:password@`` 中的 ``password`` 替换为 ``***``；SQLite 内存库做简短展示（看板用）。"""
    s = url.strip()
    if not s:
        return ""
    low = s.lower()
    if low.startswith("sqlite"):
        if ":memory:" in low:
            return "sqlite :memory:"
        return s[:max_len]

    parsed = urlparse(s)
    if not parsed.scheme:
        return s[:max_len]

    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""

    if parsed.password:
        user_part = parsed.username or ""
        auth = f"{user_part}:***@" if user_part else ":***@"
    elif parsed.username:
        auth = f"{parsed.username}@"
    else:
        auth = ""

    netloc = f"{auth}{host}{port}"
    path = parsed.path or ""
    rebuilt = urlunparse((parsed.scheme, netloc, path, "", parsed.query, ""))
    return rebuilt[:max_len]


class MonitorService:
    def __init__(self, system: SystemService, db_manager: SQLAlchemyDatabaseManager) -> None:
        self._system = system
        self._db = db_manager

    async def get_dashboard(
        self,
        request: Request,
        *,
        token_days: int,
        settings: Settings | None = None,
    ) -> MonitorDashboardData:
        cfg = settings or get_settings()
        health: HealthData = await self._system.get_health_data()

        mysql_ok = health.database_status == "ok"
        infra: list[InfraComponentStatus] = [
            InfraComponentStatus(
                key="mysql",
                label="主数据库",
                status="ok" if mysql_ok else "error",
                detail=None if mysql_ok else "连接探活失败",
                address=redact_url_for_display(cfg.database_url) or None,
            ),
        ]

        if not cfg.redis_enabled:
            infra.append(
                InfraComponentStatus(
                    key="redis",
                    label="Redis",
                    status="disabled",
                    detail="未配置 REDIS_URL",
                    address=None,
                )
            )
        else:
            redis_ok = await RedisManager.ping()
            infra.append(
                InfraComponentStatus(
                    key="redis",
                    label="Redis",
                    status="ok" if redis_ok else "error",
                    detail=None if redis_ok else "PING 失败",
                    address=redact_url_for_display(cfg.redis_url) or None,
                )
            )

        try:
            mongo = MongoDatabaseManager.get_instance()
            mongo_ok = await mongo.check_connection()
        except Exception:
            logger.exception("Mongo 探活异常")
            mongo_ok = False
        infra.append(
            InfraComponentStatus(
                key="mongodb",
                label="MongoDB",
                status="ok" if mongo_ok else "error",
                detail=None if mongo_ok else "ping 失败",
                address=redact_url_for_display(cfg.mongodb_url) or None,
            )
        )

        minio_store: MinioObjectStore | None = getattr(
            request.app.state,
            "minio_object_store",
            None,
        )
        if not cfg.minio_configured:
            infra.append(
                InfraComponentStatus(
                    key="minio",
                    label="MinIO",
                    status="disabled",
                    detail="未配置对象存储",
                    address=None,
                )
            )
        else:
            minio_ok = bool(minio_store) and await minio_store.head_bucket_ok()
            ep = cfg.minio_endpoint.strip()
            minio_addr = ep if ep.startswith(("http://", "https://")) else f"http://{ep}"
            bucket = cfg.minio_bucket_knowledge.strip() or "knowledge"
            infra.append(
                InfraComponentStatus(
                    key="minio",
                    label="MinIO",
                    status="ok" if minio_ok else "error",
                    detail=None if minio_ok else "head_bucket 失败",
                    address=f"{minio_addr} · 桶 {bucket}",
                )
            )

        if not cfg.milvus_configured:
            infra.append(
                InfraComponentStatus(
                    key="milvus",
                    label="Milvus",
                    status="disabled",
                    detail="未配置 MILVUS_URI",
                    address=None,
                )
            )
        else:
            milvus_ok = await asyncio.to_thread(
                MilvusInfra.from_settings(cfg).check_connection,
            )
            infra.append(
                InfraComponentStatus(
                    key="milvus",
                    label="Milvus",
                    status="ok" if milvus_ok else "error",
                    detail=None if milvus_ok else "连接失败",
                    address=redact_url_for_display(cfg.milvus_uri) or None,
                )
            )

        days = max(1, min(token_days, 90))
        today = date.today()
        start_day = today - timedelta(days=days - 1)

        rows = await ConversationRepository.sum_tokens_by_role_by_day(
            since_day=start_day,
            db_manager=self._db,
        )
        by_day: dict[date, tuple[int, int]] = {d: (u, a) for d, u, a in rows}

        token_series: list[TokenUsageDayPoint] = []
        grand = 0
        cur = start_day
        while cur <= today:
            u_tok, a_tok = by_day.get(cur, (0, 0))
            day_total = int(u_tok) + int(a_tok)
            grand += day_total
            token_series.append(
                TokenUsageDayPoint(
                    date=cur.isoformat(),
                    user_tokens=int(u_tok),
                    assistant_tokens=int(a_tok),
                    total_tokens=day_total,
                )
            )
            cur += timedelta(days=1)

        return MonitorDashboardData(
            health=health,
            infrastructure=infra,
            token_usage=token_series,
            token_usage_grand_total=grand,
        )
