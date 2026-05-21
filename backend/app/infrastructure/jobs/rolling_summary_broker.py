"""滚动摘要任务：在通用 :class:`~app.infrastructure.jobs.async_queue_worker.AsyncQueueWorker` 上的类型与进程内注册表。

队列/消费循环本身见 :mod:`app.infrastructure.jobs.async_queue_worker`。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from app.infrastructure.jobs.async_queue_worker import AsyncQueueWorker

logger = logging.getLogger(__name__)

ReasonLiteral = Literal["threshold", "urgent", "manual"]


class RollingSummaryJob(TypedDict, total=True):
    session_id: str
    reason: ReasonLiteral


RollingSummaryJobHandler = Callable[[RollingSummaryJob], Awaitable[None]]

RollingSummaryWorker = AsyncQueueWorker[RollingSummaryJob]

_worker: RollingSummaryWorker | None = None


def set_rolling_summary_worker(worker: RollingSummaryWorker | None) -> None:
    """由 ``lifespan`` 注册/清空，供 :func:`enqueue_rolling_summary` 解析。"""
    global _worker
    _worker = worker


def get_rolling_summary_worker() -> RollingSummaryWorker | None:
    return _worker


async def enqueue_rolling_summary(session_id: str, reason: ReasonLiteral) -> bool:
    """入队一条滚动摘要任务；若 Worker 未启动或 ``session_id`` 无效则返回 ``False``。"""
    sid = session_id.strip()
    if not sid:
        return False
    w = _worker
    if w is None:
        logger.warning(
            "rolling_summary enqueue skipped: worker not registered",
            extra={
                "event": "rolling_summary.worker_unavailable",
                "session_id": sid,
                "trigger_reason": reason,
            },
        )
        return False
    job: RollingSummaryJob = {"session_id": sid, "reason": reason}
    await w.enqueue(job)
    logger.debug(
        "Enqueued rolling summary job",
        extra={"session_id": sid, "reason": reason, "queue_registered": True},
    )
    return True
