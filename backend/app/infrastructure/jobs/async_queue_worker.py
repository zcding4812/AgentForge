"""进程内异步任务队列与消费者：仅 ``asyncio.Queue`` + 可注入 ``on_job``，与业务无关。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Final, Generic, TypeVar

logger = logging.getLogger(__name__)

JobT = TypeVar("JobT")


class AsyncQueueWorker(Generic[JobT]):
    """单消费者协程处理 ``JobT`` 队列；队列满时 ``put`` 阻塞（背压）。"""

    def __init__(
        self,
        *,
        on_job: Callable[[JobT], Awaitable[None]],
        max_queue_size: int = 1000,
        consumer_task_name: str = "async-queue-worker",
    ) -> None:
        self._on_job = on_job
        self._queue: asyncio.Queue[JobT] = asyncio.Queue(maxsize=max_queue_size)
        self._consumer_task: asyncio.Task[None] | None = None
        self._max_queue_size: Final[int] = max_queue_size
        self._consumer_task_name: Final[str] = consumer_task_name

    async def start(self) -> None:
        if self._consumer_task is not None:
            logger.warning("AsyncQueueWorker already started")
            return

        self._consumer_task = asyncio.create_task(
            self._consume_loop(),
            name=self._consumer_task_name,
        )
        logger.info(
            "AsyncQueueWorker started",
            extra={"max_queue_size": self._max_queue_size, "task_name": self._consumer_task_name},
        )

    async def stop(self, *, wait_timeout: float = 30.0) -> None:
        if self._consumer_task is None:
            return

        logger.info("Stopping AsyncQueueWorker...", extra={"task_name": self._consumer_task_name})
        try:
            await asyncio.wait_for(self._queue.join(), timeout=wait_timeout)
        except TimeoutError:
            logger.warning(
                "AsyncQueueWorker queue did not empty in time, forcing shutdown",
                extra={"remaining": self._queue.qsize(), "task_name": self._consumer_task_name},
            )

        self._consumer_task.cancel()
        try:
            await self._consumer_task
        except asyncio.CancelledError:
            pass

        self._consumer_task = None
        logger.info("AsyncQueueWorker stopped", extra={"task_name": self._consumer_task_name})

    async def enqueue(self, item: JobT) -> None:
        await self._queue.put(item)
        logger.debug(
            "Enqueued job",
            extra={"queue_size": self._queue.qsize(), "task_name": self._consumer_task_name},
        )

    async def _consume_loop(self) -> None:
        logger.info(
            "AsyncQueueWorker consumer loop started", extra={"task_name": self._consumer_task_name}
        )
        while True:
            item: JobT = await self._queue.get()
            try:
                await self._on_job(item)
                logger.debug(
                    "Processed job",
                    extra={
                        "queue_size": self._queue.qsize(),
                        "task_name": self._consumer_task_name,
                    },
                )
            except asyncio.CancelledError:
                logger.info(
                    "AsyncQueueWorker consumer loop cancelled",
                    extra={"task_name": self._consumer_task_name},
                )
                raise
            except Exception:
                logger.exception(
                    "Failed to process job",
                    extra={"task_name": self._consumer_task_name},
                )
            finally:
                self._queue.task_done()
