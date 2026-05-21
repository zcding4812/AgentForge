"""进程内滚动摘要 Worker 与入队（无 services 依赖）。"""

import asyncio

from app.infrastructure.jobs.rolling_summary_broker import (
    RollingSummaryJob,
    RollingSummaryWorker,
    enqueue_rolling_summary,
    set_rolling_summary_worker,
)


def test_enqueue_without_worker_returns_false() -> None:
    async def _run() -> None:
        set_rolling_summary_worker(None)
        ok = await enqueue_rolling_summary("sid-1", "manual")
        assert ok is False

    asyncio.run(_run())


def test_worker_enqueue_calls_on_job() -> None:
    calls: list[tuple[str, str]] = []

    async def on_job(job: RollingSummaryJob) -> None:
        calls.append((job["session_id"], job["reason"]))

    worker = RollingSummaryWorker(on_job=on_job, max_queue_size=8)
    set_rolling_summary_worker(worker)
    try:

        async def _run() -> None:
            await worker.start()
            await worker.enqueue({"session_id": "sid-1", "reason": "manual"})
            await asyncio.sleep(0.05)
            await worker.stop(wait_timeout=2)

        asyncio.run(_run())
    finally:
        set_rolling_summary_worker(None)

    assert calls == [("sid-1", "manual")]


def test_enqueue_rolling_summary_helper_with_registered_worker() -> None:
    async def on_job(job: RollingSummaryJob) -> None:
        _ = job

    worker = RollingSummaryWorker(on_job=on_job, max_queue_size=8)

    async def _run() -> None:
        set_rolling_summary_worker(worker)
        try:
            await worker.start()
            ok = await enqueue_rolling_summary("sid-x", "urgent")
            assert ok is True
            await worker.stop(wait_timeout=2)
        finally:
            set_rolling_summary_worker(None)

    asyncio.run(_run())
