"""工作台流式图事件与子 Agent 侧道 progress 的合并：主 / 侧双生产者、单出口队列。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

__all__ = ["merge_workbench_stream"]

_MAIN_DONE = object()


async def merge_workbench_stream(
    main: AsyncIterator[dict[str, Any]],
    side: asyncio.Queue[dict[str, Any]] | None,
) -> AsyncIterator[dict[str, Any]]:
    if side is None:
        async for ev in main:
            yield ev
        return
    out: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()

    async def pump_main() -> None:
        try:
            async for ev in main:
                await out.put(ev)
        finally:
            await out.put(_MAIN_DONE)

    async def pump_side() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                nxt = await side.get()
                await out.put(nxt)

    t_m = asyncio.create_task(pump_main())
    t_s = asyncio.create_task(pump_side())
    try:
        while True:
            x = await out.get()
            if x is _MAIN_DONE:
                t_s.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t_s
                with contextlib.suppress(asyncio.QueueEmpty):
                    while True:
                        o = out.get_nowait()
                        if o is not _MAIN_DONE and isinstance(o, dict):
                            yield o
                with contextlib.suppress(asyncio.QueueEmpty):
                    while True:
                        yield side.get_nowait()
                return
            if isinstance(x, dict):
                yield x
    finally:
        t_m.cancel()
        t_s.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t_m
        with contextlib.suppress(asyncio.CancelledError):
            await t_s
