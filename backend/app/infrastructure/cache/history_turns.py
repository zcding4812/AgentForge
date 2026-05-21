"""会话已组装轮次（读穿）；序列化格式见 ``history_turn_slots_*``。"""

from __future__ import annotations

import json
import logging
from functools import cache

import redis.asyncio as redis

from app.agent.kernel.spec import HistoryTurnSlot
from app.domain.conversation import (
    history_turn_slots_from_json_str,
    history_turn_slots_to_json_str,
)
from app.infrastructure.cache.redis_ops import delete_key, delete_pattern, get_text, setex_text

logger = logging.getLogger(__name__)


class HistoryTurnsCache:
    """``HistoryTurnAssembler.assemble`` 结果（去重前）。"""

    KEY_PREFIX = "agents:conversation:"
    KEY_TAG = ":assembled_turns:cap"

    def __init__(self, *, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds

    @classmethod
    def _key(cls, session_id: str, max_history_rounds_cap: int) -> str:
        return f"{cls.KEY_PREFIX}{session_id}{cls.KEY_TAG}{int(max_history_rounds_cap)}"

    @classmethod
    def _legacy_key(cls, session_id: str) -> str:
        return f"{cls.KEY_PREFIX}{session_id}:assembled_turns"

    async def get(
        self,
        client: redis.Redis,
        session_id: str,
        max_history_rounds_cap: int,
    ) -> list[HistoryTurnSlot] | None:
        key = self._key(session_id, max_history_rounds_cap)
        s = await get_text(client, key)
        if s is None:
            return None
        try:
            return history_turn_slots_from_json_str(s)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("会话历史缓存损坏，忽略 | session_id=%s err=%s", session_id, e)
            await delete_key(client, key)
            return None

    async def set(
        self,
        client: redis.Redis,
        session_id: str,
        turns: list[HistoryTurnSlot],
        max_history_rounds_cap: int,
    ) -> None:
        payload = history_turn_slots_to_json_str(turns)
        await setex_text(client, self._key(session_id, max_history_rounds_cap), self._ttl, payload)

    async def invalidate(self, client: redis.Redis, session_id: str) -> None:
        await delete_key(client, self._legacy_key(session_id))
        await delete_pattern(client, f"{self.KEY_PREFIX}{session_id}{self.KEY_TAG}*")


@cache
def get_history_turns_cache() -> HistoryTurnsCache:
    return HistoryTurnsCache(ttl_seconds=300)
