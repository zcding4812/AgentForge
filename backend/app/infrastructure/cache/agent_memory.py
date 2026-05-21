"""Agent 记忆参数与知识库绑定（``config_json`` 子集）读穿缓存。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from functools import cache

import redis.asyncio as redis

from app.domain.agent import AgentMemorySettings, AgentMemorySettingsParser
from app.domain.agent.knowledge_binding import (
    AgentKnowledgeBindingParser,
    AgentKnowledgeBindingSettings,
    knowledge_binding_from_cache_payload,
    knowledge_binding_settings_as_json,
)
from app.infrastructure.cache.redis_ops import delete_key, get_text, setex_text

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2


class AgentMemoryCache:
    """``AgentMemorySettings`` + ``AgentKnowledgeBindingSettings`` 按 ``agent_entity.id`` 缓存。"""

    KEY_PREFIX = "agents:agent:"
    KEY_SUFFIX = ":memory_settings"

    def __init__(self, *, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds

    @classmethod
    def _cache_key(cls, agent_id: int) -> str:
        return f"{cls.KEY_PREFIX}{int(agent_id)}{cls.KEY_SUFFIX}"

    async def get(
        self,
        client: redis.Redis,
        agent_id: int,
    ) -> tuple[AgentMemorySettings, AgentKnowledgeBindingSettings] | None:
        key = self._cache_key(agent_id)
        s = await get_text(client, key)
        if s is None:
            return None
        try:
            data = json.loads(s)
            if not isinstance(data, dict):
                raise ValueError("payload not object")
            mem_parser = AgentMemorySettingsParser()
            kb_parser = AgentKnowledgeBindingParser()
            if data.get("v") == _CACHE_VERSION:
                mem_block = data.get("memory")
                if not isinstance(mem_block, dict):
                    raise ValueError("v2 missing memory")
                kb_block = data.get("knowledge")
                if not isinstance(kb_block, dict):
                    knowledge = kb_parser.parse(None)
                else:
                    knowledge = knowledge_binding_from_cache_payload(kb_block)
                return mem_parser.parse({"memory": mem_block}), knowledge
            # 旧版：整段 JSON 即 memory 的扁平字段
            return mem_parser.parse({"memory": data}), kb_parser.parse(None)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Agent 配置缓存损坏，忽略 | agent_id=%s err=%s", agent_id, e)
            await delete_key(client, key)
            return None

    async def set(
        self,
        client: redis.Redis,
        agent_id: int,
        settings: AgentMemorySettings,
        knowledge: AgentKnowledgeBindingSettings,
    ) -> None:
        payload = json.dumps(
            {
                "v": _CACHE_VERSION,
                "memory": asdict(settings),
                "knowledge": knowledge_binding_settings_as_json(knowledge),
            },
            separators=(",", ":"),
        )
        await setex_text(client, self._cache_key(agent_id), self._ttl, payload)

    async def invalidate(self, client: redis.Redis, agent_id: int) -> None:
        await delete_key(client, self._cache_key(agent_id))


@cache
def get_agent_memory_cache() -> AgentMemoryCache:
    return AgentMemoryCache(ttl_seconds=300)
