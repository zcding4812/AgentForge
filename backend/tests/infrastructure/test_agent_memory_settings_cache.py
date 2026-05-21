"""``AgentMemoryCache`` JSON 与 Redis 契约。"""

import asyncio
import json
from unittest.mock import AsyncMock

from app.domain.agent import AgentMemorySettingsParser
from app.domain.agent.knowledge_binding import AgentKnowledgeBindingParser
from app.infrastructure.cache.agent_memory import AgentMemoryCache


def _run(coro):  # noqa: ANN001
    return asyncio.run(coro)


def test_cache_key_includes_agent_id() -> None:
    assert AgentMemoryCache._cache_key(7) == "agents:agent:7:memory_settings"


def test_get_set_roundtrip_v2() -> None:
    cache = AgentMemoryCache(ttl_seconds=60)
    client = AsyncMock()
    settings = AgentMemorySettingsParser().parse(
        {"memory": {"max_history_rounds_cap": 10, "allow_client_chat_history": True}},
    )
    knowledge = AgentKnowledgeBindingParser().parse(
        {
            "knowledge": {
                "enabled": True,
                "knowledge_base_ids": [1, 2],
                "top_k": 5,
                "show_sources": False,
            },
        },
    )
    _run(cache.set(client, 42, settings, knowledge))
    client.setex.assert_called_once()
    key, ttl, raw = client.setex.call_args[0]
    assert key == "agents:agent:42:memory_settings"
    assert ttl == 60
    parsed = json.loads(raw)
    assert parsed["v"] == 2
    assert parsed["memory"]["max_history_rounds_cap"] == 10
    assert parsed["knowledge"]["enabled"] is True
    assert parsed["knowledge"]["knowledge_base_ids"] == [1, 2]
    assert parsed["knowledge"]["show_sources"] is False

    client.get = AsyncMock(return_value=raw if isinstance(raw, str) else raw.encode())
    got_mem, got_kb = _run(cache.get(client, 42))
    assert got_mem == settings
    assert got_kb.enabled is True
    assert got_kb.knowledge_base_ids == (1, 2)


def test_get_legacy_flat_memory() -> None:
    cache = AgentMemoryCache(ttl_seconds=60)
    client = AsyncMock()
    legacy = {"max_history_rounds_cap": 11, "allow_client_chat_history": False}
    raw = json.dumps(legacy)
    client.get = AsyncMock(return_value=raw)
    got_mem, got_kb = _run(cache.get(client, 1))
    assert got_mem.max_history_rounds_cap == 11
    assert got_kb.enabled is False
