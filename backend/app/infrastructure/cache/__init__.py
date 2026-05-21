"""基于外部存储的缓存适配器（读穿、失效等）。"""

from app.infrastructure.cache.agent_memory import AgentMemoryCache, get_agent_memory_cache
from app.infrastructure.cache.history_turns import HistoryTurnsCache, get_history_turns_cache

__all__ = [
    "AgentMemoryCache",
    "HistoryTurnsCache",
    "get_agent_memory_cache",
    "get_history_turns_cache",
]
