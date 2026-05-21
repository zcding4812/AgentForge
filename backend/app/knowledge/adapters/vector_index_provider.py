"""进程内默认 :class:`~app.knowledge.kernel.ports.VectorIndexPort` 单例（装配 ``milvus_vector_index``）。"""

from __future__ import annotations

import threading

from app.config import Settings, get_settings
from app.knowledge.adapters.milvus_vector_index import LlamaIndexMilvusVectorIndexAdapter
from app.knowledge.kernel.ports import VectorIndexPort

_lock = threading.Lock()
_singleton: VectorIndexPort | None = None


def get_vector_index_port(settings: Settings | None = None) -> VectorIndexPort:
    """返回 ``VectorIndexPort`` 实现（单进程一份）。"""
    global _singleton
    with _lock:
        if _singleton is None:
            s = settings if settings is not None else get_settings()
            _singleton = LlamaIndexMilvusVectorIndexAdapter(s)
        return _singleton
