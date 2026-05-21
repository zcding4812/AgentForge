"""知识库用例编排：可从本包根导入；实现见 :mod:`app.knowledge.facades`、:mod:`app.knowledge.application.indexing`、`app.services.knowledge.ingest`。

``IngestTaskHandler`` / ``enqueue_ingest_after_upload`` / ``process_document_ingest`` 经 :func:`__getattr__` 延迟从 ``app.services.knowledge.ingest`` 加载，避免与 ingest 模块对 ``indexing`` 的导入形成环。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from app.knowledge.adapters.document_text import (
    document_from_upload_bytes,
    extract_text_for_ingest,
)
from app.knowledge.application.indexing import (
    persist_extracted_text_as_chunks,
    persist_upload_bytes_as_chunks,
)
from app.knowledge.facades import KnowledgeRetrievalFacade

__all__ = [
    "KnowledgeRetrievalFacade",
    "document_from_upload_bytes",
    "extract_text_for_ingest",
    "persist_extracted_text_as_chunks",
    "persist_upload_bytes_as_chunks",
    "enqueue_ingest_after_upload",
    "process_document_ingest",
    "IngestTaskHandler",
]

_LAZY_INGEST = frozenset(
    {
        "enqueue_ingest_after_upload",
        "process_document_ingest",
        "IngestTaskHandler",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_INGEST:
        mod = import_module("app.services.knowledge.ingest")
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
