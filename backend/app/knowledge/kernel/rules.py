"""纯规则：幂等键、版本比较（无 I/O）。"""

from __future__ import annotations


def ingest_idempotency_key(*, kb_id: int, doc_id: int, content_version: str) -> str:
    """逻辑幂等键：与 MySQL 文档行、Mongo/Milvus 写入对齐。

    ``content_version`` 须与 ``knowledge_document.content_version`` 一致（哈希或 profile 串）。
    """
    v = (content_version or "").strip()
    return f"{kb_id}:{doc_id}:{v}"
