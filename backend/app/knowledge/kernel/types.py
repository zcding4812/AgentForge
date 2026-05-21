"""非 HTTP 的 DTO、枚举（与 ``app.schemas.knowledge`` 可映射）。"""

from __future__ import annotations

from typing import Any, TypedDict


class RetrievalQuery(TypedDict, total=False):
    """检索门面入参草案；落地后以 OpenAPI / schemas 为准。"""

    kb_id: int
    query: str
    top_k: int
    extra: dict[str, Any]


class RetrievalHit(TypedDict, total=False):
    """单条命中草案；含溯源字段。"""

    node_id: str
    snippet: str
    score: float
    doc_id: int
    chunk_index: int | None
    content_version: str
