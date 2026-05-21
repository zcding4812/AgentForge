"""MongoDB ``knowledge_chunks`` 集合的读写与检索。

不继承 :class:`~app.repositories.base_repo.BaseRepository`（无 SQLAlchemy ORM 模型）；
由 ``MongoDatabaseManager.get_database`` 注入 ``AsyncIOMotorDatabase``。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.core.constants.knowledge import KEYWORD_SEARCH_SORT_POOL_CAP

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_chunks"


def _keyword_relevance_sort_key(text: str, query: str) -> tuple[int, int]:
    """关键词子路排序：匹配次数多、首次出现位置靠前优先（越小越靠前）。"""
    q = (query or "").strip()
    if not q:
        return (0, 0)
    pattern = re.escape(q)
    t = text or ""
    matches = list(re.finditer(pattern, t, flags=re.IGNORECASE))
    if not matches:
        return (0, 10**9)
    return (-len(matches), matches[0].start())


class ChunkRepository:
    """按 ``(kb_id, doc_id, content_version)`` 写入分片；支持简单文本检索。"""

    def __init__(self, database: AsyncIOMotorDatabase) -> None:
        self._coll: AsyncIOMotorCollection = database[COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """启动时建索引；失败仅记日志，不阻塞应用（可后续重试）。"""
        try:
            await self._coll.create_index(
                [("kb_id", 1), ("doc_id", 1), ("content_version", 1), ("chunk_index", 1)],
                unique=True,
                name="uk_kb_doc_version_chunk",
            )
            await self._coll.create_index([("kb_id", 1)], name="kb_id")
        except Exception as e:
            logger.warning("Mongo knowledge_chunks 索引创建失败（可重试）: %s", e)

    async def replace_document_chunks(
        self,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
        chunks: list[str],
    ) -> None:
        await self._coll.delete_many(
            {"kb_id": kb_id, "doc_id": doc_id, "content_version": content_version},
        )
        if not chunks:
            return
        now = datetime.now(UTC)
        docs: list[dict[str, Any]] = []
        for i, text in enumerate(chunks):
            docs.append(
                {
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "content_version": content_version,
                    "chunk_index": i,
                    "text": text,
                    "created_at": now,
                },
            )
        await self._coll.insert_many(docs)

    async def get_chunk_text(
        self,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
        chunk_index: int,
    ) -> str | None:
        """按版本与分片序号读取文本（向量命中后拼装摘要）。"""
        doc = await self._coll.find_one(
            {
                "kb_id": kb_id,
                "doc_id": doc_id,
                "content_version": content_version,
                "chunk_index": chunk_index,
            },
            projection={"text": 1},
        )
        if doc is None:
            return None
        return str(doc.get("text") or "")

    async def list_all_chunk_texts_for_version(
        self,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
        max_chunks: int = 50_000,
    ) -> list[str]:
        """按 ``chunk_index`` 升序返回某版本下全部分片文本（用于仅向量重建、不重 ingest）。"""
        filt: dict[str, Any] = {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "content_version": content_version,
        }
        lim = max(1, min(int(max_chunks), 50_000))
        cur = self._coll.find(filt).sort("chunk_index", 1).limit(lim)
        items = await cur.to_list(length=lim)
        return [str(x.get("text") or "") for x in items]

    async def search_chunks(
        self,
        kb_id: int,
        query: str,
        *,
        limit: int = 20,
        doc_id: int | None = None,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        if not q:
            return []
        escaped = re.escape(q)
        filt: dict[str, Any] = {
            "kb_id": kb_id,
            "text": {"$regex": escaped, "$options": "i"},
        }
        if doc_id is not None:
            filt["doc_id"] = int(doc_id)
        pool = max(int(limit), 1)
        fetch_cap = min(KEYWORD_SEARCH_SORT_POOL_CAP, max(pool * 25, pool))
        cur = self._coll.find(filt).limit(fetch_cap)
        raw = await cur.to_list(length=fetch_cap)
        raw.sort(key=lambda d: _keyword_relevance_sort_key(str(d.get("text", "")), q))
        return raw[:pool]

    async def count_chunks_for_document_version(
        self,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> int:
        """某 ``content_version`` 下 Mongo 分片条数（与 MySQL ``chunk_count`` 对账用）。"""
        filt: dict[str, Any] = {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "content_version": content_version,
        }
        return int(await self._coll.count_documents(filt))

    async def list_document_chunks(
        self,
        *,
        kb_id: int,
        doc_id: int,
        content_version: str,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """按 ``chunk_index`` 升序分页返回某版本下的分片文档。"""
        filt: dict[str, Any] = {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "content_version": content_version,
        }
        total = int(await self._coll.count_documents(filt))
        lim = max(1, min(int(limit), 200))
        sk = max(0, int(skip))
        cur = self._coll.find(filt).sort("chunk_index", 1).skip(sk).limit(lim)
        items = await cur.to_list(length=lim)
        return items, total

    async def delete_all_chunks_for_document(self, *, kb_id: int, doc_id: int) -> int:
        """删除该文档在 Mongo 中的全部分片（所有 content_version）。"""
        result = await self._coll.delete_many({"kb_id": kb_id, "doc_id": doc_id})
        return int(result.deleted_count)
