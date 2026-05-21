"""出站端口 Protocol。

**进程级连接**（MinIO、Mongo 等）在 ``app/infrastructure/``。
**向量索引（如 LlamaIndex Milvus）** 在 ``app/knowledge/adapters/`` 实现并注入本模块中的 ``VectorIndexPort``。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChunkStorePort(Protocol):
    """Mongo 等切块存储；读写须带 ``kb_id`` / ``doc_id`` / ``content_version`` 隔离。"""

    async def delete_by_doc_version(self, *, kb_id: int, doc_id: int, content_version: str) -> None:
        """删除该文档该版本下全部 chunk（重索引前清理）。"""
        ...

    async def upsert_chunks(
        self, *, kb_id: int, doc_id: int, content_version: str, chunks: list[dict[str, Any]]
    ) -> int:
        """写入切块；返回写入条数。"""
        ...


@runtime_checkable
class VectorIndexPort(Protocol):
    """Milvus 等向量索引；查询须带 ``kb_id`` 过滤。

    实现位于 ``knowledge/adapters/milvus_vector_index``，经 :func:`~app.knowledge.adapters.get_vector_index_port` 装配。
    """

    async def delete_by_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> None:
        """删除某文档某版本在集合中的全部向量行。"""

    async def insert_vectors(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        doc_id: int,
        content_version: str,
        vectors: list[list[float]],
        texts: list[str],
    ) -> None:
        """插入向量与对应分片原文（``len(texts) == len(vectors)``），供 Milvus ``text`` 列与混合检索展示。"""

    async def search_vectors(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> list[tuple[int, int, str, float]]:
        """ANN 检索，返回 ``(doc_id, chunk_index, content_version, distance)``。"""

    async def search_vector_nodes(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> list[tuple[Any, float]]:
        """与 :meth:`search_vectors` 同一查询；返回 ``(节点, distance)``。

        节点须带 ``metadata``（如 LlamaIndex ``TextNode`` 的 ``kb_doc_id`` / ``chunk_index`` / ``content_version``），
        供防腐层映射；第一元组类型为实现相关，在适配器内约定。
        """

    async def count_vectors_for_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> int:
        """统计集合内该 ``(kb_id, doc_id, content_version)`` 下向量行数。

        与 :meth:`delete_by_doc_version` 过滤键一致；无法查询时返回 ``-1``（调用方勿当作 0 条）。
        """
        ...
