"""基于 ``llama_index.vector_stores.milvus.MilvusVectorStore`` 的 ``VectorIndexPort`` 实现。

标量字段：``kb_id``、``kb_doc_id``（对应业务 ``doc_id``，避免与 LlamaIndex 保留的 ``doc_id`` 元数据冲突）、
``chunk_index``、``content_version``。删除操作使用 ``pymilvus.MilvusClient.delete``（无需向量维度）。

.. note::

    若 Milvus 中仍为历史自建集合（``chunk_pk`` / 旧字段名），需**重建集合并重新写入向量**后本适配器方可工作。
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from typing import cast

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.milvus import IndexManagement, MilvusVectorStore
from pymilvus import DataType

from app.config import Settings
from app.core.constants.knowledge import default_milvus_collection_name
from app.infrastructure.milvus import MilvusInfra
from app.knowledge.kernel.exceptions import VectorDimensionMismatchError

# Milvus VARCHAR 常见上限；过长文本截断避免写入失败
_MILVUS_TEXT_FIELD_MAX_CHARS = 65535


class LlamaIndexMilvusVectorIndexAdapter:
    """LlamaIndex ``MilvusVectorStore`` + ``MilvusClient`` 删除表达式。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._milvus_infra = MilvusInfra.from_settings(settings)
        self._lock = threading.Lock()
        self._stores: dict[tuple[str, int], MilvusVectorStore] = {}

    def _get_store(self, collection_name: str, dim: int) -> MilvusVectorStore | None:
        if not self._settings.milvus_configured:
            return None
        name = collection_name.strip()
        if not name:
            return None
        key = (name, dim)
        token = (self._settings.milvus_token or "").strip()
        uri = self._settings.milvus_uri.strip()
        with self._lock:
            if key not in self._stores:
                self._stores[key] = MilvusVectorStore(
                    uri=uri,
                    token=token,
                    collection_name=name,
                    dim=dim,
                    overwrite=False,
                    upsert_mode=False,
                    similarity_metric="cosine",
                    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
                    doc_id_field="ref_doc_id",
                    scalar_field_names=[
                        "kb_id",
                        "kb_doc_id",
                        "chunk_index",
                        "content_version",
                    ],
                    scalar_field_types=[
                        DataType.INT64,
                        DataType.INT64,
                        DataType.INT64,
                        DataType.VARCHAR,
                    ],
                    use_async_client=False,
                    output_fields=[
                        "kb_id",
                        "kb_doc_id",
                        "chunk_index",
                        "content_version",
                        "text",
                    ],
                    index_config={
                        "field_name": "embedding",
                        "index_type": "IVF_FLAT",
                        "metric_type": "COSINE",
                        "params": {"nlist": 128},
                    },
                    search_config={"params": {"nprobe": 32}},
                )
            return self._stores[key]

    def _delete_sync(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> None:
        self._milvus_infra.delete_by_kb_doc_version(
            collection_name=collection_name,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=content_version,
        )

    async def delete_by_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> None:
        await asyncio.to_thread(
            self._delete_sync,
            collection_name=collection_name,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=content_version,
        )

    def _insert_sync(
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
        if not vectors:
            return
        if len(texts) != len(vectors):
            raise ValueError(
                f"texts 与 vectors 条数不一致：{len(texts)} != {len(vectors)}",
            )
        store = self._get_store(collection_name, dim)
        if store is None:
            return
        cv_short = content_version[:128]
        cv_h = hashlib.sha256(content_version.encode("utf-8")).hexdigest()[:20]
        n = len(vectors)
        nodes: list[TextNode] = []
        for i in range(n):
            chunk_pk = f"k{kb_id}_d{doc_id}_{cv_h}_{i}"[:159]
            raw = (texts[i] or "").strip()
            body = raw if raw else " "
            if len(body) > _MILVUS_TEXT_FIELD_MAX_CHARS:
                body = body[:_MILVUS_TEXT_FIELD_MAX_CHARS]
            node = TextNode(
                text=body,
                id_=chunk_pk,
                metadata={
                    "ref_doc_id": "",
                    "kb_id": kb_id,
                    "kb_doc_id": doc_id,
                    "chunk_index": i,
                    "content_version": cv_short,
                },
            )
            node.embedding = vectors[i]
            nodes.append(node)
        store.add(nodes, force_flush=True)

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
        if not vectors:
            return
        await asyncio.to_thread(
            self._insert_sync,
            collection_name=collection_name,
            dim=dim,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=content_version,
            vectors=vectors,
            texts=texts,
        )

    def _run_vector_query(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> tuple[list, list[float]]:
        """执行 Milvus 向量查询，返回 ``(nodes, similarities)``。"""
        if len(query_vector) != dim:
            raise VectorDimensionMismatchError(
                f"query 向量维度 {len(query_vector)} 与集合期望 dim={dim} 不一致",
            )
        store = self._get_store(collection_name, dim)
        if store is None:
            return [], []
        flist: list = [
            MetadataFilter(key="kb_id", value=kb_id, operator=FilterOperator.EQ),
        ]
        if doc_id is not None:
            flist.append(
                MetadataFilter(key="kb_doc_id", value=doc_id, operator=FilterOperator.EQ),
            )
        filters = MetadataFilters(filters=flist)
        q = VectorStoreQuery(
            query_embedding=query_vector,
            similarity_top_k=max(1, min(int(limit), 100)),
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
            output_fields=[
                "kb_id",
                "kb_doc_id",
                "chunk_index",
                "content_version",
                "text",
            ],
        )
        result = store.query(q)
        nodes = result.nodes or []
        sims = result.similarities or []
        return nodes, sims

    def _search_sync(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> list[tuple[int, int, str, float]]:
        nodes, sims = self._run_vector_query(
            collection_name=collection_name,
            dim=dim,
            kb_id=kb_id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )
        out: list[tuple[int, int, str, float]] = []
        for node, dist in zip(nodes, sims):
            md = node.metadata or {}
            did = int(md.get("kb_doc_id", 0))
            cidx = int(md.get("chunk_index", 0))
            cv = str(md.get("content_version") or "")
            out.append((did, cidx, cv, float(dist)))
        return out

    def _search_nodes_sync(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> list[tuple[TextNode, float]]:
        """ANN 命中为 ``(TextNode, distance)``，供 :mod:`retrieval` 与官方 Retriever 路径对齐。"""
        nodes, sims = self._run_vector_query(
            collection_name=collection_name,
            dim=dim,
            kb_id=kb_id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )
        out: list[tuple[TextNode, float]] = []
        for node, dist in zip(nodes, sims):
            out.append((cast(TextNode, node), float(dist)))
        return out

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
        return await asyncio.to_thread(
            self._search_sync,
            collection_name=collection_name,
            dim=dim,
            kb_id=kb_id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )

    async def search_vector_nodes(
        self,
        *,
        collection_name: str,
        dim: int,
        kb_id: int,
        query_vector: list[float],
        limit: int,
        doc_id: int | None,
    ) -> list[tuple[TextNode, float]]:
        """与 :meth:`search_vectors` 同一查询；返回 LlamaIndex 节点以便经防腐层映射。"""
        return await asyncio.to_thread(
            self._search_nodes_sync,
            collection_name=collection_name,
            dim=dim,
            kb_id=kb_id,
            query_vector=query_vector,
            limit=limit,
            doc_id=doc_id,
        )

    def _count_sync(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> int:
        name = (collection_name or "").strip() or default_milvus_collection_name(kb_id)
        return self._milvus_infra.count_by_kb_doc_version(
            collection_name=name,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=content_version,
        )

    async def count_vectors_for_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> int:
        if not self._settings.milvus_configured:
            return -1
        return await asyncio.to_thread(
            self._count_sync,
            collection_name=collection_name,
            kb_id=kb_id,
            doc_id=doc_id,
            content_version=content_version,
        )
