"""Milvus 底层操作（``pymilvus.MilvusClient``）。

向量集合的 **LlamaIndex** 封装留在 ``knowledge/adapters``；此处仅放与框架无关的连接、过滤删除等。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pymilvus import MilvusClient

from app.config import Settings

__all__ = ["MilvusConfig", "MilvusInfra"]

_FILTER_ESCAPE_MAX_LEN = 128


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus 连接（强类型）；由 :meth:`from_settings` 从环境装配。"""

    uri: str
    token: str | None
    configured: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> MilvusConfig:
        uri = settings.milvus_uri.strip()
        token = (settings.milvus_token or "").strip() or None
        configured = bool(uri)
        return cls(uri=uri, token=token, configured=configured)


class MilvusInfra:
    """同步 ``MilvusClient`` 封装；删除表达式与知识库向量集合 schema 对齐。"""

    def __init__(self, config: MilvusConfig) -> None:
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> MilvusInfra:
        """与 :class:`MinioObjectStore` 相同，支持自 :class:`~app.config.Settings` 装配。"""
        return cls(MilvusConfig.from_settings(settings))

    @property
    def is_configured(self) -> bool:
        return self._config.configured

    @staticmethod
    def escape_filter_string(s: str) -> str:
        """转义 Milvus boolean expr 中字符串字面量内的 ``\\`` / ``"``。"""
        return re.sub(r'([\\"])', r"\\\1", s)[:_FILTER_ESCAPE_MAX_LEN]

    def _client(self) -> MilvusClient | None:
        if not self._config.configured:
            return None
        return MilvusClient(uri=self._config.uri, token=self._config.token)

    def check_connection(self) -> bool:
        """列出集合作为轻量探活；未配置或异常时返回 ``False``。"""
        if not self.is_configured:
            return False
        try:
            client = self._client()
            if client is None:
                return False
            client.list_collections()
            return True
        except Exception:
            return False

    def delete_by_kb_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> None:
        """按标量字段 ``kb_id`` / ``kb_doc_id`` / ``content_version`` 删除集合内命中行。

        与知识库向量集合 schema 一致；集合不存在或未启用 Milvus 时静默返回。
        """
        if not self.is_configured:
            return
        name = collection_name.strip()
        if not name:
            return
        client = self._client()
        if client is None:
            return
        if not client.has_collection(name):
            return
        cv = self.escape_filter_string(content_version)
        expr = f'kb_id == {int(kb_id)} && kb_doc_id == {int(doc_id)} && content_version == "{cv}"'
        client.delete(collection_name=name, filter=expr)

    def count_by_kb_doc_version(
        self,
        *,
        collection_name: str,
        kb_id: int,
        doc_id: int,
        content_version: str,
    ) -> int:
        """按与 :meth:`delete_by_kb_doc_version` 相同的标量过滤统计行数（上限内枚举）。

        集合不存在、查询失败时返回 ``-1``，表示**无法**可靠计数（调用方应视为未对齐、不短路）。
        """
        if not self.is_configured:
            return -1
        name = collection_name.strip()
        if not name:
            return -1
        client = self._client()
        if client is None:
            return -1
        try:
            if not client.has_collection(name):
                return -1
            cv = self.escape_filter_string(content_version)
            expr = (
                f'kb_id == {int(kb_id)} && kb_doc_id == {int(doc_id)} && content_version == "{cv}"'
            )
            # 与单文档分片上限对齐：超出部分视为无法廉价验证，返回 -1
            res = client.query(
                collection_name=name,
                filter=expr,
                output_fields=["kb_doc_id"],
                limit=65536,
            )
        except Exception:
            return -1
        if res is None:
            return -1
        return len(res)
