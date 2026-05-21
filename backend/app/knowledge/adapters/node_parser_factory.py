"""NodeParserFactory：切块策略唯一入口（与 :class:`~app.knowledge.adapters.document_factory.DocumentFactory` 配对）。

ingest 主路径应调用 :meth:`~NodeParserFactory.split_text_for_ingest`，避免在 ``services`` 内散落
``chunk_method`` 分支。策略实现委托 :func:`~app.knowledge.adapters.chunk_methods.split_knowledge_text_by_strategy`。
"""

from __future__ import annotations

from app.domain.knowledge.chunk_config import (
    ChunkStrategyConfig,
    effective_chunk_strategy_for_ingest,
)
from app.knowledge.adapters.chunk_methods import (
    BaseChunkStrategy,
    ChunkStrategyFactory,
    split_knowledge_text_by_strategy,
)
from app.models.knowledge_mod import KnowledgeBase, KnowledgeDocument


class NodeParserFactory:
    """由 ``ChunkStrategyConfig``、或 KB+文档行推导策略，并统一切块字符串出口。"""

    @staticmethod
    def from_chunk_config(cfg: ChunkStrategyConfig) -> BaseChunkStrategy:
        """返回与 ``cfg.strategy_type`` 对应的策略实例（无状态）。"""
        return ChunkStrategyFactory.create(cfg.strategy_type)

    @staticmethod
    def split_text_for_config(
        text: str,
        cfg: ChunkStrategyConfig,
        *,
        source_filename: str | None = None,
    ) -> list[str]:
        """按结构化策略切块（与 Pipeline① 中 ``KnowledgeChunkStrategyTransform`` 算法一致）。"""
        return split_knowledge_text_by_strategy(
            text,
            chunk_size=cfg.common.chunk_size,
            chunk_overlap=cfg.common.chunk_overlap,
            chunk_method=cfg.strategy_type,
            source_filename=source_filename,
            extra_params=cfg.runtime_extra_params(),
            trim_whitespace=cfg.common.trim_whitespace,
        )

    @staticmethod
    def split_text_for_ingest(
        text: str,
        *,
        kb: KnowledgeBase,
        doc: KnowledgeDocument,
    ) -> list[str]:
        """ingest 推荐入口：由 ``effective_chunk_strategy_for_ingest`` 得配置后再切块。"""
        cfg = effective_chunk_strategy_for_ingest(kb, doc)
        return NodeParserFactory.split_text_for_config(
            text,
            cfg,
            source_filename=doc.filename,
        )
