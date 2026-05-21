"""Pipeline①：官方 ``IngestionPipeline`` 仅负责 **bytes/文本 → ``Document`` → 切块 ``TextNode``**。

与 Pipeline②（嵌入 + Milvus，见 :mod:`ingestion_vector_pipeline`）分离；**任务表 / MySQL 权威** 在 worker 外推进。

切块仍委托 :class:`~app.knowledge.adapters.node_parser_factory.NodeParserFactory`。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from pydantic import ConfigDict, Field

from app.domain.knowledge.chunk_config import ChunkStrategyConfig
from app.knowledge.adapters.document_factory import DocumentFactory
from app.knowledge.adapters.node_parser_factory import NodeParserFactory


class KnowledgeChunkStrategyTransform(TransformComponent):
    """单步变换：``Document``/``BaseNode`` 全文 → 与策略一致的 ``TextNode`` 列表。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_config: ChunkStrategyConfig
    source_filename: str | None = Field(default=None)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        out: list[BaseNode] = []
        for node in nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            splits = NodeParserFactory.split_text_for_config(
                text,
                self.chunk_config,
                source_filename=self.source_filename,
            )
            out.extend(
                build_nodes_from_splits(
                    text_splits=splits,
                    document=node,
                    ref_doc=node,
                ),
            )
        return out


def run_chunk_pipeline_for_documents(
    documents: list[Document],
    chunk_config: ChunkStrategyConfig,
    *,
    source_filename: str | None = None,
) -> list[BaseNode]:
    """Pipeline①：``Document`` 列表 → ``TextNode`` 等子节点。"""
    pipeline = IngestionPipeline(
        transformations=[
            KnowledgeChunkStrategyTransform(
                chunk_config=chunk_config,
                source_filename=source_filename,
            ),
        ],
    )
    return list(pipeline.run(documents=documents))


def run_chunk_pipeline_for_text(
    text: str,
    chunk_config: ChunkStrategyConfig,
    *,
    source_filename: str | None = None,
) -> list[BaseNode]:
    """Pipeline①：纯文本 → ``Document`` → 切块。"""
    doc = DocumentFactory.from_extracted_text(
        text,
        metadata={"source_filename": (source_filename or "")},
    )
    return run_chunk_pipeline_for_documents(
        [doc],
        chunk_config,
        source_filename=source_filename,
    )


def chunk_strings_from_pipeline_nodes(nodes: Sequence[BaseNode]) -> list[str]:
    """切块字符串列表（顺序与条数与策略 split 一致）。"""
    return [(n.get_content(metadata_mode=MetadataMode.NONE) or "") for n in nodes]


# 兼容旧名（单测 / 外部引用）
run_ingestion_pipeline_for_documents = run_chunk_pipeline_for_documents
run_ingestion_pipeline_for_text = run_chunk_pipeline_for_text
