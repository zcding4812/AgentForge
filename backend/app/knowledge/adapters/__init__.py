"""知识库出站适配：切块、原文抽取、嵌入、检索。

**聚合入口**：``from app.knowledge.adapters import ...``；Pipeline① 全部在
:mod:`ingestion_chunk_pipeline`（含旧名 ``run_ingestion_pipeline_*`` 别名）。
"""

from __future__ import annotations

from app.infrastructure.http_client import aclose_embedding_http_client

from . import ingestion_chunk_pipeline as ingestion_chunk_pipeline
from .chunk_methods import ChunkStrategyFactory
from .document_factory import DocumentFactory
from .embedding_adapter import EmbeddingAdapter
from .ingestion_chunk_pipeline import (
    KnowledgeChunkStrategyTransform,
    chunk_strings_from_pipeline_nodes,
    run_chunk_pipeline_for_documents,
    run_chunk_pipeline_for_text,
    run_ingestion_pipeline_for_documents,
    run_ingestion_pipeline_for_text,
)
from .node_parser_factory import NodeParserFactory
from .retrieval_pipeline import (
    KnowledgeRetrieverBuilder,
    assemble_search_data,
    normalize_retrieval_type,
)
from .vector_index_provider import get_vector_index_port

__all__ = [
    "ChunkStrategyFactory",
    "DocumentFactory",
    "EmbeddingAdapter",
    "KnowledgeChunkStrategyTransform",
    "KnowledgeRetrieverBuilder",
    "NodeParserFactory",
    "aclose_embedding_http_client",
    "assemble_search_data",
    "chunk_strings_from_pipeline_nodes",
    "get_vector_index_port",
    "ingestion_chunk_pipeline",
    "normalize_retrieval_type",
    "run_chunk_pipeline_for_documents",
    "run_chunk_pipeline_for_text",
    "run_ingestion_pipeline_for_documents",
    "run_ingestion_pipeline_for_text",
]
