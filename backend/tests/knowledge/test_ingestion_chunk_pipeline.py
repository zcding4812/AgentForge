"""Pipeline① 切块与 ``NodeParserFactory.split_text_for_config`` 字符串列表一致。"""

from __future__ import annotations

from app.domain.knowledge.chunk_config import ChunkStrategyConfig, CommonChunkConfig
from app.knowledge.adapters import (
    NodeParserFactory,
    chunk_strings_from_pipeline_nodes,
    run_ingestion_pipeline_for_text,
)


def _cfg_length() -> ChunkStrategyConfig:
    return ChunkStrategyConfig(
        common=CommonChunkConfig(chunk_size=120, chunk_overlap=12),
        strategy_type="length",
        strategy_params={},
    )


def test_pipeline_chunks_match_direct_split() -> None:
    cfg = _cfg_length()
    text = "段落一。\n\n段落二。\n\n" + "x" * 500
    direct = NodeParserFactory.split_text_for_config(text, cfg, source_filename="a.md")
    nodes = run_ingestion_pipeline_for_text(text, cfg, source_filename="a.md")
    via_nodes = chunk_strings_from_pipeline_nodes(nodes)
    assert direct == via_nodes


def test_pipeline_non_empty_nodes() -> None:
    cfg = _cfg_length()
    nodes = run_ingestion_pipeline_for_text("hello", cfg, source_filename=None)
    assert len(nodes) >= 1
    assert chunk_strings_from_pipeline_nodes(nodes)
