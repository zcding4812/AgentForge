"""``NodeParserFactory`` 与 ``split_knowledge_text_llama_index`` 行为一致。"""

from __future__ import annotations

from app.domain.knowledge.chunk_config import ChunkStrategyConfig, CommonChunkConfig
from app.knowledge.adapters.chunk_methods import (
    ChunkConfig,
    split_knowledge_text_by_strategy,
    split_knowledge_text_llama_index,
)
from app.knowledge.adapters.node_parser_factory import NodeParserFactory


def test_split_text_for_config_matches_direct_split() -> None:
    cfg = ChunkStrategyConfig(
        common=CommonChunkConfig(chunk_size=64, chunk_overlap=8),
        strategy_type="length",
        strategy_params={},
    )
    text = "你好" * 80
    a = NodeParserFactory.split_text_for_config(text, cfg, source_filename="x.txt")
    b = split_knowledge_text_llama_index(
        text,
        chunk_size=cfg.common.chunk_size,
        chunk_overlap=cfg.common.chunk_overlap,
        chunk_method=cfg.strategy_type,
        source_filename="x.txt",
        extra_params=cfg.runtime_extra_params(),
        trim_whitespace=cfg.common.trim_whitespace,
    )
    assert a == b
    assert split_knowledge_text_by_strategy is split_knowledge_text_llama_index


def test_from_chunk_config_creates_strategy() -> None:
    cfg = ChunkStrategyConfig(
        common=CommonChunkConfig(),
        strategy_type="length",
        strategy_params={},
    )
    strategy = NodeParserFactory.from_chunk_config(cfg)
    cc = ChunkConfig(
        chunk_size=cfg.common.chunk_size,
        chunk_overlap=cfg.common.chunk_overlap,
        source_filename=None,
        extra_params=cfg.runtime_extra_params(),
        trim_whitespace=cfg.common.trim_whitespace,
    )
    out = strategy.split("abc" * 100, cc)
    assert len(out) >= 1
