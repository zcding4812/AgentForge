from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.knowledge.chunk_config import (
    ChunkStrategyConfig,
    chunk_strategy_from_kb_row,
)


def test_chunk_strategy_length_params_validated() -> None:
    cfg = ChunkStrategyConfig(
        strategy_type="length",
        strategy_params={"separators": ["\n", " "], "hard_limit": True},
    )
    assert cfg.strategy_params["hard_limit"] is True
    assert cfg.strategy_params["separators"] == ["\n", " "]


def test_chunk_strategy_token_params() -> None:
    cfg = ChunkStrategyConfig(
        strategy_type="token",
        strategy_params={"tokenizer_name": "bert"},
    )
    assert cfg.strategy_params["tokenizer_name"] == "bert"


def test_chunk_strategy_rejects_bad_tokenizer() -> None:
    with pytest.raises(ValidationError):
        ChunkStrategyConfig(
            strategy_type="token",
            strategy_params={"tokenizer_name": "not_a_tokenizer"},
        )


def test_chunk_strategy_from_kb_row_fallback() -> None:
    class Row:
        chunk_method = "length"
        chunk_size = 128
        chunk_overlap = 10
        config_json = None

    cfg = chunk_strategy_from_kb_row(Row())
    assert cfg.strategy_type == "length"
    assert cfg.common.chunk_size == 128
    assert cfg.common.chunk_overlap == 10


def test_chunk_strategy_from_kb_row_reads_config() -> None:
    class Row:
        chunk_method = "length"
        chunk_size = 999
        chunk_overlap = 9
        config_json = {
            "chunk_strategy": {
                "version": 1,
                "common": {"chunk_size": 256, "chunk_overlap": 20, "trim_whitespace": False},
                "strategy_type": "length",
                "strategy_params": {"hard_limit": False},
            }
        }

    cfg = chunk_strategy_from_kb_row(Row())
    assert cfg.common.chunk_size == 256
    assert cfg.common.trim_whitespace is False
