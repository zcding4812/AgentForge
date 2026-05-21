from __future__ import annotations

from app.knowledge.adapters.chunk_methods import (
    split_knowledge_text_llama_index,
    split_text_by_length,
    split_with_priority_separators,
)


def test_split_with_separators_then_length() -> None:
    text = "aaa\n\nbbb\n\nccc"
    parts = split_with_priority_separators(
        text,
        separators=["\n\n", "\n"],
        chunk_size=8,
        chunk_overlap=2,
        hard_limit=False,
    )
    assert parts
    assert "".join(parts).replace("\n", "") == "aaabbbccc"


def test_length_matches_reference() -> None:
    t = "你好" * 30
    a = split_knowledge_text_llama_index(t, chunk_size=64, chunk_overlap=8, chunk_method="length")
    b = split_text_by_length(t, chunk_size=64, chunk_overlap=8)
    assert a == b


def test_length_matches_reference_varied_sizes() -> None:
    text = "你好" * 80 + "\n\n" + "x" * 50
    for size, overlap in ((512, 50), (64, 10), (40, 5)):
        a = split_knowledge_text_llama_index(
            text,
            chunk_size=size,
            chunk_overlap=overlap,
            chunk_method="length",
        )
        b = split_text_by_length(text, chunk_size=size, chunk_overlap=overlap)
        assert a == b


def test_empty_text_length() -> None:
    assert (
        split_knowledge_text_llama_index(
            "", chunk_size=100, chunk_overlap=10, chunk_method="length"
        )
        == []
    )


def test_semantic_legacy_falls_back_to_length() -> None:
    """历史 chunk_method=semantic 不再单独注册，按未知键回退 length。"""
    t = "ab"
    a = split_knowledge_text_llama_index(
        t, chunk_size=100, chunk_overlap=10, chunk_method="semantic"
    )
    b = split_text_by_length(t, chunk_size=100, chunk_overlap=10)
    assert a == b


def test_unknown_method_falls_back_to_length() -> None:
    t = "abc" * 40
    a = split_knowledge_text_llama_index(
        t, chunk_size=20, chunk_overlap=2, chunk_method="no_such_method_xyz"
    )
    b = split_text_by_length(t, chunk_size=20, chunk_overlap=2)
    assert a == b


def test_code_splitter_smoke() -> None:
    src = "def foo():\n    return 1\n\n" * 5
    chunks = split_knowledge_text_llama_index(
        src,
        chunk_size=512,
        chunk_overlap=20,
        chunk_method="code",
        source_filename="x.py",
    )
    assert chunks
    assert "def foo" in "".join(chunks)
