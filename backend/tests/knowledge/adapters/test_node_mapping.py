"""§4.0：节点 → DTO 映射契约。"""

from __future__ import annotations

from llama_index.core.schema import TextNode

from app.knowledge.adapters.retrieval import (
    META_CHUNK_INDEX,
    META_CONTENT_VERSION,
    META_KB_DOC_ID,
    text_node_to_knowledge_search_hit,
    text_node_to_retrieval_hit,
    text_nodes_to_knowledge_search_hits,
    vector_node_pairs_to_ann_tuples,
)


def test_text_node_to_knowledge_search_hit_uses_metadata() -> None:
    n = TextNode(
        text="hello world " * 50,
        id_="pk1",
        metadata={
            META_KB_DOC_ID: 42,
            META_CHUNK_INDEX: 3,
            META_CONTENT_VERSION: "v1",
        },
    )
    hit = text_node_to_knowledge_search_hit(n, match_type="vector")
    assert hit.doc_id == 42
    assert hit.chunk_index == 3
    assert hit.text_snippet is not None
    assert "…" in hit.text_snippet
    assert hit.match_type == "vector"


def test_text_node_to_retrieval_hit_includes_optional_fields() -> None:
    n = TextNode(
        text="x",
        id_="n1",
        metadata={META_KB_DOC_ID: 7, META_CHUNK_INDEX: 0, META_CONTENT_VERSION: "cv"},
    )
    rh = text_node_to_retrieval_hit(n, score=0.91)
    assert rh["node_id"] == "n1"
    assert rh["score"] == 0.91
    assert rh.get("doc_id") == 7
    assert rh.get("chunk_index") == 0
    assert rh.get("content_version") == "cv"


def test_vector_node_pairs_to_ann_tuples() -> None:
    n = TextNode(
        text=" ",
        id_="id1",
        metadata={"kb_doc_id": 9, "chunk_index": 2, "content_version": "cv1"},
    )
    rows = vector_node_pairs_to_ann_tuples([(n, 0.12)])
    assert rows == [(9, 2, "cv1", 0.12)]


def test_text_nodes_batch_filenames() -> None:
    nodes = [
        TextNode(
            text="a",
            id_="1",
            metadata={META_KB_DOC_ID: 10},
        ),
    ]
    hits = text_nodes_to_knowledge_search_hits(
        nodes,
        filenames_by_doc_id={10: "a.pdf"},
        match_type="vector",
    )
    assert len(hits) == 1
    assert hits[0].filename == "a.pdf"
