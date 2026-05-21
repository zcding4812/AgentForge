"""§4.0.1：LlamaIndex 读文件与 Pipeline 基础行为（不绑定内部类名升级细节）。"""

from __future__ import annotations

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.readers.file import DocxReader, PDFReader


def test_llama_index_file_readers_importable() -> None:
    assert PDFReader is not None
    assert DocxReader is not None
    assert IngestionPipeline is not None


def test_llama_index_schema_and_splitter() -> None:
    doc = Document(text="hello world")
    assert doc.text == "hello world"
    sp = SentenceSplitter(chunk_size=10, chunk_overlap=0)
    nodes = sp.get_nodes_from_documents([doc])
    assert len(nodes) >= 1
    n = TextNode(text="x", id_="id1")
    assert n.node_id == "id1"


def test_sentence_ingestion_pipeline_non_empty() -> None:
    text = "First sentence. Second sentence here."
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=80, chunk_overlap=10)],
    )
    nodes = pipeline.run(documents=[Document(text=text)])
    chunks = [(n.get_content() or "").strip() for n in nodes if (n.get_content() or "").strip()]
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)
