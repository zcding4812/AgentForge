from __future__ import annotations

from llama_index.readers.file import XMLReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class XmlHandler:
    name = "xml"
    priority = 35

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in (
            "application/xml",
            "text/xml",
        ) or fn.endswith(".xml")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".xml") as path:
            docs = XMLReader(tree_level_split=0).load_data(path)
            return join_llama_documents(docs)
