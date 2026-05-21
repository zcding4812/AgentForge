from __future__ import annotations

from llama_index.readers.file import EpubReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class EpubHandler:
    name = "epub"
    priority = 30

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in ("application/epub+zip",) or fn.endswith(".epub")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".epub") as path:
            docs = EpubReader().load_data(path)
            return join_llama_documents(docs)
