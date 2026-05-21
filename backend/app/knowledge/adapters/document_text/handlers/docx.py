from __future__ import annotations

from llama_index.readers.file import DocxReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class DocxHandler:
    name = "docx"
    priority = 15

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or fn.endswith(".docx")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".docx") as path:
            docs = DocxReader().load_data(path)
            return join_llama_documents(docs)
