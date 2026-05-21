from __future__ import annotations

from llama_index.readers.file import PDFReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class PdfHandler:
    name = "pdf"
    priority = 10

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime == "application/pdf" or fn.endswith(".pdf")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".pdf") as path:
            docs = PDFReader(return_full_document=True).load_data(path)
            return join_llama_documents(docs)
