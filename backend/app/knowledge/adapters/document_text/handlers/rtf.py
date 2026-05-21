from __future__ import annotations

from llama_index.readers.file import RTFReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class RtfHandler:
    name = "rtf"
    priority = 28

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in ("application/rtf", "text/rtf") or fn.endswith(".rtf")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".rtf") as path:
            docs = RTFReader().load_data(path)
            return join_llama_documents(docs)
