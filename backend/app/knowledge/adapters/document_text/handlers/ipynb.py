from __future__ import annotations

from llama_index.readers.file import IPYNBReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class IpynbHandler:
    name = "ipynb"
    priority = 32

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in ("application/x-ipynb+json",) or fn.endswith(".ipynb")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".ipynb") as path:
            docs = IPYNBReader(concatenate=True).load_data(path)
            return join_llama_documents(docs)
