from __future__ import annotations

from llama_index.readers.file import PptxReader

from app.knowledge.adapters.document_text._llama_util import join_llama_documents
from app.knowledge.adapters.document_text._tempfile import bytes_as_temp_file


class PptxHandler:
    name = "pptx"
    priority = 18

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ) or fn.endswith(".pptx")

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        with bytes_as_temp_file(raw, ".pptx") as path:
            reader = PptxReader(
                extract_images=False,
                context_consolidation_with_llm=False,
                raise_on_error=True,
            )
            docs = reader.load_data(path)
            return join_llama_documents(docs)
