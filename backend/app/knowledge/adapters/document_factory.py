"""规格 DocumentFactory：上传字节 / 已抽取文本 → LlamaIndex ``Document``。

实现委托 :mod:`app.knowledge.adapters.document_text.extract`，便于在 ingest / Pipeline 侧统一入口。
"""

from __future__ import annotations

from llama_index.core import Document

from app.knowledge.adapters.document_text.extract import (
    document_from_extracted_text,
    document_from_upload_bytes,
    extract_text_for_ingest,
)


class DocumentFactory:
    """bytes + MIME / 文件名 → 抽取 → ``Document``；或仅包装纯文本。"""

    @staticmethod
    def from_upload_bytes(
        raw: bytes,
        *,
        mime: str | None,
        filename: str,
        extra_metadata: dict[str, object] | None = None,
    ) -> Document:
        return document_from_upload_bytes(
            raw,
            mime=mime,
            filename=filename,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_extracted_text(
        text: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Document:
        return document_from_extracted_text(text, metadata=metadata)

    @staticmethod
    def extract_text_for_ingest(raw: bytes, mime: str | None, filename: str) -> str:
        """仅抽取纯文本（不包装 ``Document``）；建议在 worker 内 ``asyncio.to_thread`` 调用。"""
        return extract_text_for_ingest(raw, mime, filename)
