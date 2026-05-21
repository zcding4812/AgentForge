"""从原始字节提取纯文本供 ingest 切块（LlamaIndex Readers + 少量 BS4 等）。

扩展方式：在 :mod:`handlers` 中新增实现 :class:`~app.knowledge.adapters.document_text.protocol.DocumentTextHandler`
的类，并注册到 :data:`~app.knowledge.adapters.document_text.registry.HANDLERS`（按 ``priority`` 升序，先匹配先执行）。
"""

from __future__ import annotations

from app.knowledge.adapters.document_text.extract import (
    document_from_extracted_text,
    document_from_upload_bytes,
    extract_text_for_ingest,
)
from app.knowledge.kernel.exceptions import DocumentTextExtractionError

__all__ = [
    "DocumentTextExtractionError",
    "document_from_extracted_text",
    "document_from_upload_bytes",
    "extract_text_for_ingest",
]
