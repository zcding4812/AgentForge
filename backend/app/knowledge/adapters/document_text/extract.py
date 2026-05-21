"""入口：按注册表选择处理器并提取文本；以及由文本/上传字节构造 LlamaIndex ``Document``。"""

from __future__ import annotations

import logging

from llama_index.core import Document

from app.knowledge.adapters.document_text.registry import HANDLERS
from app.knowledge.kernel.exceptions import DocumentTextExtractionError

logger = logging.getLogger(__name__)


def _normalize_mime(mime: str | None) -> str:
    return (mime or "").strip().lower()


def extract_text_for_ingest(raw: bytes, mime: str | None, filename: str) -> str:
    """
    从上传对象字节解析纯文本。同步阻塞（建议在 ``asyncio.to_thread`` 中调用）。

    :raises DocumentTextExtractionError: 无匹配格式或解析失败
    """
    if not raw:
        return ""

    m = _normalize_mime(mime)
    fn = filename or ""
    ordered = sorted(HANDLERS, key=lambda h: h.priority)
    for handler in ordered:
        if not handler.matches(m, fn):
            continue
        try:
            text = handler.extract(raw, m, fn)
        except Exception as e:
            logger.exception(
                "document_text handler failed | handler=%s mime=%s filename=%s",
                handler.name,
                m,
                fn,
            )
            raise DocumentTextExtractionError(
                f"{handler.name} 解析失败：{e}",
            ) from e
        if text is None:
            text = ""
        return text

    raise DocumentTextExtractionError(
        f"不支持的文档类型：mime={m or '(空)'} filename={fn or '(空)'}",
    )


def document_from_extracted_text(
    text: str,
    *,
    metadata: dict[str, object] | None = None,
) -> Document:
    """由已抽取的纯文本构造 ``Document``。"""
    md = dict(metadata) if metadata else {}
    return Document(text=text or "", metadata=md)


def document_from_upload_bytes(
    raw: bytes,
    *,
    mime: str | None,
    filename: str,
    extra_metadata: dict[str, object] | None = None,
) -> Document:
    """``bytes`` + MIME / 文件名 → 抽取文本 → ``Document``。"""
    body = extract_text_for_ingest(raw, mime, filename)
    md: dict[str, object] = {
        "filename": filename or "",
        "mime": (mime or "").strip().lower(),
    }
    if extra_metadata:
        md.update(extra_metadata)
    return Document(text=body, metadata=md)
