"""文档文本提取注册表与兜底解码。"""

from __future__ import annotations

import pytest

from app.knowledge.adapters.document_text import (
    DocumentTextExtractionError,
    extract_text_for_ingest,
)


def test_plain_utf8_and_markdown() -> None:
    assert extract_text_for_ingest(b"hello", "text/plain", "a.txt") == "hello"
    assert (
        extract_text_for_ingest("\u4e2d\u6587".encode(), "text/markdown", "x.md") == "\u4e2d\u6587"
    )


def test_html_strips_script() -> None:
    raw = b"<html><head></head><body><script>x</script><p>ok</p></body></html>"
    out = extract_text_for_ingest(raw, "text/html", "p.html")
    assert "x" not in out
    assert "ok" in out


def test_unknown_binary_raises() -> None:
    with pytest.raises(DocumentTextExtractionError):
        extract_text_for_ingest(bytes(8), "application/octet-stream", "unknown.bin")
