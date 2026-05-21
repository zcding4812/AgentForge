"""``DocumentFactory`` 委托 ``document_text.extract``，行为与现有入口一致。"""

from __future__ import annotations

from app.knowledge.adapters.document_factory import DocumentFactory
from app.knowledge.adapters.document_text.extract import (
    document_from_extracted_text,
    document_from_upload_bytes,
)


def test_from_extracted_text_delegates() -> None:
    d1 = DocumentFactory.from_extracted_text("hello", metadata={"kb_id": 1})
    d2 = document_from_extracted_text("hello", metadata={"kb_id": 1})
    assert d1.text == d2.text
    assert d1.metadata.get("kb_id") == 1


def test_from_upload_bytes_delegates() -> None:
    raw = b"plain line\n"
    d1 = DocumentFactory.from_upload_bytes(raw, mime="text/plain", filename="a.txt")
    d2 = document_from_upload_bytes(raw, mime="text/plain", filename="a.txt")
    assert d1.text == d2.text
    assert d1.metadata.get("filename") == d2.metadata.get("filename")
