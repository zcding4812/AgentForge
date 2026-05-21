"""知识库文档在对象存储中的 key 拼接（无 MinIO SDK）。"""

from __future__ import annotations

from app.core.constants.knowledge import MINIO_DEFAULT_PREFIX_TEMPLATE, MINIO_OBJECT_PATH_SEGMENT


def build_knowledge_document_object_key(
    *,
    minio_prefix: str | None,
    kb_id: int,
    safe_filename: str,
    folder_hex: str,
) -> str:
    """``{prefix}/{segment}/{folder_hex}/{safe_filename}``，prefix 缺省时用模板。"""
    base = (minio_prefix or MINIO_DEFAULT_PREFIX_TEMPLATE.format(kb_id=kb_id)).strip().strip("/")
    return f"{base}/{MINIO_OBJECT_PATH_SEGMENT}/{folder_hex}/{safe_filename}"
