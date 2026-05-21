"""上传 Content-Type 规范化。"""

from __future__ import annotations


def normalize_upload_mime(content_type: str | None, *, max_len: int) -> str | None:
    raw = (content_type or "").strip()
    if not raw:
        return None
    return raw[:max_len]
