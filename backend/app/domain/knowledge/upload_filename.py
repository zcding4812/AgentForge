"""上传原始文件名的安全化（防路径穿越与过长文件名）。"""

from __future__ import annotations

import os


def safe_upload_filename(name: str, *, max_len: int = 255) -> str:
    base = os.path.basename((name or "").strip())
    if not base or base in {".", ".."}:
        return "unnamed.bin"
    base = base.replace("\x00", "")
    if not base:
        return "unnamed.bin"
    if len(base) > max_len:
        root, ext = os.path.splitext(base)
        base = root[: max(1, max_len - len(ext))] + ext
    return base
