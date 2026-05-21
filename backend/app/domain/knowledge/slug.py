"""知识库 slug：自名称推导或占位（纯函数，可单测）。"""

from __future__ import annotations

import re
import uuid

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?$")


def is_valid_slug(s: str) -> bool:
    return bool(_SLUG_RE.fullmatch(s.strip()))


def propose_slug_from_name(name: str) -> str:
    """从展示名生成 URL 友好 slug；无法从纯中文等推导时使用随机前缀。"""
    raw = (name or "").strip().lower()
    # 仅保留 a-z0-9，其它视为分隔
    ascii_only = re.sub(r"[^a-z0-9]+", "-", raw, flags=re.ASCII)
    ascii_only = re.sub(r"-+", "-", ascii_only).strip("-")
    if len(ascii_only) >= 2:
        return ascii_only[:128]
    return f"kb-{uuid.uuid4().hex[:12]}"
