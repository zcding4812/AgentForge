"""用户消息正文 token 估算（与网关 ``usage`` 无关，仅便于与助手侧 ``tokens`` 列同形展示）。"""

from __future__ import annotations

import tiktoken

# 与多数 OpenAI 兼容模型默认上下文编码一致；未知模型时作近似
_DEFAULT_ENCODING = "cl100k_base"


def estimate_user_content_tokens(text: str) -> int:
    """对纯用户文本做 tiktoken 计数；空串为 0。"""
    raw = text or ""
    if not raw:
        return 0
    enc = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return len(enc.encode(raw))
