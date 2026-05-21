from __future__ import annotations

from llama_index.core import Document


def join_llama_documents(docs: list[Document], *, separator: str = "\n\n") -> str:
    """将 LlamaIndex ``Document`` 列表合并为单一字符串。"""
    parts = [d.text for d in docs if getattr(d, "text", None)]
    return separator.join(parts).strip()
