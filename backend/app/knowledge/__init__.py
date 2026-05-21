"""知识库有界上下文：编排、内核规则与对外门面。

对外请优先自 :mod:`app.knowledge.facades` 导入门面类。
"""

from app.knowledge.facades import EmbeddingAdapter, KnowledgeRetrievalFacade

__all__ = ["EmbeddingAdapter", "KnowledgeRetrievalFacade"]
