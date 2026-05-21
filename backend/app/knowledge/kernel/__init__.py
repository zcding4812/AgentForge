"""内核：端口协议、DTO、纯规则与领域异常（无 ORM / 无 LlamaIndex）。"""

from app.knowledge.kernel.rules import ingest_idempotency_key

__all__ = ["ingest_idempotency_key"]
