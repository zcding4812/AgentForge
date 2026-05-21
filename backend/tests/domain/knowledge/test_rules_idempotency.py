from app.knowledge.kernel.rules import ingest_idempotency_key


def test_ingest_idempotency_key_stable() -> None:
    assert ingest_idempotency_key(kb_id=1, doc_id=2, content_version="abc") == "1:2:abc"
    assert ingest_idempotency_key(kb_id=1, doc_id=2, content_version="  abc  ") == "1:2:abc"


def test_ingest_idempotency_key_empty_version() -> None:
    assert ingest_idempotency_key(kb_id=1, doc_id=2, content_version="") == "1:2:"
