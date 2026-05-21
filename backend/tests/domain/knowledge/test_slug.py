from app.domain.knowledge.slug import is_valid_slug, propose_slug_from_name


def test_propose_slug_ascii() -> None:
    assert propose_slug_from_name("Hello World") == "hello-world"


def test_propose_slug_cjk_fallback() -> None:
    s = propose_slug_from_name("产品知识库")
    assert s.startswith("kb-")
    assert len(s) >= 6


def test_is_valid_slug() -> None:
    assert is_valid_slug("ab")
    assert is_valid_slug("a-b")
    assert not is_valid_slug("A")
    assert not is_valid_slug("-ab")
