from app.domain.conversation.content_tokens import estimate_user_content_tokens


def test_estimate_empty_is_zero() -> None:
    assert estimate_user_content_tokens("") == 0


def test_estimate_non_empty_positive() -> None:
    assert estimate_user_content_tokens("hello") >= 1
