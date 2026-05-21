"""``normalize_conversation_list_pagination`` 与列表 Query 约束一致。"""

from app.core.constants.conversation import (
    CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE,
    CONVERSATION_LIST_MAX_PAGE_SIZE,
)
from app.domain.conversation.pagination import (
    normalize_conversation_list_pagination,
    normalize_conversation_messages_pagination,
)


def test_clamps_page_and_page_size() -> None:
    assert normalize_conversation_list_pagination(0, 0) == (1, 1)
    assert normalize_conversation_list_pagination(-3, 5) == (1, 5)
    assert normalize_conversation_list_pagination(2, CONVERSATION_LIST_MAX_PAGE_SIZE + 99) == (
        2,
        CONVERSATION_LIST_MAX_PAGE_SIZE,
    )


def test_preserves_valid_pair() -> None:
    assert normalize_conversation_list_pagination(3, 20) == (3, 20)


def test_messages_pagination_clamps_page_size() -> None:
    assert normalize_conversation_messages_pagination(
        1, CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE + 50
    ) == (
        1,
        CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE,
    )


def test_messages_pagination_preserves_valid_pair() -> None:
    assert normalize_conversation_messages_pagination(2, 10) == (2, 10)
