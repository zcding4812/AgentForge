"""Conversation 列表分页：与 HTTP ``Query`` 约束对齐的归一化。"""

from app.core.constants.conversation import (
    CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE,
    CONVERSATION_LIST_MAX_PAGE_SIZE,
)


def normalize_conversation_list_pagination(
    page: int,
    page_size: int,
    *,
    max_page_size: int = CONVERSATION_LIST_MAX_PAGE_SIZE,
) -> tuple[int, int]:
    """将原始 ``page`` / ``page_size`` 限制在安全范围内。

    与 ``GET /sessions`` 的 ``Query(ge=1)``、``page_size`` 上限一致；供服务层与其它调用方复用。

    Returns:
        ``(page, page_size)``，均为正整数且 ``page_size <= max_page_size``。
    """
    p = max(1, page)
    ps = min(max(1, page_size), max_page_size)
    return p, ps


def normalize_conversation_messages_pagination(
    page: int,
    page_size: int,
) -> tuple[int, int]:
    """会话详情消息列表分页：与 ``GET /sessions/{id}`` 的 ``page`` / ``page_size`` 上限一致。"""
    return normalize_conversation_list_pagination(
        page,
        page_size,
        max_page_size=CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE,
    )
