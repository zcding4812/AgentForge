"""Mongo 关键词子路：匹配次数与首次出现位置排序。"""

from __future__ import annotations

from app.repositories.chunk_repo import _keyword_relevance_sort_key


def test_keyword_sort_key_prefers_more_occurrences() -> None:
    more = _keyword_relevance_sort_key("注册表 注册表 其它", "注册表")
    one = _keyword_relevance_sort_key("仅一处注册表", "注册表")
    assert more < one


def test_keyword_sort_key_prefers_earlier_first_match_when_same_count() -> None:
    early = _keyword_relevance_sort_key("注册表在前", "注册表")
    late = _keyword_relevance_sort_key("在后" * 20 + "注册表", "注册表")
    assert early < late
