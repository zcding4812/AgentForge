"""``AgentKnowledgeBindingParser`` 与 config_json 约定。"""

from __future__ import annotations

from app.domain.agent.knowledge_binding import AgentKnowledgeBindingParser


def test_parse_defaults() -> None:
    s = AgentKnowledgeBindingParser().parse(None)
    assert s.enabled is False
    assert s.knowledge_base_ids == ()
    assert s.top_k == 8
    assert s.retrieval_override == "inherit"
    assert s.show_sources is False


def test_parse_full() -> None:
    raw = {
        "knowledge": {
            "enabled": True,
            "knowledge_base_ids": [3, 3, 99],
            "top_k": 12,
            "retrieval_override": "hybrid",
            "show_sources": True,
        }
    }
    s = AgentKnowledgeBindingParser().parse(raw)
    assert s.enabled is True
    assert s.knowledge_base_ids == (3, 99)
    assert s.top_k == 12
    assert s.retrieval_override == "hybrid"
    assert s.show_sources is True


def test_parse_caps_ids_and_top_k() -> None:
    raw = {
        "knowledge": {
            "enabled": True,
            "knowledge_base_ids": list(range(1, 20)),
            "top_k": 999,
        }
    }
    s = AgentKnowledgeBindingParser().parse(raw)
    assert len(s.knowledge_base_ids) == 8
    assert s.top_k == 30
