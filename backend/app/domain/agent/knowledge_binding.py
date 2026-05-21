"""Agent 资源上的知识库绑定：从 ``agent_entity.config_json.knowledge`` 解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RetrievalOverrideSetting = Literal["inherit", "keyword", "vector", "hybrid"]

_DEFAULT_TOP_K = 8
_MAX_KB_IDS = 8
_TOP_K_MIN = 1
_TOP_K_MAX = 30


@dataclass(frozen=True, slots=True)
class AgentKnowledgeBindingSettings:
    """与前端工作区 ``config_json.knowledge`` 对齐；用于 invoke 前自动检索并注入 context。"""

    enabled: bool
    knowledge_base_ids: tuple[int, ...]
    top_k: int
    #: inherit → 检索时 ``retrieval_override=None``（与知识库配置一致）
    retrieval_override: RetrievalOverrideSetting
    #: 为 True 时在每条分片后追加 doc_id / chunk_index 等来源说明，便于模型引用
    show_sources: bool


class AgentKnowledgeBindingParser:
    """解析 ``config_json`` 中的 ``knowledge`` 块。"""

    def parse(self, config_json: dict[str, Any] | None) -> AgentKnowledgeBindingSettings:
        if not config_json or not isinstance(config_json, dict):
            return self._defaults()
        block = config_json.get("knowledge")
        if not isinstance(block, dict):
            return self._defaults()
        return AgentKnowledgeBindingSettings(
            enabled=self._parse_enabled(block.get("enabled")),
            knowledge_base_ids=self._parse_ids(block.get("knowledge_base_ids")),
            top_k=self._parse_top_k(block.get("top_k")),
            retrieval_override=self._parse_override(block.get("retrieval_override")),
            show_sources=self._parse_show_sources(block.get("show_sources")),
        )

    @staticmethod
    def _defaults() -> AgentKnowledgeBindingSettings:
        return AgentKnowledgeBindingSettings(
            enabled=False,
            knowledge_base_ids=(),
            top_k=_DEFAULT_TOP_K,
            retrieval_override="inherit",
            show_sources=False,
        )

    @staticmethod
    def _parse_enabled(raw: Any) -> bool:
        return isinstance(raw, bool) and raw

    @staticmethod
    def _parse_show_sources(raw: Any) -> bool:
        return isinstance(raw, bool) and raw

    @staticmethod
    def _parse_ids(raw: Any) -> tuple[int, ...]:
        if not isinstance(raw, list):
            return ()
        out: list[int] = []
        for x in raw:
            if isinstance(x, bool):
                continue
            if isinstance(x, int) and x >= 1:
                out.append(x)
            elif isinstance(x, float) and x >= 1:
                out.append(int(x))
            if len(out) >= _MAX_KB_IDS:
                break
        # 去重保序
        seen: set[int] = set()
        uniq: list[int] = []
        for i in out:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        return tuple(uniq[:_MAX_KB_IDS])

    @staticmethod
    def _parse_top_k(raw: Any) -> int:
        if raw is None:
            return _DEFAULT_TOP_K
        if isinstance(raw, bool):
            return _DEFAULT_TOP_K
        if isinstance(raw, int):
            return max(_TOP_K_MIN, min(_TOP_K_MAX, raw))
        if isinstance(raw, float):
            return max(_TOP_K_MIN, min(_TOP_K_MAX, int(raw)))
        return _DEFAULT_TOP_K

    @staticmethod
    def _parse_override(raw: Any) -> RetrievalOverrideSetting:
        if raw is None:
            return "inherit"
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s == "inherit":
                return "inherit"
            if s == "keyword":
                return "keyword"
            if s == "vector":
                return "vector"
            if s == "hybrid":
                return "hybrid"
        return "inherit"


def parse_agent_knowledge_binding(
    config_json: dict[str, Any] | None,
) -> AgentKnowledgeBindingSettings:
    return AgentKnowledgeBindingParser().parse(config_json)


def knowledge_binding_settings_as_json(settings: AgentKnowledgeBindingSettings) -> dict[str, Any]:
    """写入缓存或 API 时的稳定 dict（list 可 JSON 序列化）。"""
    return {
        "enabled": settings.enabled,
        "knowledge_base_ids": list(settings.knowledge_base_ids),
        "top_k": settings.top_k,
        "retrieval_override": settings.retrieval_override,
        "show_sources": settings.show_sources,
    }


def knowledge_binding_from_cache_payload(data: dict[str, Any]) -> AgentKnowledgeBindingSettings:
    """从缓存 JSON 对象还原（字段与 :func:`knowledge_binding_settings_as_json` 对称）。"""
    return AgentKnowledgeBindingParser().parse({"knowledge": data})
