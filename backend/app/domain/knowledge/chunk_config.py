"""知识库切块策略结构化配置（持久化于 ``knowledge_base.config_json['chunk_strategy']``）。

与顶栏 ``chunk_method`` / ``chunk_size`` / ``chunk_overlap`` 同步，供 ingest 与 OpenAPI 使用。
策略专属参数中部分项为预留能力：当前运行时仅对 length/char 的 ``separators``、``hard_limit`` 与 common 的 ``trim_whitespace`` 生效。
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

ChunkStrategyType = Literal[
    "length", "char", "token", "sentence", "markdown", "md", "json", "html", "code"
]


class CommonChunkConfig(BaseModel):
    """通用切块配置（与 MySQL 列对齐范围）。"""

    chunk_size: int = Field(default=512, ge=32, le=32768, description="切块目标长度")
    chunk_overlap: int = Field(default=50, ge=0, le=8192, description="切块重叠长度")
    trim_whitespace: bool = Field(default=True, description="去除首尾空白")
    keep_separator: bool = Field(default=False, description="保留分隔符（预留）")
    add_chunk_index: bool = Field(default=True, description="添加切块索引（预留）")


class LengthStrategyParams(BaseModel):
    """字符长度策略专属参数。"""

    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", "；", "，", " ", ""],
        description="优先切分分隔符，按顺序尝试",
    )
    hard_limit: bool = Field(default=False, description="严格长度限制（无重叠固定窗）")
    length_function: Literal["len", "utf8_len"] = Field(
        default="len", description="长度计算（utf8_len 预留）"
    )


class TokenStrategyParams(BaseModel):
    """Token 策略专属参数。"""

    tokenizer_name: Literal["cl100k_base", "p50k_base", "llama", "bert"] = Field(
        default="cl100k_base",
        description="Tokenizer 编码（当前运行时未绑定，仅持久化）",
    )
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", "；", "，", " ", ""],
        description="预留：与递归分句结合",
    )


class SentenceStrategyParams(BaseModel):
    """句边界策略专属参数。"""

    sentence_splitter: Literal["spacy", "jieba", "nltk", "regex"] = Field(
        default="spacy",
        description="分句工具（当前 ingest 使用 LlamaIndex SentenceSplitter，未绑定此项）",
    )
    max_sentences_per_chunk: int = Field(
        default=10, ge=1, le=50, description="单块最大句子数（预留）"
    )
    language: Literal["zh", "en", "multi"] = Field(default="zh", description="分句语言（预留）")


class MarkdownStrategyParams(BaseModel):
    """Markdown 结构策略。"""

    split_by_header_level: int = Field(default=2, ge=1, le=6, description="按标题层级切分（预留）")
    keep_code_blocks_intact: bool = Field(default=True, description="完整保留代码块（预留）")


class MdStrategyParams(BaseModel):
    """Markdown 简写策略；与 markdown 同实现，可配置单块上限说明字段。"""

    max_length_per_block: int = Field(
        default=2000, ge=100, le=32000, description="说明性上限（与 chunk_size 配合理解）"
    )


class JsonStrategyParams(BaseModel):
    """JSON 结构策略。"""

    split_by_key_path: str = Field(default="", max_length=256, description="JSON 路径（预留）")
    max_array_elements_per_chunk: int = Field(
        default=20, ge=1, le=500, description="单块最大数组元素数（预留）"
    )


class HtmlStrategyParams(BaseModel):
    """HTML 结构策略。"""

    split_by_tags: list[str] = Field(
        default_factory=lambda: ["h1", "h2", "h3", "p", "section"],
        description="按标签切分（预留）",
    )
    remove_script_style: bool = Field(default=True, description="移除脚本/样式（预留）")


class CodeStrategyParams(BaseModel):
    """代码 AST 策略。"""

    language: Literal["python", "java", "javascript", "go", "cpp"] = Field(
        default="python",
        description="代码语言（ingest 仍以文件扩展名推断为主，此项预留）",
    )
    split_by_unit: Literal["function", "class", "method", "module"] = Field(
        default="function",
        description="切分单元（预留）",
    )


def _params_model_for(strategy_type: str) -> type[BaseModel]:
    m: dict[str, type[BaseModel]] = {
        "length": LengthStrategyParams,
        "char": LengthStrategyParams,
        "token": TokenStrategyParams,
        "sentence": SentenceStrategyParams,
        "markdown": MarkdownStrategyParams,
        "md": MdStrategyParams,
        "json": JsonStrategyParams,
        "html": HtmlStrategyParams,
        "code": CodeStrategyParams,
    }
    return m.get(strategy_type, LengthStrategyParams)


class ChunkStrategyConfig(BaseModel):
    """完整切块策略（写入 ``config_json['chunk_strategy']``）。"""

    version: int = Field(default=1, ge=1, le=1, description="配置模式版本")
    common: CommonChunkConfig = Field(default_factory=CommonChunkConfig)
    strategy_type: ChunkStrategyType
    strategy_params: dict[str, Any] = Field(
        default_factory=dict, description="按 strategy_type 校验后的参数字典"
    )

    @model_validator(mode="after")
    def _validate_strategy_params(self) -> Self:
        model_cls = _params_model_for(self.strategy_type)
        typed = model_cls.model_validate(self.strategy_params or {})
        object.__setattr__(self, "strategy_params", typed.model_dump())
        return self

    def runtime_extra_params(self) -> dict[str, Any]:
        """传入分片层的策略参数字典。"""
        return dict(self.strategy_params)


def _normalize_method(method: str | None) -> ChunkStrategyType:
    key = (method or "length").strip().lower()
    allowed: set[str] = {
        "length",
        "char",
        "token",
        "sentence",
        "markdown",
        "md",
        "json",
        "html",
        "code",
    }
    if key not in allowed:
        logger.warning("unknown chunk_method=%r, coerce strategy to length", method)
        return "length"
    return key  # type: ignore[return-value]


def chunk_strategy_from_kb_row(row: Any) -> ChunkStrategyConfig:
    """从 ORM 行或 ``KnowledgeBaseOut`` 构造策略对象（无结构化配置时由列推导）。"""
    cj = getattr(row, "config_json", None) or {}
    raw = cj.get("chunk_strategy") if isinstance(cj, dict) else None
    if isinstance(raw, dict) and raw.get("strategy_type"):
        try:
            return ChunkStrategyConfig.model_validate(raw)
        except Exception:
            logger.warning(
                "invalid chunk_strategy in config_json, falling back to columns", exc_info=True
            )
    method = _normalize_method(getattr(row, "chunk_method", None))
    return ChunkStrategyConfig(
        version=1,
        common=CommonChunkConfig(
            chunk_size=int(getattr(row, "chunk_size", 512) or 512),
            chunk_overlap=int(getattr(row, "chunk_overlap", 50) or 50),
        ),
        strategy_type=method,
        strategy_params={},
    )


def effective_chunk_strategy_for_ingest(kb_row: Any, doc_row: Any) -> ChunkStrategyConfig:
    """ingest 用：知识库策略 + 文档级覆盖（``knowledge_document.chunk_method`` 为空则用库级）。"""
    base = chunk_strategy_from_kb_row(kb_row)
    dm = getattr(doc_row, "chunk_method", None)
    if dm is None or (isinstance(dm, str) and not str(dm).strip()):
        return base
    method = _normalize_method(dm)
    ds = getattr(doc_row, "chunk_size", None)
    dovl = getattr(doc_row, "chunk_overlap", None)
    chunk_size = int(ds if ds is not None else base.common.chunk_size)
    chunk_overlap = int(dovl if dovl is not None else base.common.chunk_overlap)
    return ChunkStrategyConfig(
        version=base.version,
        common=CommonChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            trim_whitespace=base.common.trim_whitespace,
            keep_separator=base.common.keep_separator,
            add_chunk_index=base.common.add_chunk_index,
        ),
        strategy_type=method,
        strategy_params=base.strategy_params,
    )


def apply_chunk_strategy_to_updates(
    *,
    cfg: ChunkStrategyConfig,
    base_config_json: dict[str, Any] | None,
    client_config_json: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并 ``chunk_strategy`` 到 ``config_json``，并返回应写入 ORM 的 ``config_json`` 快照。"""
    merged: dict[str, Any] = {**(base_config_json or {})}
    if client_config_json is not None:
        merged.update(client_config_json)
    merged["chunk_strategy"] = cfg.model_dump(mode="json")
    return merged


def orm_updates_from_chunk_strategy(cfg: ChunkStrategyConfig) -> dict[str, Any]:
    """由结构化策略生成与 MySQL 列同步的字段。"""
    return {
        "chunk_method": cfg.strategy_type,
        "chunk_size": cfg.common.chunk_size,
        "chunk_overlap": cfg.common.chunk_overlap,
    }
