"""知识库文本分片：策略注册表（LlamaIndex 绑定）与 ``chunk_method`` 解析、工厂、降级、统一入口。

新增策略：实现 :class:`BaseChunkStrategy` 并加入 ``_BUILTIN_STRATEGY_CLASSES``。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

from llama_index.core import Document
from llama_index.core.node_parser import (
    HTMLNodeParser,
    JSONNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.node_parser.text.code import CodeSplitter
from llama_index.core.schema import BaseNode

from app.knowledge.kernel.exceptions import ChunkSplitFailedError

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 50
DEFAULT_CHUNK_METHOD: Final[str] = "length"
MAX_SEGMENT_CHAR_LIMIT: Final[int] = 4096
MIN_CHUNK_SIZE: Final[int] = 32

_EXT_TO_TREE_SITTER_LANG: Final[dict[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".json": "json",
}


def split_text_by_length(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    hard_limit: bool = False,
) -> list[str]:
    """按字符窗切分（``length`` / ``char``）；``hard_limit=True`` 时为无重叠固定窗。"""
    if not text:
        return []
    size = max(MIN_CHUNK_SIZE, chunk_size)
    if hard_limit:
        return [text[i : i + size] for i in range(0, len(text), size)]
    overlap = max(0, min(chunk_overlap, size - 1))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def split_with_priority_separators(
    text: str,
    *,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    hard_limit: bool = False,
) -> list[str]:
    """按分隔符优先级拆段，再对超长段做字符窗切分。"""
    if not text:
        return []
    seps = [s for s in separators if s is not None]
    if not seps:
        return split_text_by_length(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hard_limit=hard_limit,
        )
    pieces: list[str] = [text]
    for sep in seps:
        if sep == "":
            break
        next_pieces: list[str] = []
        for piece in pieces:
            next_pieces.extend(piece.split(sep))
        pieces = [p for p in next_pieces if p != ""]
        if not pieces:
            return split_text_by_length(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                hard_limit=hard_limit,
            )
        if max(len(p) for p in pieces) <= chunk_size:
            break
    out: list[str] = []
    for p in pieces:
        if len(p) <= chunk_size:
            out.append(p)
        else:
            out.extend(
                split_text_by_length(
                    p,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    hard_limit=hard_limit,
                )
            )
    return out


@dataclass
class ChunkConfig:
    """分片参数（服务层传入后在此统一裁剪）。"""

    chunk_size: int
    chunk_overlap: int
    source_filename: str | None = None
    extra_params: dict[str, object] | None = None
    trim_whitespace: bool = False

    def __post_init__(self) -> None:
        self.chunk_size = max(MIN_CHUNK_SIZE, self.chunk_size)
        self.chunk_overlap = max(0, min(self.chunk_overlap, self.chunk_size - 1))


class BaseChunkStrategy(ABC):
    """分片策略抽象基类。"""

    strategy_name: str

    @abstractmethod
    def split(self, text: str, config: ChunkConfig) -> list[str]:
        """返回非空文本块列表。"""

    @staticmethod
    def _nodes_to_chunks(nodes: list[BaseNode]) -> list[str]:
        out: list[str] = []
        for n in nodes:
            t = (n.get_content() or "").strip()
            if t:
                out.append(t)
        return out

    def _resplit_oversized(self, chunks: list[str], config: ChunkConfig) -> list[str]:
        limit = max(config.chunk_size * 3, MAX_SEGMENT_CHAR_LIMIT)
        merged: list[str] = []
        for c in chunks:
            if len(c) <= limit:
                merged.append(c)
            else:
                merged.extend(
                    split_text_by_length(
                        c,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        hard_limit=False,
                    )
                )
        return merged


class CharChunkStrategy(BaseChunkStrategy):
    strategy_name = "char"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        extra = config.extra_params or {}
        hard = bool(extra.get("hard_limit", False))
        seps = extra.get("separators")
        if isinstance(seps, list) and len(seps) > 0 and any(str(s) != "" for s in seps):
            return split_with_priority_separators(
                text,
                separators=[str(s) for s in seps],
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                hard_limit=hard,
            )
        return split_text_by_length(
            text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            hard_limit=hard,
        )


class TokenChunkStrategy(BaseChunkStrategy):
    strategy_name = "token"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return [p for p in splitter.split_text(text) if p]


class SentenceChunkStrategy(BaseChunkStrategy):
    strategy_name = "sentence"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        return self._nodes_to_chunks(nodes)


class MarkdownChunkStrategy(BaseChunkStrategy):
    strategy_name = "markdown"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        parser = MarkdownNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        chunks = self._nodes_to_chunks(nodes)
        if not chunks:
            return split_text_by_length(
                text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                hard_limit=False,
            )
        return self._resplit_oversized(chunks, config)


class JsonChunkStrategy(BaseChunkStrategy):
    strategy_name = "json"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        parser = JSONNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        chunks = self._nodes_to_chunks(nodes)
        if not chunks:
            return split_text_by_length(
                text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                hard_limit=False,
            )
        return self._resplit_oversized(chunks, config)


class HtmlChunkStrategy(BaseChunkStrategy):
    strategy_name = "html"

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        parser = HTMLNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents([Document(text=text)])
        chunks = self._nodes_to_chunks(nodes)
        if not chunks:
            return split_text_by_length(
                text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                hard_limit=False,
            )
        return self._resplit_oversized(chunks, config)


class CodeChunkStrategy(BaseChunkStrategy):
    strategy_name = "code"

    @staticmethod
    def _infer_language(filename: str | None) -> str:
        ext = os.path.splitext((filename or "").lower())[1]
        return _EXT_TO_TREE_SITTER_LANG.get(ext, "python")

    def split(self, text: str, config: ChunkConfig) -> list[str]:
        lang = self._infer_language(config.source_filename)
        max_chars = max(256, min(config.chunk_size, 16000))
        chunk_lines = max(8, min(200, max_chars // 40))
        lines_overlap = max(0, min(config.chunk_overlap // 4, chunk_lines - 1))
        try:
            splitter = CodeSplitter.from_defaults(
                language=lang,
                chunk_lines=chunk_lines,
                chunk_lines_overlap=lines_overlap or 1,
                max_chars=max_chars,
                count_mode="char",
            )
        except Exception as e:
            logger.warning(
                "code splitter unavailable, fallback to token | lang=%s err=%s",
                lang,
                e,
            )
            return TokenChunkStrategy().split(text, config)
        parts = splitter.split_text(text)
        return [p for p in parts if p]


_BUILTIN_STRATEGY_CLASSES: Final[tuple[type[BaseChunkStrategy], ...]] = (
    CharChunkStrategy,
    TokenChunkStrategy,
    SentenceChunkStrategy,
    MarkdownChunkStrategy,
    JsonChunkStrategy,
    HtmlChunkStrategy,
    CodeChunkStrategy,
)


def _build_strategy_registry() -> dict[str, type[BaseChunkStrategy]]:
    reg: dict[str, type[BaseChunkStrategy]] = {}
    for cls in _BUILTIN_STRATEGY_CLASSES:
        key = cls.strategy_name.strip().lower()
        reg[key] = cls
    # 别名：与 ``length`` / ``md`` 请求键对齐，避免空子类重复继承
    reg["length"] = CharChunkStrategy
    reg["md"] = MarkdownChunkStrategy
    return reg


STRATEGY_REGISTRY: Final[dict[str, type[BaseChunkStrategy]]] = _build_strategy_registry()

CHUNK_METHOD_KEYS: Final[tuple[str, ...]] = tuple(sorted(STRATEGY_REGISTRY.keys()))
CHUNK_METHODS: Final[tuple[str, ...]] = CHUNK_METHOD_KEYS


# ---------- 门面：解析 chunk_method、工厂、降级 ----------


def _strategy_for_method(chunk_method: str | None) -> tuple[type[BaseChunkStrategy], str]:
    """返回 (策略类, 规范化后的请求键，供日志与分支判断)。"""
    key = (chunk_method or DEFAULT_CHUNK_METHOD).strip().lower()
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        logger.warning("unknown chunk_method=%r, using length", chunk_method)
        return CharChunkStrategy, key
    return cls, key


class ChunkStrategyFactory:
    """由策略名创建策略实例（无单例状态）。"""

    @staticmethod
    def create(strategy_name: str) -> BaseChunkStrategy:
        cls, _ = _strategy_for_method(strategy_name)
        return cls()


def _trim_chunks(chunks: list[str], *, trim_whitespace: bool) -> list[str]:
    if not trim_whitespace:
        return chunks
    return [c.strip() for c in chunks if c.strip()]


def _split_with_char_fallback(text: str, config: ChunkConfig) -> list[str]:
    """统一降级：字符窗策略（与 ``length`` / ``char`` 注册项一致）。"""
    return CharChunkStrategy().split(text, config)


def split_knowledge_text_llama_index(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_method: str,
    source_filename: str | None = None,
    extra_params: dict[str, object] | None = None,
    trim_whitespace: bool = False,
) -> list[str]:
    """
    按知识库 ``chunk_method`` 将全文切成字符串块列表。

    :param source_filename: 可选；``chunk_method=code`` 时用于推断 tree-sitter 语言。
    """
    if not text:
        return []

    cls, log_key = _strategy_for_method(chunk_method)
    strategy = cls()
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_filename=source_filename,
        extra_params=extra_params,
        trim_whitespace=trim_whitespace,
    )

    try:
        chunks = strategy.split(text, config)
        return _trim_chunks(chunks, trim_whitespace=trim_whitespace)
    except NotImplementedError:
        raise
    except Exception as e:
        logger.exception("chunk_method=%s failed, fallback to char/length", log_key)
        if cls is CharChunkStrategy:
            raise ChunkSplitFailedError(f"分片失败（{log_key}）：{e}") from e
        try:
            chunks = _split_with_char_fallback(text, config)
            return _trim_chunks(chunks, trim_whitespace=trim_whitespace)
        except Exception as e2:
            raise ChunkSplitFailedError(
                f"分片失败（{log_key}），降级仍失败：{e2}",
            ) from e2


# 与 ingest / NodeParserFactory 文档对齐的稳定名称（实现即 :func:`split_knowledge_text_llama_index`）。
split_knowledge_text_by_strategy = split_knowledge_text_llama_index


def split_knowledge_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_method: str = DEFAULT_CHUNK_METHOD,
    source_filename: str | None = None,
    extra_params: dict[str, object] | None = None,
    trim_whitespace: bool = False,
) -> list[str]:
    """门面入口（带默认切块参数），与 :func:`split_knowledge_text_llama_index` 行为一致。"""
    return split_knowledge_text_llama_index(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_method=chunk_method,
        source_filename=source_filename,
        extra_params=extra_params,
        trim_whitespace=trim_whitespace,
    )


split_text = split_knowledge_text
