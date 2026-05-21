from __future__ import annotations

# 兜底：各类源码、配置、日志等文本；具体二进制格式由更高优先级的处理器覆盖。
_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".jsonl",
        ".jsonc",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        ".py",
        ".pyi",
        ".pyw",
        ".pyx",
        ".ts",
        ".tsx",
        ".mts",
        ".cts",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
        ".svelte",
        ".css",
        ".scss",
        ".less",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
        ".cmake",
        ".gradle",
        ".lock",
        ".log",
        ".rst",
        ".adoc",
        ".tex",
        ".bib",
        ".svg",
    }
)

_PLAIN_MIME_EXACT: frozenset[str] = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/x-yaml",
        "application/x-toml",
        "application/javascript",
        "application/typescript",
        "application/x-sh",
    }
)


def _decode_plain(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class PlainTextHandler:
    name = "plain_text"
    priority = 80

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        if any(fn.endswith(s) for s in _TEXT_SUFFIXES):
            return True
        if mime in _PLAIN_MIME_EXACT:
            return True
        if mime.startswith("text/") and mime not in ("text/html", "text/csv"):
            return True
        if mime == "application/octet-stream" and any(fn.endswith(s) for s in _TEXT_SUFFIXES):
            return True
        return False

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        return _decode_plain(raw)
