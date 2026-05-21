"""持久化外部工具枚举：与库表存储及 ORM 取值一致。"""

from __future__ import annotations

from enum import StrEnum


class ToolKind(StrEnum):
    HTTP = "http"
    MCP = "mcp"
