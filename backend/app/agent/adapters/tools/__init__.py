"""工具适配器包：对外聚合导出注册表与内置工具注册。"""

from app.agent.adapters.tools.builtins import register_builtin_tools
from app.agent.adapters.tools.registry import (
    NamespacedToolRegistry,
    ResolvedTools,
    ToolRegistrationConflictError,
    ToolResolutionStrictError,
    aclose_all_registered_tools,
    register_tool,
    resolve_tool_names,
    resolve_tool_names_with_report,
    tool_registry,
    unregister_tool,
)

__all__ = [
    "NamespacedToolRegistry",
    "ResolvedTools",
    "ToolRegistrationConflictError",
    "ToolResolutionStrictError",
    "aclose_all_registered_tools",
    "register_builtin_tools",
    "register_tool",
    "resolve_tool_names",
    "resolve_tool_names_with_report",
    "tool_registry",
    "unregister_tool",
]
