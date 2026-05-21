"""按命名空间注册 LangChain 工具，并按名称解析为可注入图的句柄。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from app.agent.kernel.tool_runtime import ToolLifecycle

logger = logging.getLogger(__name__)


class ToolRegistrationConflictError(ValueError):
    """同名同命名空间已存在且未指定 overwrite。"""

    def __init__(self, name: str, namespace: str) -> None:
        self.name = name
        self.namespace = namespace
        super().__init__(
            f"工具已注册: namespace={namespace!r} name={name!r}；如需覆盖请使用 overwrite=True"
        )


class ToolResolutionStrictError(ValueError):
    """strict_tool_names 为 true 时存在未注册名称。"""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing_names = missing
        joined = ", ".join(missing)
        super().__init__(f"以下工具名未注册: {joined}")


@dataclass(frozen=True, slots=True)
class ResolvedTools:
    """工具解析结果：合法句柄 + 未在注册表中的名称（顺序与请求中首次出现一致）。"""

    tools: tuple[BaseTool | dict[str, Any], ...]
    missing_names: tuple[str, ...]


class NamespacedToolRegistry:
    """``namespace`` → ``tool_name`` → ``BaseTool``；解析结果带简单缓存。"""

    __slots__ = ("_resolve_cache", "_tools")

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, BaseTool]] = {}
        self._resolve_cache: dict[str, BaseTool] = {}

    def register(
        self,
        name: str,
        tool: BaseTool,
        namespace: str = "default",
        *,
        overwrite: bool = False,
    ) -> None:
        ns = self._tools.setdefault(namespace, {})
        if name in ns and not overwrite:
            raise ToolRegistrationConflictError(name, namespace)
        ns[name] = tool
        self._resolve_cache.clear()

    def unregister(self, name: str, namespace: str = "default") -> None:
        ns = self._tools.get(namespace)
        if ns and name in ns:
            del ns[name]
            self._resolve_cache.clear()

    def list_tool_entries(self, namespace: str = "default") -> list[tuple[str, BaseTool]]:
        """返回命名空间内已注册工具名与实例，按名称排序。"""
        ns = self._tools.get(namespace)
        if not ns:
            return []
        return sorted(ns.items(), key=lambda x: x[0])

    def resolve_tool_names_with_report(
        self,
        names: Sequence[str] | None,
        namespace: str = "default",
        *,
        strict: bool = False,
    ) -> ResolvedTools:
        """按名称解析；未知名称记入 missing_names。strict=True 且存在 missing 时抛错。"""
        if not names:
            return ResolvedTools((), ())
        missing: list[str] = []
        out: list[BaseTool | dict[str, Any]] = []
        for n in names:
            cache_key = f"{namespace}:{n}"
            cached = self._resolve_cache.get(cache_key)
            if cached is not None:
                out.append(cached)
                continue
            tool = self._tools.get(namespace, {}).get(n)
            if tool is not None:
                self._resolve_cache[cache_key] = tool
                out.append(tool)
            else:
                missing.append(n)
        if strict and missing:
            raise ToolResolutionStrictError(tuple(missing))
        if missing:
            logger.warning(
                "agent.tools.missing_skipped",
                extra={
                    "event": "agent.tools.missing_skipped",
                    "namespace": namespace,
                    "missing_count": len(missing),
                    "missing_sample": missing[:20],
                },
            )
        return ResolvedTools(tuple(out), tuple(missing))

    def resolve_tool_names(
        self,
        names: Sequence[str],
        namespace: str = "default",
    ) -> tuple[BaseTool | dict[str, Any], ...]:
        """兼容旧调用：忽略未注册名称并打日志（非 strict）。"""
        return self.resolve_tool_names_with_report(names, namespace, strict=False).tools

    async def aclose_all_tools(self) -> None:
        """进程关闭时调用：对实现 :class:`~app.agent.kernel.tool_runtime.ToolLifecycle` 的工具执行 ``aclose``。"""
        seen: set[int] = set()
        for ns in self._tools.values():
            for tool in ns.values():
                tid = id(tool)
                if tid in seen:
                    continue
                seen.add(tid)
                if isinstance(tool, ToolLifecycle):
                    try:
                        await tool.aclose()
                    except Exception:
                        logger.exception(
                            "agent.tool.lifecycle_close_failed",
                            extra={"event": "agent.tool.lifecycle_close_failed"},
                        )


tool_registry = NamespacedToolRegistry()
register_tool = tool_registry.register
unregister_tool = tool_registry.unregister
resolve_tool_names = tool_registry.resolve_tool_names
resolve_tool_names_with_report = tool_registry.resolve_tool_names_with_report
aclose_all_registered_tools = tool_registry.aclose_all_tools
