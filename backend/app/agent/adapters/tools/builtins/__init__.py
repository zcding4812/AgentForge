"""进程内置工具：启动时扫描本子包内模块并注册到全局 :class:`~app.agent.adapters.tools.registry.NamespacedToolRegistry`。

约定：每个子模块（如 ``server_time.py``）内用 ``@tool`` 声明工具；本包在
:func:`register_builtin_tools` 中 **自动 import 子模块** 并注册其中所有
:class:`~langchain_core.tools.BaseTool` 实例，无需在 ``__init__`` 手写名称列表。
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from langchain_core.tools import BaseTool

from app.agent.adapters.tools.registry import register_tool

# 在 register_builtin_tools 内就地 update，依赖方 ``name in BUILTIN_TOOL_NAMES`` 始终指向同一 set 对象
BUILTIN_TOOL_NAMES: set[str] = set()


def _discover_base_tools_in_module(mod: ModuleType) -> list[BaseTool]:
    found: list[BaseTool] = []
    for key in dir(mod):
        if key.startswith("_"):
            continue
        obj = getattr(mod, key)
        if isinstance(obj, BaseTool):
            found.append(obj)
    return found


def register_builtin_tools(*, overwrite: bool = True) -> None:
    """扫描 ``builtins`` 包下子模块，注册各模块中出现的 ``BaseTool``（含 ``@tool`` 装饰结果）。"""
    BUILTIN_TOOL_NAMES.clear()
    pkg_name = __name__
    for _finder, mod_short, _ispkg in pkgutil.iter_modules(__path__):
        if mod_short.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg_name}.{mod_short}")
        for t in _discover_base_tools_in_module(mod):
            register_tool(t.name, t, overwrite=overwrite)
            BUILTIN_TOOL_NAMES.add(t.name)


__all__ = [
    "BUILTIN_TOOL_NAMES",
    "register_builtin_tools",
]
