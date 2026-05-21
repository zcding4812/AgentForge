"""工作台工具横切：统一出参信封与工具名过滤。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkbenchToolEnvelope:
    """工具响应值对象；LangChain 侧使用 :meth:`as_tool_dict`。"""

    code: str
    message: str
    data: Any

    def as_tool_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "data": self.data}


class WorkbenchEnvelopeFactory:
    """响应工厂：统一成功 / 失败信封。"""

    OK_CODE: str = "OK"

    @staticmethod
    def ok(data: Any, *, message: str = "success") -> WorkbenchToolEnvelope:
        return WorkbenchToolEnvelope(
            code=WorkbenchEnvelopeFactory.OK_CODE, message=message, data=data
        )

    @staticmethod
    def err(code: str, message: str, *, data: Any | None = None) -> WorkbenchToolEnvelope:
        return WorkbenchToolEnvelope(
            code=code, message=message, data=data if data is not None else {}
        )


def wb_ok(data: Any, *, message: str = "success") -> dict[str, Any]:
    """成功：``code`` 恒为 ``OK``。"""
    return WorkbenchEnvelopeFactory.ok(data, message=message).as_tool_dict()


def wb_err(code: str, message: str, *, data: Any | None = None) -> dict[str, Any]:
    """失败或非 OK：``data`` 无额外载荷时为 ``{}``。"""
    return WorkbenchEnvelopeFactory.err(code, message, data=data).as_tool_dict()


class IWorkbenchToolNameFilter(ABC):
    """工具名过滤策略（可替换实现，便于单测）。"""

    @abstractmethod
    def filter_names(self, names: list[str]) -> list[str]:
        raise NotImplementedError


class WorkbenchBuiltinToolFilter(IWorkbenchToolNameFilter):
    """去掉工作台内置工具名，避免子调用递归挂载编排能力。"""

    def filter_names(self, names: list[str]) -> list[str]:
        out: list[str] = []
        for n in names:
            s = (n or "").strip()
            if not s or s.startswith("workbench_"):
                continue
            out.append(s)
        return out


WORKBENCH_BUILTIN_TOOL_FILTER = WorkbenchBuiltinToolFilter()


def tool_names_from_agent_config_json(config_json: dict[str, Any] | None) -> list[str]:
    """与前端 ``config_json.tool_names`` 对齐，读出后做 ``workbench_*`` 过滤。"""
    if not isinstance(config_json, dict):
        return []
    raw = config_json.get("tool_names")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for x in raw:
        if isinstance(x, str) and x.strip():
            names.append(x.strip())
        elif isinstance(x, bool):
            continue
        elif isinstance(x, (int, float)):
            names.append(str(int(x)))
    return WORKBENCH_BUILTIN_TOOL_FILTER.filter_names(names)
