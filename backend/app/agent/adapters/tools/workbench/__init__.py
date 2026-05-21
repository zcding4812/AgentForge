"""工作台内置工具包：对外导出 ``WORKBENCH_TOOL_INSTANCES`` / ``WORKBENCH_TOOL_NAMES`` 与工厂外观。

实现分布于同目录 ``handlers``、``resource_services``、``sub_agent_invoke``、``tool_support`` 等模块。
"""

from __future__ import annotations

from app.agent.adapters.tools.workbench.handlers import WorkbenchToolFactory, WorkbenchToolsFacade

WORKBENCH_TOOL_INSTANCES = WorkbenchToolsFacade.tool_instances()
WORKBENCH_TOOL_NAMES = WorkbenchToolsFacade.tool_names()

__all__ = [
    "WORKBENCH_TOOL_INSTANCES",
    "WORKBENCH_TOOL_NAMES",
    "WorkbenchToolFactory",
    "WorkbenchToolsFacade",
]
