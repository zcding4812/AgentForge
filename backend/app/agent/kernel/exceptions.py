"""Agent 领域异常（内核定义，应用层抛出，HTTP/SSE 映射见 ``core.exceptions``）。"""


class AgentExecutionError(Exception):
    """Agent 执行统一异常。"""
