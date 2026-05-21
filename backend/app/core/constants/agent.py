"""Agent **HTTP 层**可调参数：与网关、流式错误码等横切约定对齐。

流式事件 `type` / `progress.stage` / 默认文案见 ``app.agent.kernel.spec``。
"""

from enum import StrEnum


class AgentSseErrorCode(StrEnum):
    """流式 `type=error` 事件的稳定 `code`。"""

    EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"


# 长时间无事件时插入 ping 的间隔（秒），用于避免网关/反代空闲超时断连
AGENT_SSE_PING_INTERVAL_SEC = 60.0

# LangGraph 单轮对话最大步数（含模型节点与工具节点）；默认可被 ``Settings.agent_graph_recursion_limit``（环境变量 ``AGENT_GRAPH_RECURSION_LIMIT``）覆盖
AGENT_GRAPH_RECURSION_LIMIT = 100

# workbench_invoke_sub_agent 写回工具消息的 JSON 内正文/结构化结果字符上限（与窗内历史、parent_messages 截断策略配合）
WORKBENCH_SUB_AGENT_TOOL_RESULT_MAX_CHARS = 32_000

# 子 Agent 嵌套观测（sub_tool_history / sub_process_trace）写入父工具返回时的总预算（超出则裁条目或截断正文）
WORKBENCH_SUB_AGENT_NESTED_OBSERVABILITY_MAX_CHARS = 24_000

# workbench_invoke_sub_agents_parallel 单次允许的子任务数上限（与 asyncio.gather 并发度一致）
WORKBENCH_PARALLEL_MAX_TASKS = 8
