"""LangGraph 进程内 ``MemorySaver``：供各策略 ``compile(checkpointer=...)`` 与门面清理线程共用。

**事实来源（SSOT）**：对话消息与滑动窗口仍以 **MySQL 会话表** 为准，由 ``prepare_invoke`` /
``MessageBuilder`` 每轮注入完整 ``messages``。

**职责**：提供进程级单例 ``checkpointer``。每轮 ``thread_id``、``config``、``adelete_thread`` 由
``AgentRunFacade`` 编排（见 ``application/facade.py`` 中 ``_resolve_checkpoint_thread_id``、
``_build_graph_run_config``、``_cleanup_checkpoint_thread``），避免 ``add_messages`` reducer
在同一线程上**跨 HTTP 请求**累积；checkpoint **不**作为跨请求消息来源。

子调用（如 ``workbench_invoke_sub_agent``）各自新的 ``run_chat_turn`` 会生成新 ``thread_id``。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

_checkpointer: MemorySaver | None = None


def get_agent_checkpointer() -> MemorySaver:
    """进程级单例；多 worker 进程各有一份内存实例。"""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer
