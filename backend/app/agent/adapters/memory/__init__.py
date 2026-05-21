"""LangChain ``trim_messages`` 封装：对窗内消息列表做 token 预算裁剪（见 ``lc_message_trim``）。"""

from app.agent.adapters.memory.lc_message_trim import apply_max_history_token_budget

__all__ = [
    "apply_max_history_token_budget",
]
