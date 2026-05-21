"""Agent 资源上的记忆参数：从 ``agent_entity.config_json.memory`` 解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_HISTORY_ROUNDS_CAP = 20
DEFAULT_ALLOW_CLIENT_CHAT_HISTORY = False
DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD = 0.4
DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT = 0.7
DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST = 20
DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST = 1000
# 用于「历史 token / 上下文窗口」比例；与具体挂载模型不一致时可在工作区覆盖
DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS = 8192
# 分层滚动摘要：尾窗原文轮数、每批压缩轮数（可配置）
DEFAULT_MIN_TAIL_RAW_ROUNDS = 6
DEFAULT_COMPRESS_BATCH_ROUNDS = 3

_CAP_MIN = 1
_CAP_MAX = 200
_RATIO_MIN = 0.05
_RATIO_MAX = 0.95
_ROUNDS_MIN = 1
_ROUNDS_MAX = 500
_WB_MIN = 1
_WB_MAX = 64
_TOKENS_MIN = 100
_TOKENS_MAX = 500_000
_CTX_WINDOW_MIN = 512
_CTX_WINDOW_MAX = 2_000_000


@dataclass(frozen=True, slots=True)
class AgentMemorySettings:
    """与前端工作区 ``config_json.memory`` 对齐。"""

    max_history_rounds_cap: int
    allow_client_chat_history: bool
    #: 滚动摘要专用 ``sys_model.id``；``None`` 表示未指定（后续可与主模型一致或走全局默认）
    summarization_sys_model_id: int | None
    #: 历史总 token 相对上下文窗口超过该比例时，可考虑触发摘要（常规策略，与业务调度配合）
    summary_context_ratio_threshold: float
    #: 超过该比例时为「紧急」摘要调度
    summary_context_ratio_urgent: float
    #: 自上次摘要成功以来，至少新增完整对话轮数（与 token 条件二选一由上层策略决定）
    summary_min_rounds_since_last: int
    #: 自上次摘要成功以来，至少新增 token 数（估算）
    summary_min_tokens_since_last: int
    #: 主模型上下文窗口 token 上限（用于与 ``history_total_tokens`` 求比例）
    summary_context_window_tokens: int
    #: 存在非空滚动摘要时，注入模型的 **最近多少轮** 保留原文（尾窗）
    min_tail_raw_rounds: int
    #: 链式摘要：每批合并多少 **user 轮次** 与上一摘要一并压缩
    compress_batch_rounds: int


class AgentMemorySettingsParser:
    """解析 ``config_json`` 中的 ``memory`` 块；可注入单测或替换策略时子类化。"""

    def parse(self, config_json: dict[str, Any] | None) -> AgentMemorySettings:
        """
        缺省或非法时回退默认值。

        约定形状::

            {
              "memory": {
                "max_history_rounds_cap": 50,
                "allow_client_chat_history": false,
                "summarization_sys_model_id": null,
                "summary_context_ratio_threshold": 0.4,
                "summary_context_ratio_urgent": 0.7,
                "summary_min_rounds_since_last": 20,
                "summary_min_tokens_since_last": 1000,
                "summary_context_window_tokens": 8192,
                "min_tail_raw_rounds": 6,
                "compress_batch_rounds": 3
              }
            }
        """
        if not config_json or not isinstance(config_json, dict):
            return self._defaults()
        block = config_json.get("memory")
        if not isinstance(block, dict):
            return self._defaults()
        return AgentMemorySettings(
            max_history_rounds_cap=self._parse_cap(block.get("max_history_rounds_cap")),
            allow_client_chat_history=self._parse_allow(block.get("allow_client_chat_history")),
            summarization_sys_model_id=self._parse_optional_sys_model_id(
                block.get("summarization_sys_model_id"),
            ),
            summary_context_ratio_threshold=self._parse_ratio(
                block.get("summary_context_ratio_threshold"),
                DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD,
            ),
            summary_context_ratio_urgent=self._parse_ratio(
                block.get("summary_context_ratio_urgent"),
                DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT,
            ),
            summary_min_rounds_since_last=self._parse_positive_int(
                block.get("summary_min_rounds_since_last"),
                DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST,
                _ROUNDS_MIN,
                _ROUNDS_MAX,
            ),
            summary_min_tokens_since_last=self._parse_positive_int(
                block.get("summary_min_tokens_since_last"),
                DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST,
                _TOKENS_MIN,
                _TOKENS_MAX,
            ),
            summary_context_window_tokens=self._parse_positive_int(
                block.get("summary_context_window_tokens"),
                DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS,
                _CTX_WINDOW_MIN,
                _CTX_WINDOW_MAX,
            ),
            min_tail_raw_rounds=self._parse_positive_int(
                block.get("min_tail_raw_rounds"),
                DEFAULT_MIN_TAIL_RAW_ROUNDS,
                _WB_MIN,
                _WB_MAX,
            ),
            compress_batch_rounds=self._parse_positive_int(
                block.get("compress_batch_rounds"),
                DEFAULT_COMPRESS_BATCH_ROUNDS,
                _WB_MIN,
                _WB_MAX,
            ),
        )

    @staticmethod
    def _defaults() -> AgentMemorySettings:
        return AgentMemorySettings(
            max_history_rounds_cap=DEFAULT_MAX_HISTORY_ROUNDS_CAP,
            allow_client_chat_history=DEFAULT_ALLOW_CLIENT_CHAT_HISTORY,
            summarization_sys_model_id=None,
            summary_context_ratio_threshold=DEFAULT_SUMMARY_CONTEXT_RATIO_THRESHOLD,
            summary_context_ratio_urgent=DEFAULT_SUMMARY_CONTEXT_RATIO_URGENT,
            summary_min_rounds_since_last=DEFAULT_SUMMARY_MIN_ROUNDS_SINCE_LAST,
            summary_min_tokens_since_last=DEFAULT_SUMMARY_MIN_TOKENS_SINCE_LAST,
            summary_context_window_tokens=DEFAULT_SUMMARY_CONTEXT_WINDOW_TOKENS,
            min_tail_raw_rounds=DEFAULT_MIN_TAIL_RAW_ROUNDS,
            compress_batch_rounds=DEFAULT_COMPRESS_BATCH_ROUNDS,
        )

    @staticmethod
    def _clamp_cap(value: int) -> int:
        return max(_CAP_MIN, min(_CAP_MAX, value))

    def _parse_cap(self, raw: Any) -> int:
        if isinstance(raw, bool):
            return DEFAULT_MAX_HISTORY_ROUNDS_CAP
        if isinstance(raw, int):
            return self._clamp_cap(raw)
        if isinstance(raw, float):
            return self._clamp_cap(int(raw))
        return DEFAULT_MAX_HISTORY_ROUNDS_CAP

    @staticmethod
    def _parse_allow(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        return DEFAULT_ALLOW_CLIENT_CHAT_HISTORY

    @staticmethod
    def _parse_optional_sys_model_id(raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int) and raw >= 1:
            return raw
        if isinstance(raw, float) and raw >= 1:
            return int(raw)
        return None

    @staticmethod
    def _parse_ratio(raw: Any, default: float) -> float:
        if raw is None:
            return default
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return default
        if not v == v:  # NaN
            return default
        return max(_RATIO_MIN, min(_RATIO_MAX, v))

    @staticmethod
    def _parse_positive_int(raw: Any, default: int, lo: int, hi: int) -> int:
        if raw is None:
            return default
        if isinstance(raw, bool):
            return default
        if isinstance(raw, int):
            return max(lo, min(hi, raw))
        if isinstance(raw, float):
            return max(lo, min(hi, int(raw)))
        return default


def parse_agent_memory_settings(config_json: dict[str, Any] | None) -> AgentMemorySettings:
    """模块级便捷函数，等价于 ``AgentMemorySettingsParser().parse(...)``。"""
    return AgentMemorySettingsParser().parse(config_json)
