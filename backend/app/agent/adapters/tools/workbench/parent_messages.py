"""解析父工作台传入的 ``parent_messages`` JSON 为 ``ChatHistoryTurn`` 列表。

策略模式拆分原生 / OpenAI 格式；工厂按首条消息选择策略；对外保留
``parse_parent_messages_json`` 与 ``ParentMessagesParseStrategy``。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.agent.adapters.tools.workbench.context import WorkbenchConstants
from app.schemas.agent import ChatHistoryTurn


class ParentMessageParserConfig:
    MAX_TURNS = WorkbenchConstants.MAX_PARENT_MESSAGE_TURNS
    ADAPTER = TypeAdapter(list[ChatHistoryTurn])


class ParentMessageUtils:
    @staticmethod
    def content_to_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def role_token(row: dict[str, Any]) -> str:
        return str(row.get("role", "")).strip().lower()

    @staticmethod
    def is_openai_style_row(row: dict[str, Any]) -> bool:
        if "role" not in row:
            return False
        if "user" in row and "assistant" in row:
            return False
        return True


class IParentMessageParser(ABC):
    @abstractmethod
    def parse(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError


class NativeChatHistoryParser(IParentMessageParser):
    def parse(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not data:
            return []
        for idx, x in enumerate(data):
            if not isinstance(x, dict):
                raise ValueError(f"parent_messages[{idx}] 须为 JSON 对象")
        return data


class OpenAIChatMessageParser(IParentMessageParser):
    def parse(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dict_rows = [x for x in data if isinstance(x, dict)]
        if len(dict_rows) != len(data):
            raise ValueError("parent_messages 数组元素须均为 JSON 对象")
        return self._convert_to_chat_turns(dict_rows)

    def _convert_to_chat_turns(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        i = 0
        n = len(rows)
        while i < n:
            row = rows[i]
            if not isinstance(row, dict):
                raise ValueError("parent_messages 数组元素须为 JSON 对象")
            role = ParentMessageUtils.role_token(row)
            if role == "system":
                i += 1
                continue
            if role != "user":
                raise ValueError(
                    "OpenAI 风格 parent_messages 须按 user → assistant/tool 分段；"
                    f"在索引 {i} 处期望 role=user，实际为 {role!r}"
                )
            u = ParentMessageUtils.content_to_str(row.get("content")).strip()
            if not u:
                raise ValueError("role=user 的 content 不能为空")
            i += 1
            asst_parts: list[str] = []
            tool_results: list[dict[str, Any]] = []
            while i < n:
                nxt = rows[i]
                if not isinstance(nxt, dict):
                    raise ValueError("parent_messages 数组元素须为 JSON 对象")
                r2 = ParentMessageUtils.role_token(nxt)
                if r2 == "user":
                    break
                if r2 == "assistant":
                    asst_parts.append(ParentMessageUtils.content_to_str(nxt.get("content")))
                    i += 1
                    continue
                if r2 == "tool":
                    tool_results.append(
                        {
                            "tool_call_id": str(
                                nxt.get("tool_call_id") or nxt.get("name") or "tool"
                            ).strip()
                            or "tool",
                            "name": str(nxt.get("name") or "tool").strip() or "tool",
                            "content": ParentMessageUtils.content_to_str(nxt.get("content")),
                        }
                    )
                    i += 1
                    continue
                raise ValueError(
                    f"parent_messages 不支持的 role: {r2!r}（支持 user/assistant/tool/system）"
                )
            a = "".join(asst_parts).strip()
            if not a:
                a = "（无）"
            out.append(
                {
                    "user": u,
                    "assistant": a,
                    "tool_calls": [],
                    "tool_results": tool_results,
                }
            )
        return out


class ParentMessageParserFactory:
    @staticmethod
    def get_parser(data: list[dict[str, Any]]) -> IParentMessageParser:
        if not data:
            return NativeChatHistoryParser()
        first = data[0]
        if not isinstance(first, dict):
            raise ValueError("parent_messages 数组元素须为 JSON 对象")
        if "user" in first and "role" not in first:
            return NativeChatHistoryParser()
        if ParentMessageUtils.is_openai_style_row(first):
            return OpenAIChatMessageParser()
        raise ValueError(
            "parent_messages 无法识别：须为 ChatHistoryTurn 数组（键 user、assistant）"
            "，或为 OpenAI 消息数组（键 role、content）"
        )


class ParentMessageParserService:
    _instance: ParentMessageParserService | None = None

    def __new__(cls) -> ParentMessageParserService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def parse_json(self, raw: str | None) -> list[ChatHistoryTurn]:
        if raw is None:
            return []
        s = str(raw).strip()
        if not s:
            return []
        try:
            raw_data: Any = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError("parent_messages 不是合法 JSON") from e
        self._validate_raw_data(raw_data)
        data_list: list[dict[str, Any]] = raw_data  # type: ignore[assignment]
        parser = ParentMessageParserFactory.get_parser(data_list)
        normalized = parser.parse(data_list)
        try:
            result = ParentMessageParserConfig.ADAPTER.validate_python(normalized)
        except ValidationError as e:
            raise ValueError(f"parent_messages 格式不符合 ChatHistoryTurn: {e}") from e
        cap = ParentMessageParserConfig.MAX_TURNS
        if len(result) > cap:
            return list(result[-cap:])
        return result

    @staticmethod
    def _validate_raw_data(data: Any) -> None:
        if not isinstance(data, list):
            raise ValueError("parent_messages 须为 JSON 数组")
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"parent_messages[{idx}] 须为 JSON 对象")


DEFAULT_PARSER_SERVICE = ParentMessageParserService()


class IParentMessagesParseStrategy(ABC):
    @abstractmethod
    def parse(self, raw: str | None) -> list[ChatHistoryTurn]:
        raise NotImplementedError


class ParentMessagesParseStrategy(IParentMessagesParseStrategy):
    _service = DEFAULT_PARSER_SERVICE

    @classmethod
    def default(cls) -> ParentMessagesParseStrategy:
        return _default_parent_messages_strategy

    def parse(self, raw: str | None) -> list[ChatHistoryTurn]:
        return self._service.parse_json(raw)


_default_parent_messages_strategy = ParentMessagesParseStrategy()


def parse_parent_messages_json(raw: str | None) -> list[ChatHistoryTurn]:
    """``raw`` 为 JSON 数组：ChatHistoryTurn 或 OpenAI 风格消息数组。"""
    return DEFAULT_PARSER_SERVICE.parse_json(raw)
