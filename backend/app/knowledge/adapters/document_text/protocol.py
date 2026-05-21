"""文档文本提取处理器协议（可扩展注册）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentTextHandler(Protocol):
    """单种或一类 MIME/后缀的提取策略。"""

    #: 用于日志与错误信息
    name: str
    #: 越小越先匹配（互斥规则下应保证唯一命中）
    priority: int

    def matches(self, mime: str, filename: str) -> bool:
        """``mime`` 已小写规范化；``filename`` 为原始文件名（可能无后缀）。"""
        ...

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        """返回 UTF-8 语义下的纯文本（可含换行）；无内容时返回 ``\"\"``。"""
        ...
