"""对话持久化子域的可预期错误；服务层抛出，HTTP 层映射为 4xx。"""


class ConversationDomainError(Exception):
    """基类：携带建议 HTTP 状态码（默认 400）。"""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        self.status_code = status_code
        super().__init__(message)


class ReferencedAgentNotFoundError(ConversationDomainError):
    """创建会话时引用的 ``agent_entity`` 不存在。"""

    def __init__(self) -> None:
        super().__init__("Agent 不存在", status_code=400)


class SessionNotFoundError(ConversationDomainError):
    """会话不存在或已软删除。"""

    def __init__(self) -> None:
        super().__init__("对话会话不存在或已删除", status_code=400)


class SessionNotWritableError(ConversationDomainError):
    """会话存在但非活跃（已关闭），不可写入。"""

    def __init__(self) -> None:
        super().__init__("对话会话已关闭，无法写入", status_code=400)


class SessionAgentMismatchError(ConversationDomainError):
    """会话与给定 ``agent_id`` 不一致。"""

    def __init__(self) -> None:
        super().__init__("对话会话与 agent_id 不匹配", status_code=400)
