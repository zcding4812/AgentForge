"""Agent 对话会话列表与状态常量（单租户持久化 API）。"""

# 仅按「最近 N 条」拉取时（如组装轮次）的默认上界；显式传入 ``limit_last`` 时以此类调用为准
LIST_MESSAGES_DEFAULT_LIMIT_LAST: int = 2_500

# 会话详情 GET：消息分页默认每页条数；第 1 页为时间轴上最近一批（页内按时间正序）
CONVERSATION_DETAIL_MESSAGES_DEFAULT_PAGE_SIZE: int = 10
CONVERSATION_DETAIL_MESSAGES_MAX_PAGE_SIZE: int = 100

CONVERSATION_LIST_DEFAULT_PAGE_SIZE: int = 20
CONVERSATION_LIST_MAX_PAGE_SIZE: int = 100

# 与表 ``status`` 一致：1=活跃，0=已关闭
CONVERSATION_SESSION_ACTIVE: int = 1
CONVERSATION_SESSION_CLOSED: int = 0

# ``conversation_session.summary_job_status``：滚动摘要异步任务
SUMMARY_JOB_STATUS_IDLE: int = 0
SUMMARY_JOB_STATUS_PENDING: int = 1
SUMMARY_JOB_STATUS_RUNNING: int = 2

# 助手消息 ``metadata.tool_history`` 中单条工具返回正文字符上限（超出截断，避免单行 metadata 过大）
TOOL_HISTORY_METADATA_MAX_RESULT_CHARS: int = 48_000
