from typing import Literal

from pydantic import BaseModel, Field

InfraStatus = Literal["ok", "error", "disabled"]


class HealthData(BaseModel):
    status: str = Field(description="服务状态")
    uptime_seconds: float = Field(description="进程已运行秒数")
    database_status: str = Field(description="主数据库连接状态")


class InfraComponentStatus(BaseModel):
    key: str = Field(description="组件标识")
    label: str = Field(description="展示名")
    status: InfraStatus = Field(description="ok=可用，error=不可用，disabled=未启用")
    detail: str | None = Field(default=None, description="补充说明")
    address: str | None = Field(
        default=None,
        description="脱敏后的连接地址或端点（不含密钥），未配置可为 null",
    )


class TokenUsageDayPoint(BaseModel):
    date: str = Field(description="自然日 YYYY-MM-DD")
    user_tokens: int = Field(
        ge=0,
        description="当日 role=user 且 tokens 非空行的合计（含 tiktoken 估算）",
    )
    assistant_tokens: int = Field(
        ge=0,
        description="当日 role=assistant 且 tokens 非空行的合计",
    )
    total_tokens: int = Field(
        ge=0,
        description="user_tokens + assistant_tokens（兼容仅读取总量的客户端）",
    )


class MonitorDashboardData(BaseModel):
    health: HealthData
    infrastructure: list[InfraComponentStatus] = Field(description="中间件与外部依赖探活")
    token_usage: list[TokenUsageDayPoint] = Field(
        description="按自然日填充的序列（无数据日为 0）",
    )
    token_usage_grand_total: int = Field(
        ge=0,
        description="所选区间内 user + assistant tokens 合计",
    )
