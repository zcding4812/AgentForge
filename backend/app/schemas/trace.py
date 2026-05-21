from typing import Any

from pydantic import BaseModel, Field

from app.schemas.response import PageMeta


class TraceNode(BaseModel):
    span_id: str = Field(description="Span 唯一 ID")
    parent_span_id: str | None = Field(default=None, description="父 Span ID")
    name: str = Field(description="节点名称")
    span_type: str = Field(description="节点类型")
    depth: int = Field(ge=0, description="节点深度，根节点为 0")
    start_ms: int = Field(ge=0, description="相对链路起点的开始时间（毫秒）")
    duration_ms: int = Field(ge=1, description="节点耗时（毫秒）")
    status: str = Field(description="节点状态：ok/error")
    attempt: int = Field(ge=1, description="执行序号")
    component: str | None = Field(default=None, description="组件名（如 agent-graph）")
    error_message: str | None = Field(default=None, description="失败/超时时的错误摘要")
    tags: dict[str, Any] | None = Field(default=None, description="落库时的 span tags（JSON）")


class TraceSummary(BaseModel):
    trace_id: str = Field(description="链路 ID")
    total_duration_ms: int = Field(ge=0, description="链路总耗时（毫秒）")
    node_count: int = Field(ge=0, description="节点总数")
    success_count: int = Field(ge=0, description="成功节点数")
    error_count: int = Field(ge=0, description="异常节点数")
    slow_count: int = Field(ge=0, description="慢节点数（>1000ms）")


class TraceNodeLog(BaseModel):
    log_id: int = Field(description="trace_span_logs 表主键，同一链路内按此排序即可")
    span_id: str = Field(description="所属 Span ID")
    log_level: str = Field(description="日志级别")
    event_name: str = Field(description="日志事件名")
    message: str = Field(description="日志内容")
    occurred_at: str = Field(description="发生时间")
    payload: dict[str, Any] | None = Field(
        default=None,
        description="结构化附加字段（与落库 trace_span_logs.payload 一致）",
    )


class TraceDetailData(BaseModel):
    summary: TraceSummary
    nodes: list[TraceNode]
    logs: list[TraceNodeLog] = Field(
        default_factory=list,
        description="链路内全部 Span 日志；同一 span_id 可出现多条，按 log_id 排序",
    )


class TraceListItem(BaseModel):
    trace_name: str = Field(description="链路名称")
    trace_id: str = Field(description="链路 ID（W3C trace id，与单次请求根链路对应）")
    session_id: str = Field(description="用户会话 ID，无则占位符")
    agent_task_id: str = Field(description="Agent/异步任务 ID，无则占位符")
    duration_ms: int = Field(ge=0, description="链路耗时（毫秒）")
    status: str = Field(description="链路状态：success/failed/running")
    executed_at: str = Field(description="执行时间")


class TraceListStats(BaseModel):
    total_count: int = Field(ge=0, description="链路总数")
    success_count: int = Field(ge=0, description="成功数")
    failed_count: int = Field(ge=0, description="失败数")
    running_count: int = Field(ge=0, description="运行中数")
    success_rate: float = Field(ge=0, le=100, description="成功率，百分比")
    avg_duration_ms: int = Field(ge=0, description="平均耗时（毫秒）")
    p95_duration_ms: int = Field(ge=0, description="P95 耗时（毫秒）")


class TraceListData(BaseModel):
    items: list[TraceListItem]
    stats: TraceListStats
    meta: PageMeta = Field(description="运行列表分页（默认每页 10 条）")
