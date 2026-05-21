from datetime import datetime

from app.core.constants import (
    TRACE_LIST_DEFAULT_PAGE_SIZE,
    TRACE_LIST_MAX_PAGE_SIZE,
    TraceSpanStatus,
)
from app.core.logger import get_logger
from app.core.tracing.engine import normalize_span_status, normalize_trace_run_status
from app.models.trace_mod import TraceRun, TraceSpan, TraceSpanLog
from app.repositories.trace_repo import TraceRepository
from app.schemas.response import PageMeta
from app.schemas.trace import (
    TraceDetailData,
    TraceListData,
    TraceListItem,
    TraceListStats,
    TraceNode,
    TraceNodeLog,
    TraceSummary,
)

logger = get_logger(__name__)


class TraceService:
    """链路列表与详情编排；无仓储会话参数，依赖 :class:`TraceRepository` 静态入口。"""

    async def get_trace_list(
        self,
        trace_id: str | None = None,
        *,
        page: int = 1,
        page_size: int = TRACE_LIST_DEFAULT_PAGE_SIZE,
    ) -> TraceListData:
        """分页列表 + 聚合统计；``trace_id`` 非空时按链路过滤。"""
        page = max(1, page)
        page_size = min(max(1, page_size), TRACE_LIST_MAX_PAGE_SIZE)
        total_rows = await self._count_db_trace_runs(trace_id=trace_id)
        offset = (page - 1) * page_size
        items = await self._build_db_list_items(
            trace_id=trace_id,
            offset=offset,
            limit=page_size,
        )
        (
            total_count,
            success_count,
            failed_count,
            running_count,
            avg_duration_ms,
            p95_duration_ms,
        ) = await self._aggregate_db_stats(trace_id=trace_id)
        success_rate = round((success_count / total_count) * 100, 2) if total_count else 0.0

        stats = TraceListStats(
            total_count=total_count,
            success_count=success_count,
            failed_count=failed_count,
            running_count=running_count,
            success_rate=success_rate,
            avg_duration_ms=avg_duration_ms,
            p95_duration_ms=p95_duration_ms,
        )
        meta = PageMeta(page=page, page_size=page_size, total=total_rows)
        return TraceListData(items=items, stats=stats, meta=meta)

    async def get_trace_detail(self, trace_id: str) -> TraceDetailData:
        """单条 trace 的节点树与日志。"""
        nodes = await self._build_db_detail_nodes(trace_id=trace_id)
        logs = await self._build_db_detail_logs(trace_id=trace_id)
        total_duration_ms = max((node.start_ms + node.duration_ms for node in nodes), default=0)
        success_count = sum(1 for node in nodes if node.status == TraceSpanStatus.OK.value)
        error_count = len(nodes) - success_count
        slow_count = sum(1 for node in nodes if node.duration_ms > 1000)
        summary = TraceSummary(
            trace_id=trace_id,
            total_duration_ms=total_duration_ms,
            node_count=len(nodes),
            success_count=success_count,
            error_count=error_count,
            slow_count=slow_count,
        )
        return TraceDetailData(summary=summary, nodes=nodes, logs=logs)

    async def _build_db_list_items(
        self,
        trace_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = TRACE_LIST_DEFAULT_PAGE_SIZE,
    ) -> list[TraceListItem]:
        try:
            runs = await TraceRepository.list_trace_runs(
                trace_id=trace_id,
                offset=offset,
                limit=limit,
            )
            return [self._to_trace_list_item(run) for run in runs]
        except Exception:
            logger.warning("Trace list query failed, return empty list.", exc_info=True)
            return []

    async def _count_db_trace_runs(self, trace_id: str | None = None) -> int:
        try:
            return await TraceRepository.count_trace_runs(trace_id=trace_id)
        except Exception:
            logger.warning("Trace count query failed, return zero.", exc_info=True)
            return 0

    async def _aggregate_db_stats(
        self, trace_id: str | None = None
    ) -> tuple[int, int, int, int, int, int]:
        try:
            return await TraceRepository.aggregate_trace_run_stats(trace_id=trace_id)
        except Exception:
            logger.warning("Trace aggregate stats failed, return zeros.", exc_info=True)
            return (0, 0, 0, 0, 0, 0)

    async def _build_db_detail_nodes(self, trace_id: str) -> list[TraceNode]:
        try:
            spans = await TraceRepository.list_trace_spans(trace_id=trace_id)
            if not spans:
                return []
            root_started_at = min(span.started_at for span in spans)
            by_id = {s.span_id: s for s in spans}
            ordered = self._order_spans_tree_first(spans, by_id=by_id)
            raw_nodes = [
                self._to_trace_node(
                    span,
                    root_started_at=root_started_at,
                    depth=self._tree_depth(span, by_id=by_id),
                )
                for span in ordered
            ]
            return self._extend_durations_to_cover_subtree_end(raw_nodes)
        except Exception:
            logger.warning("Trace detail spans query failed, return empty nodes.", exc_info=True)
            return []

    @staticmethod
    def _tree_depth(span: TraceSpan, *, by_id: dict[str, TraceSpan]) -> int:
        depth = 0
        cur = span
        for _ in range(len(by_id) + 1):
            if not cur.parent_span_id or cur.parent_span_id not in by_id:
                return depth
            depth += 1
            cur = by_id[cur.parent_span_id]
        return depth

    @staticmethod
    def _order_spans_tree_first(
        spans: list[TraceSpan],
        *,
        by_id: dict[str, TraceSpan],
    ) -> list[TraceSpan]:
        children: dict[str | None, list[TraceSpan]] = {}
        for s in spans:
            pid = s.parent_span_id
            if pid is not None and pid not in by_id:
                pid = None
            children.setdefault(pid, []).append(s)
        for lst in children.values():
            lst.sort(key=lambda x: (x.started_at, x.id))

        ordered: list[TraceSpan] = []
        seen: set[str] = set()

        def dfs(span: TraceSpan) -> None:
            if span.span_id in seen:
                return
            seen.add(span.span_id)
            ordered.append(span)
            for ch in children.get(span.span_id, []):
                dfs(ch)

        for root in children.get(None, []):
            dfs(root)
        for s in spans:
            if s.span_id not in seen:
                ordered.append(s)
        return ordered

    async def _build_db_detail_logs(self, trace_id: str) -> list[TraceNodeLog]:
        try:
            logs = await TraceRepository.list_span_logs_by_trace(trace_id=trace_id)
            return [self._to_trace_node_log(item) for item in logs]
        except Exception:
            logger.warning("Trace detail logs query failed, return empty logs.", exc_info=True)
            return []

    def _to_trace_list_item(self, run: TraceRun) -> TraceListItem:
        return TraceListItem(
            trace_name=run.trace_name,
            trace_id=run.trace_id,
            session_id=run.session_id or "-",
            agent_task_id=run.agent_task_id or "-",
            duration_ms=run.duration_ms or 0,
            status=normalize_trace_run_status(run.status).value,
            executed_at=self._format_datetime(run.started_at),
        )

    def _extend_durations_to_cover_subtree_end(self, nodes: list[TraceNode]) -> list[TraceNode]:
        """
        让父节点 ``duration_ms`` 至少覆盖「自身 + 全部子孙」在 ``start_ms`` 轴上的最远结束点。

        落库侧 ``duration_ms`` 来自各 span 的 ``perf_counter`` 区间，而 ``start_ms`` 来自各
        ``started_at`` 与根的时间差（墙钟）。混用后可能出现「子条右端超出父条」的甘特错位；
        此处仅在展示层把父条拉长到包住子树时间窗（不修改 DB）。
        """
        if len(nodes) <= 1:
            return nodes
        by_id = {n.span_id: n for n in nodes}
        child_ids: dict[str | None, list[str]] = {}
        for n in nodes:
            pid = n.parent_span_id
            if pid is not None and pid not in by_id:
                pid = None
            child_ids.setdefault(pid, []).append(n.span_id)

        memo: dict[str, int] = {}

        def subtree_max_end_ms(span_id: str) -> int:
            if span_id in memo:
                return memo[span_id]
            n = by_id[span_id]
            end = n.start_ms + n.duration_ms
            for cid in child_ids.get(span_id, []):
                end = max(end, subtree_max_end_ms(cid))
            memo[span_id] = end
            return end

        out: list[TraceNode] = []
        for n in nodes:
            smax = subtree_max_end_ms(n.span_id)
            need_ms = smax - n.start_ms
            if need_ms > n.duration_ms:
                out.append(n.model_copy(update={"duration_ms": max(need_ms, 1)}))
            else:
                out.append(n)
        return out

    def _to_trace_node(
        self,
        span: TraceSpan,
        root_started_at: datetime,
        *,
        depth: int | None = None,
    ) -> TraceNode:
        start_ms = int((span.started_at - root_started_at).total_seconds() * 1000)
        depth_val = depth if depth is not None else max(span.depth or 0, 0)
        return TraceNode(
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.span_name,
            span_type=span.span_type.upper(),
            depth=max(depth_val, 0),
            start_ms=max(start_ms, 0),
            duration_ms=max(span.duration_ms or 1, 1),
            status=normalize_span_status(span.status).value,
            attempt=span.attempt or 1,
            component=span.component,
            error_message=span.error_message,
            tags=span.tags if isinstance(span.tags, dict) else None,
        )

    def _to_trace_node_log(self, log_item: TraceSpanLog) -> TraceNodeLog:
        payload = log_item.payload if isinstance(log_item.payload, dict) else None
        return TraceNodeLog(
            log_id=log_item.id,
            span_id=log_item.span_id,
            log_level=log_item.log_level,
            event_name=log_item.event_name,
            message=log_item.message,
            occurred_at=self._format_datetime(log_item.occurred_at),
            payload=payload,
        )

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if value is None:
            return "-"
        return value.strftime("%Y/%m/%d %H:%M:%S")
