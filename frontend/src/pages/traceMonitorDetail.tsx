/**
 * 链路详情：像素时间轴泳道 + Card 顶栏信息 + 类型筛选 + 轨道悬停详情（Portal）
 * 样式由 `ConsolePages` 入口已加载的 `pages.css` 提供。
 */
import { Button, Card, Drawer, Empty, Select, Space, Tag, Typography } from 'antd'
import { createPortal } from 'react-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { readJsonBody } from '../api/parseEnvelope'

const { Text } = Typography

type TraceNode = {
  span_id: string
  parent_span_id?: string | null
  name: string
  span_type: string
  depth: number
  start_ms: number
  duration_ms: number
  status: string
  attempt: number
  component?: string | null
  error_message?: string | null
  tags?: Record<string, unknown> | null
}

type TraceSummary = {
  trace_id: string
  total_duration_ms: number
  node_count: number
  success_count: number
  error_count: number
  slow_count: number
}

type TraceNodeLog = {
  log_id: number
  span_id: string
  log_level: string
  event_name: string
  message: string
  occurred_at: string
  payload?: Record<string, unknown> | null
}

type TraceDetailData = {
  summary: TraceSummary
  nodes: TraceNode[]
  logs?: TraceNodeLog[]
}

function formatTraceDurationMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function traceBarSolidColor(spanType: string, isError: boolean): string {
  if (isError) return '#ef4444'
  const t = spanType.toUpperCase()
  const map: Record<string, string> = {
    FASTAPI: '#2563eb',
    API: '#0284c7',
    SERVICE: '#16a34a',
    DB: '#ea580c',
    CUSTOM: '#7c3aed',
  }
  return map[t] ?? '#14b8a6'
}

function traceStatusIconEmoji(status: string): string {
  const s = status.toLowerCase()
  if (s === 'ok' || s === 'success' || s === 'running') return '🟢'
  if (s === 'timeout') return '🟡'
  return '🔴'
}

function traceStatusShortLabel(status: string): string {
  const s = status.toLowerCase()
  if (s === 'ok' || s === 'success' || s === 'running') return '成功'
  if (s === 'timeout') return '超时'
  return '失败'
}

function formatJsonBlock(value: unknown): string {
  if (value == null) return ''
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function clampTracePopoverPosition(clientX: number, clientY: number) {
  const POP_EST_W = 340
  const POP_EST_H = 280
  const OFFSET_X = 14
  const OFFSET_Y = 10
  const pad = 10
  const vw = window.innerWidth
  const vh = window.innerHeight
  let left = clientX + OFFSET_X
  let top = clientY + OFFSET_Y
  if (left + POP_EST_W > vw - pad) left = vw - POP_EST_W - pad
  if (left < pad) left = pad
  if (top + POP_EST_H > vh - pad) top = clientY - POP_EST_H - OFFSET_Y
  if (top < pad) top = pad
  return { position: 'fixed' as const, left, top, zIndex: 3200 }
}

function TraceSpanHoverDetailBody({
  node,
  traceId,
  pctParent,
}: {
  node: TraceNode
  traceId: string
  pctParent: number | null
}) {
  const endMs = node.start_ms + node.duration_ms
  const tagsJson = formatJsonBlock(node.tags)
  return (
    <div className="trace-popover-detail">
      <div className="trace-popover-row">
        <Text type="secondary">层级</Text>
        <Text>L{node.depth}</Text>
      </div>
      <div className="trace-popover-row">
        <Text type="secondary">类型</Text>
        <Text>{node.span_type}</Text>
      </div>
      {node.component ? (
        <div className="trace-popover-row">
          <Text type="secondary">组件</Text>
          <Text>{node.component}</Text>
        </div>
      ) : null}
      <div className="trace-popover-row">
        <Text type="secondary">状态</Text>
        <Text>
          {traceStatusIconEmoji(node.status)} {traceStatusShortLabel(node.status)}
        </Text>
      </div>
      <div className="trace-popover-row">
        <Text type="secondary">span_id</Text>
        <Text code copyable={{ text: node.span_id }} className="trace-popover-mono">
          {node.span_id}
        </Text>
      </div>
      <div className="trace-popover-row">
        <Text type="secondary">parent</Text>
        <Text code className="trace-popover-mono">
          {node.parent_span_id ?? '—'}
        </Text>
      </div>
      <div className="trace-popover-row">
        <Text type="secondary">时间（相对链路）</Text>
        <Text>
          开始 {node.start_ms}ms → 结束 {endMs}ms，耗时 {node.duration_ms}ms
        </Text>
      </div>
      {pctParent != null ? (
        <div className="trace-popover-row">
          <Text type="secondary">占父节点</Text>
          <Text strong>{pctParent.toFixed(1)}%</Text>
        </div>
      ) : null}
      {node.attempt > 1 ? (
        <div className="trace-popover-row">
          <Text type="secondary">重试</Text>
          <Text>@{node.attempt}</Text>
        </div>
      ) : null}
      {node.error_message ? (
        <div className="trace-popover-row trace-popover-row--block">
          <Text type="secondary">错误</Text>
          <Text type="danger" className="trace-popover-error">
            {node.error_message}
          </Text>
        </div>
      ) : null}
      {tagsJson ? (
        <div className="trace-popover-row trace-popover-row--block">
          <Text type="secondary">tags</Text>
          <pre className="trace-popover-pre">{tagsJson}</pre>
        </div>
      ) : null}
      <div className="trace-popover-row">
        <Text type="secondary">trace_id</Text>
        <Text code copyable={{ text: traceId }} className="trace-popover-mono">
          {traceId}
        </Text>
      </div>
    </div>
  )
}

function treeRailContinues(
  node: TraceNode,
  byId: Map<string, TraceNode>,
  childrenByParent: Map<string, TraceNode[]>,
): boolean[] {
  const continues: boolean[] = []
  const chain: TraceNode[] = []
  let cur: TraceNode | undefined = node
  while (cur) {
    chain.push(cur)
    cur = cur.parent_span_id ? byId.get(cur.parent_span_id) : undefined
  }
  chain.reverse()
  for (let j = 0; j < chain.length - 2; j++) {
    const anc = chain[j]
    const childOnPath = chain[j + 1]
    const sibs = childrenByParent.get(anc.span_id) ?? []
    const isLast = sibs[sibs.length - 1]?.span_id === childOnPath.span_id
    continues.push(!isLast)
  }
  return continues
}

function isLastChild(
  node: TraceNode,
  byId: Map<string, TraceNode>,
  childrenByParent: Map<string, TraceNode[]>,
): boolean {
  if (!node.parent_span_id) return true
  const p = byId.get(node.parent_span_id)
  if (!p) return true
  const sibs = childrenByParent.get(p.span_id) ?? []
  return sibs[sibs.length - 1]?.span_id === node.span_id
}

function buildChildrenByParent(nodes: TraceNode[]): Map<string, TraceNode[]> {
  const m = new Map<string, TraceNode[]>()
  for (const n of nodes) {
    const k = n.parent_span_id ?? ''
    if (!m.has(k)) m.set(k, [])
    m.get(k)!.push(n)
  }
  for (const [, arr] of m) {
    arr.sort((a, b) => a.start_ms - b.start_ms || a.span_id.localeCompare(b.span_id))
  }
  return m
}

function lineageSpanIds(
  hoverId: string,
  nodeById: Map<string, TraceNode>,
  childrenByParent: Map<string, TraceNode[]>,
): Set<string> {
  const s = new Set<string>()
  let cur: TraceNode | undefined = nodeById.get(hoverId)
  while (cur) {
    s.add(cur.span_id)
    cur = cur.parent_span_id ? nodeById.get(cur.parent_span_id) : undefined
  }
  const stack = [hoverId]
  while (stack.length) {
    const id = stack.pop()!
    for (const k of childrenByParent.get(id) ?? []) {
      s.add(k.span_id)
      stack.push(k.span_id)
    }
  }
  return s
}

function axisTicksNice(min: number, max: number): number[] {
  if (max <= min) return [min]
  const span = max - min
  const rough = span / 6
  const pow10 = 10 ** Math.floor(Math.log10(Math.max(rough, 1)))
  const candidates = [1, 2, 5, 10].map((x) => x * pow10)
  let step = candidates[0]
  for (const c of candidates) {
    if (c >= rough) {
      step = c
      break
    }
    step = c
  }
  const ticks: number[] = []
  let t = Math.ceil(min / step) * step
  while (t <= max + step * 0.001) {
    if (t >= min - 0.001 && t <= max) ticks.push(Math.round(t))
    t += step
  }
  if (ticks.length === 0) ticks.push(min)
  return ticks
}

const TRACE_ROW_HEIGHT_PX = 28
const TRACE_BAR_HEIGHT_PX = 6
const TRACE_LANE_GAP_PX = 0
const TRACE_SLOW_MS = 1000
const TRACE_LABEL_COL_PX = 228
const TRACE_FOCUS_PAD_MS = 6

export function TraceMonitorDetailPage() {
  const { traceId } = useParams()
  const [traceDetail, setTraceDetail] = useState<TraceDetailData | null>(null)
  const [error, setError] = useState('')

  const loadTraceDetail = async () => {
    if (!traceId) return
    setError('')
    try {
      const response = await fetch(`/api/traces/${encodeURIComponent(traceId)}`, {
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) throw new Error(`trace detail request failed: ${response.status}`)
      const payload = await readJsonBody<{ data?: TraceDetailData }>(response)
      if (!payload?.data) throw new Error('trace detail payload missing data')
      setTraceDetail(payload.data)
    } catch {
      setTraceDetail(null)
      setError('链路详情加载失败，请稍后重试。')
    }
  }

  useEffect(() => {
    void loadTraceDetail()
    const handleRefresh = () => void loadTraceDetail()
    window.addEventListener('app:refresh', handleRefresh)
    return () => window.removeEventListener('app:refresh', handleRefresh)
  }, [traceId])

  const totalDuration = Math.max(traceDetail?.summary.total_duration_ms ?? 1, 1)
  const nodes = traceDetail?.nodes
  const logsBySpanId = useMemo(() => {
    const m = new Map<string, TraceNodeLog[]>()
    for (const log of traceDetail?.logs ?? []) {
      const list = m.get(log.span_id) ?? []
      list.push(log)
      m.set(log.span_id, list)
    }
    for (const [k, list] of m) {
      list.sort((a, b) => a.log_id - b.log_id)
      m.set(k, list)
    }
    return m
  }, [traceDetail?.logs])

  const nodeById = useMemo(() => new Map((nodes ?? []).map((n) => [n.span_id, n])), [nodes])

  const spanTypeOptions = useMemo(() => {
    const s = new Set<string>()
    for (const n of nodes ?? []) s.add(n.span_type.toUpperCase())
    return Array.from(s).sort()
  }, [nodes])

  const [typeFilter, setTypeFilter] = useState<string>('all')

  const displayNodes = useMemo(() => {
    if (!nodes) return []
    return nodes.filter((n) => {
      if (typeFilter !== 'all' && n.span_type.toUpperCase() !== typeFilter) return false
      return true
    })
  }, [nodes, typeFilter])

  const parentOf = (n: TraceNode) => (n.parent_span_id ? nodeById.get(n.parent_span_id) : undefined)

  const pctOfParent = (n: TraceNode): number | null => {
    const p = parentOf(n)
    if (!p || p.duration_ms <= 0) return null
    return (n.duration_ms / p.duration_ms) * 100
  }

  const childrenByParent = useMemo(
    () => (nodes ? buildChildrenByParent(nodes) : new Map<string, TraceNode[]>()),
    [nodes],
  )

  const laneNodes = displayNodes

  const [focusSpanId, setFocusSpanId] = useState<string | null>(null)
  const viewWindow = useMemo(() => {
    if (!focusSpanId) return { start: 0, end: totalDuration }
    const n = nodeById.get(focusSpanId)
    if (!n) return { start: 0, end: totalDuration }
    const s = n.start_ms
    const e = n.start_ms + n.duration_ms
    return {
      start: Math.max(0, s - TRACE_FOCUS_PAD_MS),
      end: Math.min(totalDuration, e + TRACE_FOCUS_PAD_MS),
    }
  }, [focusSpanId, nodeById, totalDuration])

  const axisTicksFull = useMemo(() => axisTicksNice(0, totalDuration), [totalDuration])

  const laneNodesInView = useMemo(() => {
    if (!focusSpanId) return laneNodes
    const a = viewWindow.start
    const b = viewWindow.end
    return laneNodes.filter((n) => {
      const ns = n.start_ms
      const ne = n.start_ms + n.duration_ms
      return ns < b && ne > a
    })
  }, [laneNodes, focusSpanId, viewWindow.start, viewWindow.end])

  const [hoverSpanId, setHoverSpanId] = useState<string | null>(null)
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null)
  const [traceChartDetail, setTraceChartDetail] = useState(false)
  const [spanRailHover, setSpanRailHover] = useState<{
    spanId: string
    clientX: number
    clientY: number
  } | null>(null)
  const spanRailHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const clearSpanRailHideTimer = () => {
    if (spanRailHideTimerRef.current) {
      clearTimeout(spanRailHideTimerRef.current)
      spanRailHideTimerRef.current = null
    }
  }
  useEffect(
    () => () => {
      if (spanRailHideTimerRef.current) clearTimeout(spanRailHideTimerRef.current)
    },
    [],
  )

  const lineageHighlight = useMemo(() => {
    if (!hoverSpanId || !nodes) return new Set<string>()
    return lineageSpanIds(hoverSpanId, nodeById, childrenByParent)
  }, [hoverSpanId, nodes, nodeById, childrenByParent])

  const panelLogs = selectedSpanId ? (logsBySpanId.get(selectedSpanId) ?? []) : []
  const panelNode = selectedSpanId ? nodeById.get(selectedSpanId) : undefined

  const barTopPx = (TRACE_ROW_HEIGHT_PX - TRACE_BAR_HEIGHT_PX) / 2

  return (
    <div className="page trace-monitor-page">
      {error ? (
        <div className="trace-page-error">
          <Text type="danger">{error}</Text>
        </div>
      ) : null}
      {!error ? (
        <Card className="trace-monitor-card" bordered={false}>
          <div className="trace-monitor-body-inner">
            {nodes && nodes.length > 0 ? (
              <div className="trace-monitor-filters">
                <Space wrap size={[8, 8]} align="center">
                  {focusSpanId ? (
                    <Button size="small" type="link" onClick={() => setFocusSpanId(null)}>
                      显示全链路时间轴
                    </Button>
                  ) : null}
                  <Select
                    placeholder="Span 类型"
                    style={{ minWidth: 160 }}
                    value={typeFilter}
                    onChange={(v) => setTypeFilter(v ?? 'all')}
                    options={[
                      { label: '全部类型', value: 'all' },
                      ...spanTypeOptions.map((t) => ({ label: t, value: t })),
                    ]}
                  />
                </Space>
              </div>
            ) : null}
            {!nodes || nodes.length === 0 ? (
              traceDetail ? (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无 Span 数据" />
              ) : (
                <Text type="secondary">加载中…</Text>
              )
            ) : displayNodes.length === 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前类型下无 Span，请切换类型" />
            ) : laneNodesInView.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="当前聚焦时间范围内无 Span，点击「显示全链路时间轴」恢复"
              />
            ) : (
              <>
                <div
                  className={`trace-pixel-wrap${traceChartDetail ? ' trace-pixel-wrap--detail' : ''}`}
                  onMouseEnter={() => setTraceChartDetail(true)}
                  onMouseLeave={() => {
                    setTraceChartDetail(false)
                    setHoverSpanId(null)
                  }}
                >
                  <div className="trace-pixel-scroll">
                    <div className="trace-pixel-body">
                      <div className="trace-pixel-axis-row trace-pixel-split-row">
                        <div
                          className="trace-pixel-label-col trace-pixel-thead trace-pixel-label-sticky"
                          style={{ width: TRACE_LABEL_COL_PX, flexShrink: 0 }}
                        >
                          <Text type="secondary" className="trace-thead-compact">
                            节点
                          </Text>
                        </div>
                        <div className="trace-chart-pane trace-chart-pane--axis">
                          <div className="trace-chart-track">
                            <span className="trace-pixel-tick trace-pixel-tick--edge" style={{ left: 0, transform: 'none' }}>
                              0ms
                            </span>
                            {traceChartDetail
                              ? axisTicksFull
                                  .filter((t) => t > 0.5 && t < totalDuration - 0.5)
                                  .map((t) => (
                                    <span
                                      key={`tick-mid-${t}`}
                                      className="trace-pixel-tick trace-pixel-tick--mid"
                                      style={{
                                        left: `${totalDuration > 0 ? (t / totalDuration) * 100 : 0}%`,
                                        transform: 'translateX(-50%)',
                                      }}
                                    >
                                      {Math.round(t)}ms
                                    </span>
                                  ))
                              : null}
                            <span
                              className="trace-pixel-tick trace-pixel-tick--edge trace-pixel-tick--end"
                              style={{ left: '100%', transform: 'translateX(-100%)' }}
                            >
                              {Math.round(totalDuration)}ms
                            </span>
                            {focusSpanId ? <span className="trace-pixel-focus-hint">聚焦时间窗</span> : null}
                          </div>
                        </div>
                      </div>
                      {laneNodesInView.map((node) => {
                        const depthEven = node.depth % 2 === 0
                        const continues = treeRailContinues(node, nodeById, childrenByParent)
                        const lastChild = isLastChild(node, nodeById, childrenByParent)
                        const railCols = Math.min(continues.length, 6)
                        const railW = Math.min(8 + railCols * 7 + (node.depth > 0 ? 12 : 10), 64)
                        const td = totalDuration
                        const leftPct = td > 0 ? (node.start_ms / td) * 100 : 0
                        const widthPct =
                          td > 0
                            ? Math.max((node.duration_ms / td) * 100, node.duration_ms > 0 ? 0.06 : 0)
                            : 0
                        const isOk = node.status === 'ok' || node.status === 'success'
                        const isSlow = node.duration_ms >= TRACE_SLOW_MS
                        const barBg = traceBarSolidColor(node.span_type, !isOk)
                        const dotColor = barBg
                        const isLineage = Boolean(hoverSpanId && lineageHighlight.has(node.span_id))
                        const isFocusTarget = focusSpanId === node.span_id
                        return (
                          <div
                            key={node.span_id}
                            role="button"
                            tabIndex={0}
                            className={`trace-pixel-lane trace-pixel-split-row trace-pixel-lane--d${node.depth}${depthEven ? ' trace-pixel-lane--even' : ' trace-pixel-lane--odd'}${isLineage ? ' is-lineage' : ''}${isSlow ? ' is-slow' : ''}${isFocusTarget ? ' is-focus-target' : ''}`}
                            style={{ minHeight: TRACE_ROW_HEIGHT_PX, marginBottom: TRACE_LANE_GAP_PX }}
                            title="单击：右侧 Span 日志与详情 · 双击：聚焦该段时间窗"
                            onMouseEnter={() => setHoverSpanId(node.span_id)}
                            onDoubleClick={(e) => {
                              e.stopPropagation()
                              setFocusSpanId(node.span_id)
                            }}
                            onClick={() =>
                              setSelectedSpanId((s) => (s === node.span_id ? null : node.span_id))
                            }
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault()
                                setSelectedSpanId((s) => (s === node.span_id ? null : node.span_id))
                              }
                            }}
                          >
                            <div
                              className="trace-pixel-label-col trace-pixel-label-body trace-pixel-label-sticky trace-pixel-label-single"
                              style={{ width: TRACE_LABEL_COL_PX, flexShrink: 0 }}
                            >
                              <div className="trace-pixel-label-row trace-pixel-label-row--single">
                                <div className="trace-tree-rail" style={{ width: railW, flexShrink: 0 }}>
                                  {continues.slice(0, railCols).map((c, idx) => (
                                    <span key={`${node.span_id}-v-${idx}`} className="trace-tree-rail-col">
                                      <span
                                        className={`trace-tree-vline${c ? ' trace-tree-vline--on' : ' trace-tree-vline--dim'}`}
                                      />
                                    </span>
                                  ))}
                                  <span className="trace-tree-rail-end">
                                    {node.depth > 0 ? (
                                      <span className="trace-tree-branch">{lastChild ? '└' : '├'}</span>
                                    ) : null}
                                    <span
                                      className="trace-tree-dot"
                                      style={{ background: dotColor }}
                                      title={node.span_type}
                                    />
                                  </span>
                                </div>
                                <div className="trace-node-name-scroll">
                                  <span className="trace-node-label trace-node-label-pixel" title={node.name}>
                                    {node.name}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="trace-chart-pane trace-chart-pane--lane trace-pixel-lane-chart trace-pixel-lane-chart--bare">
                              <div
                                className="trace-chart-track"
                                onMouseEnter={(e) => {
                                  clearSpanRailHideTimer()
                                  setSpanRailHover({
                                    spanId: node.span_id,
                                    clientX: e.clientX,
                                    clientY: e.clientY,
                                  })
                                }}
                                onMouseMove={(e) => {
                                  clearSpanRailHideTimer()
                                  setSpanRailHover({
                                    spanId: node.span_id,
                                    clientX: e.clientX,
                                    clientY: e.clientY,
                                  })
                                }}
                                onMouseLeave={() => {
                                  spanRailHideTimerRef.current = window.setTimeout(() => {
                                    setSpanRailHover(null)
                                    spanRailHideTimerRef.current = null
                                  }, 180)
                                }}
                              >
                                {traceChartDetail ? (
                                  <div className="trace-pixel-lane-grid" aria-hidden>
                                    {axisTicksFull.map((t) => {
                                      const isMid = t > 0.5 && t < totalDuration - 0.5
                                      return (
                                        <span
                                          key={`g-${node.span_id}-${t}`}
                                          className={`trace-pixel-grid-v${isMid ? ' trace-pixel-grid-v--mid' : ''}`}
                                          style={{
                                            left: totalDuration > 0 ? `${(t / totalDuration) * 100}%` : '0%',
                                          }}
                                        />
                                      )
                                    })}
                                  </div>
                                ) : null}
                                <div
                                  className={`trace-bar trace-bar-pixel trace-bar-minimal${!isOk ? ' trace-bar--error' : ''}`}
                                  style={{
                                    left: `${leftPct}%`,
                                    width: `${widthPct}%`,
                                    top: barTopPx,
                                    height: TRACE_BAR_HEIGHT_PX,
                                    background: barBg,
                                  }}
                                />
                                <div
                                  className="trace-bar-endcap trace-bar-endcap--rail-right"
                                  style={{
                                    position: 'absolute',
                                    right: 0,
                                    top: 0,
                                    bottom: 0,
                                    zIndex: 4,
                                    pointerEvents: 'none',
                                  }}
                                >
                                  <span className="trace-bar-endcap-text">
                                    {formatTraceDurationMs(node.duration_ms)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </Card>
      ) : null}
      {spanRailHover && traceDetail
        ? (() => {
            const hNode = nodeById.get(spanRailHover.spanId)
            if (!hNode) return null
            return createPortal(
              <div
                role="tooltip"
                className="trace-popover-floating"
                style={clampTracePopoverPosition(spanRailHover.clientX, spanRailHover.clientY)}
                onMouseEnter={clearSpanRailHideTimer}
                onMouseLeave={() => setSpanRailHover(null)}
              >
                <TraceSpanHoverDetailBody
                  node={hNode}
                  traceId={traceDetail.summary.trace_id}
                  pctParent={pctOfParent(hNode)}
                />
              </div>,
              document.body,
            )
          })()
        : null}
      <Drawer
        title={
          <div className="trace-span-drawer-title-wrap">
            <Text strong className="trace-span-drawer-name" title={panelNode?.name}>
              {panelNode?.name ?? 'Span 详情'}
            </Text>
            {selectedSpanId ? (
              <Text code copyable={{ text: selectedSpanId }} className="trace-span-drawer-id">
                {selectedSpanId}
              </Text>
            ) : null}
          </div>
        }
        placement="right"
        width={Math.min(560, typeof window !== 'undefined' ? window.innerWidth - 24 : 560)}
        open={Boolean(selectedSpanId)}
        onClose={() => setSelectedSpanId(null)}
        destroyOnClose
        className="trace-span-detail-drawer"
        aria-label="Span 详情与日志"
      >
        {panelNode ? (
          <div className="trace-span-drawer-section">
            <Text type="secondary" className="trace-span-drawer-section-label">
              节点信息
            </Text>
            <div className="trace-span-drawer-meta">
              <div className="trace-span-drawer-meta-row">
                <span>类型</span>
                <span>{panelNode.span_type}</span>
              </div>
              {panelNode.component ? (
                <div className="trace-span-drawer-meta-row">
                  <span>组件</span>
                  <span>{panelNode.component}</span>
                </div>
              ) : null}
              <div className="trace-span-drawer-meta-row">
                <span>状态</span>
                <span>
                  {traceStatusIconEmoji(panelNode.status)} {traceStatusShortLabel(panelNode.status)}
                </span>
              </div>
              <div className="trace-span-drawer-meta-row">
                <span>耗时</span>
                <span>{formatTraceDurationMs(panelNode.duration_ms)}</span>
              </div>
              {panelNode.error_message ? (
                <div className="trace-span-drawer-meta-row trace-span-drawer-meta-row--block">
                  <span>错误</span>
                  <Text type="danger" className="trace-span-drawer-error">
                    {panelNode.error_message}
                  </Text>
                </div>
              ) : null}
              {panelNode.tags && Object.keys(panelNode.tags).length > 0 ? (
                <div className="trace-span-drawer-meta-row trace-span-drawer-meta-row--block">
                  <span>tags</span>
                  <pre className="trace-span-drawer-pre">{formatJsonBlock(panelNode.tags)}</pre>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="trace-span-drawer-section trace-span-drawer-section--logs">
          <Space align="center" size={8} wrap>
            <Text type="secondary" className="trace-span-drawer-section-label">
              Span 日志
            </Text>
            {panelLogs.length > 0 ? <Tag>{panelLogs.length} 条</Tag> : null}
          </Space>
          {panelLogs.length === 0 ? (
            <Text type="secondary">
              暂无结构化日志。流式请求请关注「agent.stream_turn」下的 stream.* 与 span.lifecycle.*
              事件；图节点 span 含 lifecycle 起止与上下文 payload。
            </Text>
          ) : null}
          {panelLogs.length > 0 ? (
            <ul className="trace-span-log-list trace-span-log-list--drawer">
              {panelLogs.map((log) => (
                <li key={log.log_id}>
                  <div className="trace-log-line">
                    <Tag>{log.log_level}</Tag>
                    <Text strong>{log.event_name}</Text>
                    <Text type="secondary" className="trace-log-time">
                      {log.occurred_at}
                    </Text>
                  </div>
                  <Text className="trace-log-msg">{log.message}</Text>
                  {log.payload && Object.keys(log.payload).length > 0 ? (
                    <pre className="trace-log-payload-pre">{formatJsonBlock(log.payload)}</pre>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </Drawer>
    </div>
  )
}
