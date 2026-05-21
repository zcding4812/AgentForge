import type { AgentStreamProgressEvent } from '../api/agentApi'

/** 比较后端/JSON 可能出现的 string | number agent 主键 */
export function agentEntityIdEq(a: unknown, b: unknown): boolean {
  if (a == null || b == null) return false
  const na = Number(a)
  const nb = Number(b)
  return !Number.isNaN(na) && !Number.isNaN(nb) && na === nb
}

/**
 * 侧栏「对话生成中 / 过程」文案是否应随该 progress 更新。
 * 子 Agent 流式里的 `pixel_tool_*` 已带其 `agent_entity.id`；与当前壳层选中的主 Agent 不一致时，
 * 仅由 `applyPixelStreamProgress` 驱动对应像素小人工具条，避免主 Agent 过程条被子工具状态覆盖。
 */
export function streamProgressPhaseBelongsToShellAgent(
  ev: AgentStreamProgressEvent,
  shellAgentEntityId: number | null | undefined,
): boolean {
  if (shellAgentEntityId == null) return true
  const stage = (ev.stage || '').trim()
  if (
    stage === 'pixel_tool_start' ||
    stage === 'pixel_tool_done' ||
    stage === 'pixel_tools_clear'
  ) {
    if (ev.agent_id == null) return false
    return agentEntityIdEq(ev.agent_id, shellAgentEntityId)
  }
  return true
}

/** 将 SSE `progress` 事件格式化为一条可展示的流程状态（与后端 `stage` / `sub_phase` 对齐）。 */
export function formatAgentStreamProgressLabel(ev: AgentStreamProgressEvent): string {
  const stage = (ev.stage || '').trim()
  const toolRaw = (ev.tool || '').trim()
  const rawText = (ev.text || '').trim()

  const invokeToolLabel = (t: string): string => {
    if (t === 'workbench_invoke_sub_agent') return '单次委派'
    if (t === 'workbench_invoke_sub_agents_parallel') return '并行委派'
    if (t === 'knowledge_retrieval') return '知识库检索'
    return t
  }
  const toolLbl = toolRaw ? invokeToolLabel(toolRaw) : ''

  if (stage === 'sub_agent') {
    const sid = ev.active_agent_id != null ? `#${ev.active_agent_id}` : ''
    const via = toolLbl ? ` · ${toolLbl}` : ''
    if (ev.sub_phase === 'start') {
      return sid ? `编排 · 子 Agent ${sid} 执行中${via}` : `编排 · 子 Agent 执行中${via}`
    }
    if (ev.sub_phase === 'end') {
      return sid ? `编排 · 子 Agent ${sid} 已完成${via}` : `编排 · 子 Agent 已完成${via}`
    }
    return rawText || (sid ? `编排 · 子 Agent ${sid}${via}` : `编排 · 子 Agent${via}`)
  }

  if (stage === 'knowledge_query') {
    // 与 pixel_tool_start 一致使用「工具 · …」前缀，便于工作台/控制台/像素壳统一识别
    const name = toolLbl || '知识库检索'
    if (rawText) return `工具 · ${name} · ${rawText}`
    return `工具 · ${name} · 准备生成`
  }

  if (stage === 'generating') {
    return rawText || '模型 · 正在生成回复'
  }

  if (stage === 'pixel_tool_start') {
    const n = (ev.tool_name || ev.tool || '').trim()
    const st = (ev.status || '').trim()
    if (st) return st
    return n ? `工具 · ${n}` : '工具 · 执行中'
  }
  if (stage === 'pixel_tool_done') {
    const n = (ev.tool_name || ev.tool_id || '').trim()
    return n ? `工具 · ${n} · 已完成` : '工具 · 已完成'
  }
  if (stage === 'pixel_tools_clear') {
    return ''
  }
  if (stage === 'pixel_agent_status') {
    const sid = ev.agent_id != null ? `#${ev.agent_id}` : ''
    if (ev.status === 'active') return sid ? `角色 ${sid} · 执行中` : '执行中'
    if (ev.status === 'waiting') return sid ? `角色 ${sid} · 待命` : '待命'
    return ''
  }

  if (rawText) return rawText
  if (stage && toolLbl) return `${stage} · ${toolLbl}`
  if (stage) return stage
  if (toolLbl) return toolLbl
  return ''
}
