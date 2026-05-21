import type { AgentStreamProgressEvent } from '../api/agentApi'

import { agentEntityIdEq } from './formatAgentStreamProgress'

/** 将 SSE `progress` 中 pixel_* / pixel_agent_status 转为画布消息（由调用方 `postMessage` 或 `forwardCanvas` 投递） */
const KNOWLEDGE_PIXEL_TOOL_ID = '__knowledge_query'

export type ApplyPixelStreamProgressOptions = {
  /**
   * 部分 progress（如 knowledge_query）依赖 body.agent_id；若后端未带，可用当前壳层选中的 Agent id 对齐小人。
   */
  fallbackAgentId?: number | null
  /**
   * 若设置：仅将 `pixel_tool_*` / `pixel_tools_clear` 投递到画布（`agent_id` 与该 id 一致时）。
   * 像素主包若未按 postMessage 的 `id` 分角色绑定工具条，子 Agent 工具帧会误刷新主角；工作台场景传入当前选中的编排 Agent id 可规避。
   */
  canvasPixelToolAgentIdAllowlist?: number | null
}

function allowCanvasPixelToolEvent(
  ev: AgentStreamProgressEvent,
  allowlist: number | null | undefined,
): boolean {
  if (allowlist == null) return true
  if (ev.agent_id == null) return false
  return agentEntityIdEq(ev.agent_id, allowlist)
}

/** 与后端 ReAct ``pixel_tool_start`` 一致：主包头顶条优先展示 ``status``（见 ``facade._collect_pixel_tool_sse_from_stream_message``） */
function knowledgePixelStatus(ev: AgentStreamProgressEvent): string {
  const hint = (ev.text || '').trim()
  if (hint) return `调用 知识库检索 · ${hint}`
  return '调用 知识库检索…'
}

export function applyPixelStreamProgress(
  ev: AgentStreamProgressEvent,
  deliver: (payload: object) => void,
  options?: ApplyPixelStreamProgressOptions,
): void {
  const stage = (ev.stage || '').trim()
  if (stage === 'knowledge_query') {
    const aid = ev.agent_id ?? options?.fallbackAgentId
    if (aid == null) return
    deliver({
      type: 'agentToolStart',
      id: aid,
      toolId: KNOWLEDGE_PIXEL_TOOL_ID,
      toolName: '知识库检索',
      status: knowledgePixelStatus(ev),
      permissionActive: false,
      runInBackground: true,
    })
    // 勿在同一 tick 内 agentToolDone：主包往往来不及绘制头顶文案；本轮结束由壳层 agentToolsClear 统一清理
    return
  }
  if (stage === 'pixel_tool_start') {
    if (!allowCanvasPixelToolEvent(ev, options?.canvasPixelToolAgentIdAllowlist)) return
    const aid = ev.agent_id
    const tid = ev.tool_id
    if (aid == null || !tid) return
    deliver({
      type: 'agentToolStart',
      id: aid,
      toolId: tid,
      toolName: ev.tool_name ?? 'Task',
      status: ev.status ?? '…',
      permissionActive: ev.permission_active ?? false,
      runInBackground: ev.run_in_background ?? true,
    })
    return
  }
  if (stage === 'pixel_tool_done') {
    if (!allowCanvasPixelToolEvent(ev, options?.canvasPixelToolAgentIdAllowlist)) return
    const aid = ev.agent_id
    const tid = ev.tool_id
    if (aid == null || !tid) return
    deliver({ type: 'agentToolDone', id: aid, toolId: tid })
    return
  }
  if (stage === 'pixel_tools_clear') {
    if (!allowCanvasPixelToolEvent(ev, options?.canvasPixelToolAgentIdAllowlist)) return
    const aid = ev.agent_id
    if (aid == null) return
    deliver({ type: 'agentToolsClear', id: aid })
    return
  }
  if (stage === 'pixel_agent_status') {
    const aid = ev.agent_id
    const st = ev.status
    if (aid == null || (st !== 'active' && st !== 'waiting')) return
    deliver({ type: 'agentStatus', id: aid, status: st })
  }
}
