/**
 * Agent 对话历史 API（GET /api/agent-conversations/...，信封 message + data）
 */

import { parseApiEnvelope } from './parseEnvelope'

export type PageMeta = {
  page: number
  page_size: number
  total: number
}

export type ConversationMessageRole = 'user' | 'assistant' | 'system'

export type ConversationSessionOut = {
  session_id: string
  agent_id: number
  title: string
  status: number
  created_at: string
  updated_at: string
  summary?: string | null
  summary_version?: number
  summary_job_status?: number
}

export type ConversationMessageOut = {
  id: number
  session_id: string
  /** 本条消息归属/产生的 Agent；与同 session 下其它 agent 消息区分（像素办公室等多角色同会话） */
  agent_id?: number
  role: ConversationMessageRole
  content: string
  content_type: string
  created_at: string
  metadata: Record<string, unknown> | null
  /** 会话内轮次；0 为占位（system / 历史回填） */
  turn_index: number
  /** assistant 指向本轮 user 消息 id */
  reply_message_id?: number | null
  /** user：正文估算；assistant：网关 usage 总 token */
  tokens?: number | null
}

export type ConversationSessionListData = {
  items: ConversationSessionOut[]
  meta: PageMeta
}

export type ConversationDetailData = {
  session: ConversationSessionOut
  messages: ConversationMessageOut[]
  /** 消息分页：第 1 页为最近一批，页内时间正序 */
  messages_meta: PageMeta
}

/** GET .../messages/{id}/execution — 与后端 ConversationMessageExecutionOut 一致 */
export type ConversationMessageExecutionOut = {
  message_id: number
  session_id: string
  agent_id: number
  role: ConversationMessageRole
  process_trace?: unknown[] | null
  tool_history?: Record<string, unknown>[] | null
  thinking_text?: string | null
}

const jsonHeaders = { 'Content-Type': 'application/json', Accept: 'application/json' }

export type ListConversationSessionsParams = {
  page?: number
  page_size?: number
  agent_id?: number
}

export type ConversationSessionCreateBody = {
  agent_id: number
  title?: string | null
}

/** POST /api/agent-conversations/sessions — 创建空会话（绑定 agent_entity） */
export async function createConversationSession(
  body: ConversationSessionCreateBody,
): Promise<ConversationSessionOut> {
  const res = await fetch('/api/agent-conversations/sessions', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<ConversationSessionOut>(res)
}

export async function listConversationSessions(
  params?: ListConversationSessionsParams,
): Promise<ConversationSessionListData> {
  const q = new URLSearchParams()
  if (params?.page != null) q.set('page', String(params.page))
  if (params?.page_size != null) q.set('page_size', String(params.page_size))
  if (params?.agent_id != null) q.set('agent_id', String(params.agent_id))
  const qs = q.toString()
  const url = `/api/agent-conversations/sessions${qs ? `?${qs}` : ''}`
  const res = await fetch(url, { headers: jsonHeaders })
  return parseApiEnvelope<ConversationSessionListData>(res)
}

export type GetConversationSessionDetailParams = {
  page?: number
  page_size?: number
  /** 仅返回该 agent_id 的会话消息（工作台共享会话下排除子 Agent 行） */
  messages_agent_id?: number
  /**
   * 是否随列表返回每条消息的 metadata（含 process_trace / tool_history 等大字段）。
   * 默认 false：不拉 JSON 列，执行过程用 `getConversationMessageExecution`。
   */
  include_message_metadata?: boolean
}

export async function getConversationSessionDetail(
  sessionId: string,
  params?: GetConversationSessionDetailParams,
): Promise<ConversationDetailData> {
  const q = new URLSearchParams()
  if (params?.page != null) q.set('page', String(params.page))
  if (params?.page_size != null) q.set('page_size', String(params.page_size))
  if (params?.messages_agent_id != null) q.set('messages_agent_id', String(params.messages_agent_id))
  const includeMeta = params?.include_message_metadata === true
  q.set('include_message_metadata', includeMeta ? 'true' : 'false')
  const qs = q.toString()
  const res = await fetch(
    `/api/agent-conversations/sessions/${encodeURIComponent(sessionId)}${qs ? `?${qs}` : ''}`,
    {
      headers: jsonHeaders,
    },
  )
  return parseApiEnvelope<ConversationDetailData>(res)
}

/** GET /api/agent-conversations/sessions/{sessionId}/messages/{messageId}/execution */
export async function getConversationMessageExecution(
  sessionId: string,
  messageId: number,
): Promise<ConversationMessageExecutionOut> {
  const res = await fetch(
    `/api/agent-conversations/sessions/${encodeURIComponent(sessionId)}/messages/${messageId}/execution`,
    { headers: jsonHeaders },
  )
  return parseApiEnvelope<ConversationMessageExecutionOut>(res)
}

export type RollingSummaryEnqueueOut = {
  /** 是否已入队后台摘要任务 */
  queued: boolean
}

/** POST /api/agent-conversations/sessions/{id}/rolling-summary — 手动触发滚动摘要（异步） */
export async function postSessionRollingSummary(
  sessionId: string,
): Promise<RollingSummaryEnqueueOut> {
  const res = await fetch(
    `/api/agent-conversations/sessions/${encodeURIComponent(sessionId)}/rolling-summary`,
    { method: 'POST', headers: jsonHeaders },
  )
  return parseApiEnvelope<RollingSummaryEnqueueOut>(res)
}
