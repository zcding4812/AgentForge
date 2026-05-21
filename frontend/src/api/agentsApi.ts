/**
 * Agent 资源 API（GET/POST /api/agents、GET /api/agents/{id}），信封 message + data
 */

import type { AgentKind } from './agentApi'
import { parseApiEnvelope, readJsonBody } from './parseEnvelope'

export type AgentOut = {
  id: number
  workspace_namespace?: string
  name: string
  description: string | null
  agent_kind: AgentKind
  status: string
  sys_model_id: number | null
  /** 第 1 类系统提示（与 prompts.system_prompt 对齐） */
  system_prompt?: string | null
  created_at: string
  updated_at: string
}

export type AgentDetailOut = AgentOut & {
  config_json: Record<string, unknown> | null
}

export type AgentListData = {
  items: AgentOut[]
  total: number
  page: number
  page_size: number
}

const jsonHeaders = { 'Content-Type': 'application/json', Accept: 'application/json' }

const base = '/api/agents'

export async function fetchAgentsList(params: {
  page?: number
  page_size?: number
  q?: string
  /** 多选：同一 query key 重复传 agent_kind */
  agent_kind?: AgentKind[]
  /** 从列表排除的类型（如系统工作台 workbench） */
  exclude_agent_kind?: AgentKind[]
  /** 按工作区命名空间精确筛选（与入库 workspace_namespace 一致） */
  workspace_namespace?: string
}): Promise<AgentListData> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.q?.trim()) sp.set('q', params.q.trim())
  if (params.workspace_namespace?.trim()) {
    sp.set('workspace_namespace', params.workspace_namespace.trim())
  }
  for (const k of params.agent_kind ?? []) {
    sp.append('agent_kind', k)
  }
  for (const k of params.exclude_agent_kind ?? []) {
    sp.append('exclude_agent_kind', k)
  }
  const q = sp.toString()
  const url = q ? `${base}?${q}` : base
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<AgentListData>(res)
}

export type AgentCreateBody = {
  name: string
  description?: string | null
  agent_kind: AgentKind
  /** 默认 `default`；创建后变更需后端支持（当前仅创建时可写） */
  workspace_namespace?: string
  system_prompt?: string | null
  /** 挂载对话模型 `sys_model.id`（须为已启用的对话类模型） */
  sys_model_id?: number | null
  /** 工作区配置快照（温度、tool_names 等），与 PATCH 语义一致 */
  config_json?: Record<string, unknown> | null
}

export async function createAgent(body: AgentCreateBody): Promise<AgentOut> {
  const res = await fetch(base, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentOut>(res)
}

/** 在全新命名空间下创建唯一工作台 Agent（POST /api/agents/workbench） */
export async function createWorkbenchWorkspace(body: {
  workspace_namespace: string
  name?: string
  description?: string | null
}): Promise<AgentOut> {
  const res = await fetch(`${base}/workbench`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentOut>(res)
}

export type AgentUpdateBody = {
  name?: string
  description?: string | null
  agent_kind?: AgentKind
  sys_model_id?: number | null
  system_prompt?: string | null
  config_json?: Record<string, unknown> | null
}

export async function updateAgent(agentId: number, body: AgentUpdateBody): Promise<AgentDetailOut> {
  const res = await fetch(`${base}/${agentId}`, {
    method: 'PATCH',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentDetailOut>(res)
}

/** DELETE /api/agents/{id}，成功返回 204 无 body（物理删除） */
export async function deleteAgent(agentId: number): Promise<void> {
  const res = await fetch(`${base}/${agentId}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  })
  if (res.status === 204) return
  let detail: unknown = res.statusText
  const j = await readJsonBody<{ detail?: unknown }>(res)
  if (j && typeof j === 'object' && 'detail' in j) detail = j.detail
  const msg =
    typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? JSON.stringify(detail)
        : JSON.stringify(detail ?? res.statusText)
  throw new Error(msg)
}

export async function fetchAgentById(agentId: number): Promise<AgentDetailOut> {
  const res = await fetch(`${base}/${agentId}`, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<AgentDetailOut>(res)
}

/** HTTP 工具在列表中的请求摘要（与 OpenAPI AgentHttpToolConfigOut 一致） */
export type AgentHttpToolConfigOut = {
  method: string
  url: string
  headers: Record<string, string> | null
  body: string | null
  timeout_seconds: number
}

/** 与 GET /api/agents/tools、进程内 LangChain 工具注册表一致 */
export type AgentRegisteredToolOut = {
  name: string
  namespace: string
  description: string
  parameters_summary: string
  parameters_json_schema: Record<string, unknown> | null
  origin: 'builtin' | 'runtime' | 'http' | 'mcp'
  http_request: AgentHttpToolConfigOut | null
  mcp_config: Record<string, unknown> | null
}

export type AgentRegisteredToolsData = {
  items: AgentRegisteredToolOut[]
  namespace: string
}

export async function fetchAgentRegisteredTools(params?: {
  namespace?: string
}): Promise<AgentRegisteredToolsData> {
  const sp = new URLSearchParams()
  if (params?.namespace?.trim()) sp.set('namespace', params.namespace.trim())
  const q = sp.toString()
  const url = q ? `${base}/tools?${q}` : `${base}/tools`
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<AgentRegisteredToolsData>(res)
}

/** 与 POST/PATCH /api/agents/tools/http 一致：持久化 HTTP 工具 */
export type AgentRegisterHttpToolBody = {
  name: string
  description: string
  url: string
  method?: string
  headers?: Record<string, string> | null
  body?: string | null
  timeout_seconds?: number
  enabled?: boolean
  input_schema?: Record<string, unknown> | null
  output_schema?: Record<string, unknown> | null
}

export type AgentUpdateHttpToolBody = {
  description?: string
  url?: string
  method?: string
  headers?: Record<string, string> | null
  body?: string | null
  timeout_seconds?: number
  enabled?: boolean
  input_schema?: Record<string, unknown> | null
  output_schema?: Record<string, unknown> | null
  version?: number
  clear_headers?: boolean
  clear_body?: boolean
}

/**
 * MCP 工具 `tools_config` 常用键（与后端 `make_mcp_streamable_http_tool` 一致，可并存其它扩展）：
 * - `mcp_tool_name`：远端 `tools/call` 的 name；缺省则等于注册名或由后端按 server_url 默认。
 * - `dispatch_mode`：为 true 时注册调度工具，先 `list_tools` 再 `call_tool`。
 */
export type McpToolsConfigFields = {
  mcp_tool_name?: string
  dispatch_mode?: boolean
  [key: string]: unknown
}

/** 与 POST/PATCH /api/agents/tools/mcp 一致 */
export type AgentRegisterMcpToolBody = {
  name: string
  description: string
  transport_type?: 'http' | 'sse'
  server_url: string
  /** 可含 `protocol_version`（如 `2025-06-18`）、headers 等，会并入进程内 MCP 客户端配置 */
  connection_config?: Record<string, unknown> | null
  /** 见 {@link McpToolsConfigFields} */
  tools_config?: Record<string, unknown> | null
  enabled?: boolean
  input_schema?: Record<string, unknown> | null
  output_schema?: Record<string, unknown> | null
}

export type AgentUpdateMcpToolBody = {
  description?: string
  transport_type?: 'http' | 'sse'
  server_url?: string
  connection_config?: Record<string, unknown> | null
  /** 见 {@link McpToolsConfigFields} */
  tools_config?: Record<string, unknown> | null
  enabled?: boolean
  input_schema?: Record<string, unknown> | null
  output_schema?: Record<string, unknown> | null
  version?: number
  clear_connection_config?: boolean
  clear_tools_config?: boolean
}

export type AgentToolRegisterResultData = { name: string }

export type AgentHttpToolPersistedOut = {
  id: number
  name: string
  description: string
  kind: 'http' | 'mcp'
  enabled: boolean
  version: number
  input_schema: Record<string, unknown> | null
  output_schema: Record<string, unknown> | null
  url: string | null
  method: string | null
  headers: Record<string, string> | null
  body: string | null
  timeout_ms: number | null
  timeout_seconds: number | null
  transport_type: string | null
  server_url: string | null
  connection_config: Record<string, unknown> | null
  tools_config: Record<string, unknown> | null
  mcp_config: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export type AgentRuntimeHttpToolsListData = {
  items: AgentHttpToolPersistedOut[]
}

export async function fetchPersistedHttpTools(): Promise<AgentRuntimeHttpToolsListData> {
  const res = await fetch(`${base}/tools/http`, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<AgentRuntimeHttpToolsListData>(res)
}

export async function createPersistedHttpTool(
  body: AgentRegisterHttpToolBody,
): Promise<AgentToolRegisterResultData> {
  const res = await fetch(`${base}/tools/http`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentToolRegisterResultData>(res)
}

export async function updatePersistedHttpTool(
  toolName: string,
  body: AgentUpdateHttpToolBody,
): Promise<AgentHttpToolPersistedOut> {
  const enc = encodeURIComponent(toolName)
  const res = await fetch(`${base}/tools/http/${enc}`, {
    method: 'PATCH',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentHttpToolPersistedOut>(res)
}

export async function createPersistedMcpTool(
  body: AgentRegisterMcpToolBody,
): Promise<AgentToolRegisterResultData> {
  const res = await fetch(`${base}/tools/mcp`, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentToolRegisterResultData>(res)
}

export async function updatePersistedMcpTool(
  toolName: string,
  body: AgentUpdateMcpToolBody,
): Promise<AgentHttpToolPersistedOut> {
  const enc = encodeURIComponent(toolName)
  const res = await fetch(`${base}/tools/mcp/${enc}`, {
    method: 'PATCH',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<AgentHttpToolPersistedOut>(res)
}

/** POST /api/agents/tools/mcp/probe-list */
export type McpProbeListToolsBody = {
  server_url: string
  transport_type?: 'http' | 'sse'
  connection_config?: Record<string, unknown> | null
}

export type McpProbeListToolsData = {
  tools: Record<string, unknown>[]
}

export async function probeMcpListTools(body: McpProbeListToolsBody): Promise<McpProbeListToolsData> {
  const res = await fetch(`${base}/tools/mcp/probe-list`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<McpProbeListToolsData>(res)
}

export async function deletePersistedHttpTool(toolName: string): Promise<void> {
  const enc = encodeURIComponent(toolName)
  const res = await fetch(`${base}/tools/http/${enc}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  })
  if (res.status === 204) return
  let detail: unknown = res.statusText
  const j = await readJsonBody<{ detail?: unknown }>(res)
  if (j && typeof j === 'object' && 'detail' in j) detail = j.detail
  const msg =
    typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? JSON.stringify(detail)
        : JSON.stringify(detail ?? res.statusText)
  throw new Error(msg)
}
