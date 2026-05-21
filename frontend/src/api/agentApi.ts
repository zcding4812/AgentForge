/**
 * Agent 调用 API（与 POST /api/agents/invoke 及 OpenAPI 字段对齐，信封 message + data）
 */

import { parseApiEnvelope } from './parseEnvelope'

export type AgentKind = 'simple_chat' | 'react' | 'plan_execute' | 'workbench'

export type ResponseFormat = 'text' | 'json_object' | 'json_schema'

export type ToolChoiceMode = 'auto' | 'none' | 'required' | 'specific'

export type ModelIdentityBody = {
  provider: string
  model_name: string
  deployment?: string | null
  base_url_override?: string | null
  api_key?: string | null
}

export type HyperparametersBody = {
  temperature?: number | null
  max_tokens?: number | null
  top_p?: number | null
  /** 可选；非 OpenAI 官方 Chat 参数，部分厂商/自建网关可用 */
  top_k?: number | null
  frequency_penalty?: number | null
  presence_penalty?: number | null
  stop?: string[] | null
  seed?: number | null
}

export type ResponseConstraintsBody = {
  response_format: ResponseFormat
  json_schema_id?: string | null
  /** `response_format=json_schema` 时与 OpenAI `response_format.json_schema.schema` 一致 */
  response_json_schema?: Record<string, unknown> | null
  parallel_tool_calls?: boolean | null
}

export type ToolChoiceBody = {
  mode: ToolChoiceMode
  forced_tool_name?: string | null
}

export type OutputControlBody = {
  strip_thinking_blocks: boolean
  include_tool_messages_in_raw: boolean
}

/** 与后端 `ToolCallPart` / `ToolResultPart` / `ChatHistoryTurn` 一致 */
export type ToolCallPart = {
  id: string
  name: string
  /** JSON 对象字符串 */
  arguments?: string
}

export type ToolResultPart = {
  tool_call_id: string
  name: string
  content?: string
}

export type ChatHistoryTurn = {
  user: string
  /** 可为空：若该轮仅有工具调用链 */
  assistant?: string
  tool_calls?: ToolCallPart[]
  tool_results?: ToolResultPart[]
}

/** 与后端 `PromptEngineeringBody` 一致 */
export type PromptEngineeringBody = {
  /** 第 1 类 System Prompt */
  system_prompt?: string | null
  /** 参考资料，并入末条 Human（控制台工作区不再编辑；其它客户端仍可传） */
  context?: string | null
  chat_history?: ChatHistoryTurn[]
  max_history_rounds?: number
}

/** 与后端 `AgentInvokeRequest` 一致 */
export type AgentInvokeRequest = {
  agent_kind: AgentKind
  user_message: string
  model_identity?: ModelIdentityBody
  hyperparameters?: HyperparametersBody
  response?: ResponseConstraintsBody
  tool_choice?: ToolChoiceBody | null
  output?: OutputControlBody
  config_id?: string | number | null
  request_id?: string | null
  /**
   * 工作区 Agent 主键。续聊须与 `conversation_session_id` 同传。
   * 仅传 `agent_id` 不传 `conversation_session_id` 时，每次请求都会新建会话，须在客户端保存响应里的 `conversation_session_id` 并下轮回传。
   */
  agent_id?: number | null
  /** 续聊时必传；省略且传了 agent_id 时服务端每轮新建会话 */
  conversation_session_id?: string | null
  /** 会话在库中的归属 agent；与 agent_id（执行方）可不同，须同 workspace_namespace（如像素办公室同会话） */
  conversation_owner_agent_id?: number | null
  tool_names?: string[]
  /** 为 true 时存在未注册工具名则 400；默认 false 时忽略并在 warnings 中提示 */
  strict_tool_names?: boolean
  /** 工作区命名空间；`workbench` 编排须与入库 `agent_entity.workspace_namespace` 一致 */
  workspace_namespace?: string
  prompts?: PromptEngineeringBody | null
  /**
   * 可选；写入用户消息 metadata.user_display_name（如工作台展示发言人）
   */
  conversation_user_display_name?: string | null
}

/** 与后端 `KnowledgeInvokeCitation` 一致：本轮知识库检索注入上下文的分片来源 */
export type KnowledgeInvokeCitation = {
  kb_id: number
  kb_name: string
  doc_id: number
  filename?: string | null
  chunk_index?: number | null
  text_snippet?: string | null
  match_type?: string | null
}

/** 与后端 `OrchestrationEdge` 一致：工作台本轮子 Agent 调用顺序 */
export type OrchestrationEdge = {
  order: number
  parent_agent_id: number
  child_agent_id: number
}

export type WorkbenchFanInItem = {
  tool: string
  code: string
  ok: boolean
  message: string
  child_agent_id?: number | null
}

export type WorkbenchFanInChildStat = {
  child_agent_id: number
  total_calls: number
  success_calls: number
  failed_calls: number
  last_code?: string | null
}

export type WorkbenchFanInSummary = {
  total_calls: number
  success_calls: number
  failed_calls: number
  unique_child_agent_ids: number[]
  items: WorkbenchFanInItem[]
  child_stats: WorkbenchFanInChildStat[]
}

/** 与后端 `AgentProcessTraceStep` / metadata.process_trace 一致 */
export type AgentProcessTraceStep = {
  seq: number
  text: string
  phase: 'step' | 'final'
  kind?: 'model' | 'tool_call' | 'tool_result'
  tool_call_id?: string | null
  name?: string | null
  arguments?: unknown
  content?: string | null
}

/** 与后端 metadata.tool_history / AgentInvokeData.tool_history 项一致 */
export type AgentToolHistoryEntry = {
  seq: number
  phase: 'call' | 'result'
  tool_call_id?: string | null
  name?: string | null
  arguments?: unknown
  content?: string | null
  /** 嵌套子 Agent 工具链时由后端展开标注 */
  sub_agent_id?: number | null
  nesting?: string | null
}

export type AgentInvokeData = {
  assistant_text: string
  /** 与正文分离的思考片段；历史消息见 metadata.thinking_text */
  thinking_text?: string | null
  structured?: Record<string, unknown> | null
  /** 本轮网关 usage 总 token */
  tokens?: number | null
  /** 本轮会话 ID；多轮时在下一次请求的 `conversation_session_id` 中原样回传 */
  conversation_session_id?: string | null
  /** 非致命提示，如未注册工具名已忽略 */
  warnings?: string[] | null
  /** 本轮绑定知识库检索命中，用于对话区展示来源 */
  knowledge_citations?: KnowledgeInvokeCitation[] | null
  /** 仅 workbench：本轮 `workbench_invoke_sub_agent` 解析顺序 */
  orchestration_edges?: OrchestrationEdge[] | null
  /** 仅 workbench：本轮 fan-in 聚合摘要（总数/成功失败/子 Agent 列表） */
  workbench_fan_in?: WorkbenchFanInSummary | null
  /** 执行过程时间线（模型与工具交错）；与助手消息 metadata.process_trace 一致 */
  process_trace?: AgentProcessTraceStep[] | null
  /** 本轮工具调用与结果时间序列；与助手消息 metadata.tool_history 一致 */
  tool_history?: AgentToolHistoryEntry[] | null
}

const jsonHeaders = { 'Content-Type': 'application/json', Accept: 'application/json' }

export type InvokeAgentOptions = {
  /** 传入后可调用 `AbortController.abort()` 取消进行中的请求（停止按钮） */
  signal?: AbortSignal
}

export type AgentStreamStartEvent = {
  type: 'start'
  agent_kind: string
  model: string
  provider: string
  config_id?: string | number | null
  tool_names: string[]
  conversation_session_id?: string | null
}

export type AgentStreamProgressEvent = {
  type: 'progress'
  stage: string
  tool?: string | null
  text?: string | null
  /** 知识库注入 progress 时：当前调用的 agent_entity.id */
  agent_id?: number | null
  /** 工作台子 Agent 起止；与 active_agent_id / workbench_agent_id 联用 */
  sub_phase?: 'start' | 'end' | null
  active_agent_id?: number | null
  workbench_agent_id?: number | null
  /** pixel_tool_start / pixel_tool_done：LangChain tool_call id */
  tool_id?: string | null
  /** pixel_tool_start：展示名 */
  tool_name?: string | null
  /** pixel_tool_start：状态文案；pixel_agent_status：active | waiting */
  status?: string | null
  permission_active?: boolean | null
  run_in_background?: boolean | null
}

export type AgentStreamPingEvent = { type: 'ping' }

export type AgentStreamErrorEvent = {
  type: 'error'
  code?: string
  /** 新契约优先；旧实现可能仅有 `detail` */
  message?: string
  detail?: string | null
}

export type InvokeAgentStreamOptions = InvokeAgentOptions & {
  /**
   * 收到增量后的回调：第一参为写入对语气泡的累计正文，第二参为写入顶部 Exploring 条的累计正文。
   * 工作台 SSE 可能带 `lane`：编排过程进第二参，最终汇总（workbench_aggregate）进第一参。
   */
  onDelta?: (contentAccumulated: string, exploringAccumulated: string) => void
  /** 首帧元数据（模型、工具列表等） */
  onStart?: (ev: AgentStreamStartEvent) => void
  /** 阶段进度（思考 / 调工具 / 生成等） */
  onProgress?: (ev: AgentStreamProgressEvent) => void
}

type AgentStreamServerEvent =
  | AgentStreamStartEvent
  | AgentStreamProgressEvent
  | AgentStreamPingEvent
  | { type: 'delta'; text: string; lane?: 'exploring' | 'content' }
  | {
      type: 'done'
      assistant_text: string
      thinking_text?: string | null
      structured?: Record<string, unknown> | null
      tokens?: number | null
      warnings?: string[] | null
      knowledge_citations?: KnowledgeInvokeCitation[] | null
      orchestration_edges?: OrchestrationEdge[] | null
      workbench_fan_in?: WorkbenchFanInSummary | null
      process_trace?: AgentProcessTraceStep[] | null
      tool_history?: AgentToolHistoryEntry[] | null
    }
  | AgentStreamErrorEvent

/**
 * 非流式调用；自动带 `X-Request-Id` 便于与后端链路对齐（可与 body.request_id 叠加）。
 */
export async function invokeAgent(
  body: AgentInvokeRequest,
  options?: InvokeAgentOptions,
): Promise<AgentInvokeData> {
  const traceId = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`
  const res = await fetch('/api/agents/invoke', {
    method: 'POST',
    headers: {
      ...jsonHeaders,
      'X-Request-Id': traceId,
    },
    body: JSON.stringify(body),
    signal: options?.signal,
  })
  return parseApiEnvelope<AgentInvokeData>(res)
}

/**
 * 流式调用（POST + `text/event-stream`）；用 fetch ReadableStream 解析 `data: {json}\\n\\n`。
 */
export async function invokeAgentStream(
  body: AgentInvokeRequest,
  options?: InvokeAgentStreamOptions,
): Promise<AgentInvokeData> {
  const traceId =
    typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`
  const res = await fetch('/api/agents/invoke/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'X-Request-Id': traceId,
    },
    body: JSON.stringify(body),
    signal: options?.signal,
  })

  if (!res.ok) {
    const text = await res.text()
    let detail: unknown = text
    try {
      const j = JSON.parse(text) as { detail?: unknown }
      if (j && typeof j === 'object' && 'detail' in j) detail = j.detail
    } catch {
      /* 保持原始 text */
    }
    const msg =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? JSON.stringify(detail)
          : JSON.stringify(detail ?? text)
    throw new Error(msg)
  }

  const reader = res.body?.getReader()
  if (!reader) {
    throw new Error('响应无 body（ReadableStream）')
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let accumulated = ''
  let accumulatedExploring = ''

  const handleSseLine = (rawLine: string): AgentInvokeData | undefined => {
    const line = rawLine.trimStart()
    if (!line.startsWith('data:')) return undefined
    const jsonStr = line.startsWith('data: ') ? line.slice(6).trim() : line.slice(5).trim()
    if (!jsonStr) return undefined
    let ev: AgentStreamServerEvent
    try {
      ev = JSON.parse(jsonStr) as AgentStreamServerEvent
    } catch {
      return undefined
    }
    if (ev.type === 'ping') {
      return undefined
    }
    if (ev.type === 'start') {
      options?.onStart?.(ev)
      return undefined
    }
    if (ev.type === 'progress') {
      options?.onProgress?.(ev)
      return undefined
    }
    if (ev.type === 'delta' && ev.text) {
      const lane = ev.lane === 'exploring' ? 'exploring' : 'content'
      if (lane === 'exploring') {
        accumulatedExploring += ev.text
      } else {
        accumulated += ev.text
      }
      options?.onDelta?.(accumulated, accumulatedExploring)
    }
    if (ev.type === 'done') {
      return {
        assistant_text: ev.assistant_text,
        thinking_text: ev.thinking_text ?? null,
        structured: ev.structured ?? null,
        tokens: ev.tokens ?? null,
        warnings: ev.warnings ?? null,
        knowledge_citations: ev.knowledge_citations ?? null,
        orchestration_edges: ev.orchestration_edges ?? null,
        workbench_fan_in: ev.workbench_fan_in ?? null,
        process_trace: ev.process_trace ?? null,
        tool_history: ev.tool_history ?? null,
      }
    }
    if (ev.type === 'error') {
      const msg =
        'message' in ev && ev.message
          ? ev.message
          : 'detail' in ev && typeof ev.detail === 'string'
            ? ev.detail
            : '流式执行失败'
      throw new Error(msg)
    }
    return undefined
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (value) {
        buffer += decoder.decode(value, { stream: true })
      }
      if (done) {
        buffer += decoder.decode()
        const tail = buffer.trim() === '' ? '' : buffer.endsWith('\n') ? buffer : `${buffer}\n`
        for (const line of tail.split(/\r?\n/)) {
          const out = handleSseLine(line)
          if (out) return out
        }
        break
      }
      const parts = buffer.split(/\r?\n/)
      buffer = parts.pop() ?? ''
      for (const line of parts) {
        const out = handleSseLine(line)
        if (out) return out
      }
    }
  } finally {
    reader.releaseLock?.()
  }

  throw new Error('流式响应未正常结束（缺少 done 事件）')
}
