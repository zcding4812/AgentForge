/**
 * 从 GET /api/agents/{id} 构造 invoke 请求体（与 Agent 工作区一致）。
 */

import type { AgentInvokeRequest, PromptEngineeringBody, ResponseFormat } from '../api/agentApi'
import type { AgentDetailOut } from '../api/agentsApi'
import { DEFAULT_MAX_TOKENS } from '../constants'

function buildPromptsPayload(systemPrompt: string): PromptEngineeringBody | undefined {
  const out: PromptEngineeringBody = {}
  if (systemPrompt.trim()) out.system_prompt = systemPrompt.trim()
  return Object.keys(out).length > 0 ? out : undefined
}

function parseStopWordsForApi(s: string): string[] | undefined {
  const lines = s
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  return lines.length ? lines : undefined
}

export type ParsedInvokeWorkspace = {
  temperature: number
  maxTokens: number
  topP: number
  stopWords: string
  seed: number | null
  frequencyPenalty: number
  presencePenalty: number
  responseFormat: ResponseFormat
  toolNames: string[]
  useToolChoice: boolean
  toolChoiceMode: 'auto' | 'none' | 'required' | 'specific'
  forcedToolName: string
  stripThinking: boolean
  streamOutput: boolean
  includeToolMessagesInRaw: boolean
  promptSystem: string
  /** 结构化输出：OpenAI json_schema 模式下的根 schema；与 jsonSchemaId 可选并存 */
  responseJsonSchema: Record<string, unknown> | null
  /** 可选：作为下发给 API 的 schema 名（`json_schema.name`） */
  jsonSchemaId: string
}

const DEFAULTS: ParsedInvokeWorkspace = {
  temperature: 0.7,
  maxTokens: DEFAULT_MAX_TOKENS,
  topP: 0.9,
  stopWords: '',
  seed: null,
  frequencyPenalty: 0,
  presencePenalty: 0,
  responseFormat: 'text',
  toolNames: [],
  useToolChoice: false,
  toolChoiceMode: 'auto',
  forcedToolName: '',
  stripThinking: true,
  streamOutput: false,
  includeToolMessagesInRaw: false,
  promptSystem: '',
  responseJsonSchema: null,
  jsonSchemaId: '',
}

export function parseInvokeWorkspaceFromConfigJson(
  raw: AgentDetailOut['config_json'],
): ParsedInvokeWorkspace {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    return { ...DEFAULTS }
  }
  const o = raw as Record<string, unknown>
  const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
  const str = (v: unknown, d: string) => (typeof v === 'string' ? v : d)
  const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d)

  let promptSystem = str(o.prompt_system, '')
  if ('prompt_system_prompt' in o) {
    promptSystem = str(o.prompt_system_prompt, promptSystem)
  } else if ('prompt_developer' in o || 'prompt_auxiliary' in o) {
    const dev = str(o.prompt_developer, '')
    const aux = str(o.prompt_auxiliary, '')
    if (dev) promptSystem = promptSystem ? `${promptSystem}\n\n## 能力（补充）\n${dev}` : `## 能力（补充）\n${dev}`
    if (aux) promptSystem = promptSystem ? `${promptSystem}\n\n## 约束与输出（补充）\n${aux}` : `## 约束与输出（补充）\n${aux}`
  }

  let responseFormat: ResponseFormat = DEFAULTS.responseFormat
  if (o.response_format === 'text' || o.response_format === 'json_object' || o.response_format === 'json_schema') {
    responseFormat = o.response_format
  }

  let toolNames: string[] = DEFAULTS.toolNames
  if (Array.isArray(o.tool_names) && o.tool_names.every((x) => typeof x === 'string')) {
    toolNames = o.tool_names as string[]
  }

  let toolChoiceMode = DEFAULTS.toolChoiceMode
  if (o.tool_choice_mode === 'auto' || o.tool_choice_mode === 'none' || o.tool_choice_mode === 'required' || o.tool_choice_mode === 'specific') {
    toolChoiceMode = o.tool_choice_mode
  }

  let seed: number | null = DEFAULTS.seed
  if ('seed' in o) {
    if (o.seed === null) seed = null
    else if (typeof o.seed === 'number' && Number.isFinite(o.seed)) seed = Math.round(o.seed)
  }

  let responseJsonSchema: Record<string, unknown> | null = DEFAULTS.responseJsonSchema
  if (o.response_json_schema != null && typeof o.response_json_schema === 'object' && !Array.isArray(o.response_json_schema)) {
    responseJsonSchema = o.response_json_schema as Record<string, unknown>
  }
  const jsonSchemaId = typeof o.json_schema_id === 'string' ? o.json_schema_id.trim() : DEFAULTS.jsonSchemaId

  return {
    temperature: num(o.temperature, DEFAULTS.temperature),
    maxTokens: Math.round(num(o.max_tokens, DEFAULTS.maxTokens)),
    topP: num(o.top_p, DEFAULTS.topP),
    stopWords: str(o.stop_words, ''),
    seed,
    frequencyPenalty:
      o.frequency_penalty === null || o.frequency_penalty === undefined
        ? DEFAULTS.frequencyPenalty
        : num(o.frequency_penalty as number, DEFAULTS.frequencyPenalty),
    presencePenalty:
      o.presence_penalty === null || o.presence_penalty === undefined
        ? DEFAULTS.presencePenalty
        : num(o.presence_penalty as number, DEFAULTS.presencePenalty),
    responseFormat,
    toolNames,
    useToolChoice: bool(o.use_tool_choice, DEFAULTS.useToolChoice),
    toolChoiceMode,
    forcedToolName: str(o.forced_tool_name, ''),
    stripThinking: bool(o.strip_thinking, DEFAULTS.stripThinking),
    streamOutput: bool(o.stream_output, DEFAULTS.streamOutput),
    includeToolMessagesInRaw: bool(o.include_tool_messages_in_raw, DEFAULTS.includeToolMessagesInRaw),
    promptSystem,
    responseJsonSchema,
    jsonSchemaId,
  }
}

export function parseInvokeWorkspaceFromAgentDetail(detail: AgentDetailOut): ParsedInvokeWorkspace {
  const base = parseInvokeWorkspaceFromConfigJson(detail.config_json)
  const col = detail.system_prompt?.trim()
  if (col) {
    return { ...base, promptSystem: col }
  }
  return base
}

export type BuildAgentInvokeBodyExtras = {
  /** 与会话行 agent_id 一致；执行方为 detail.id 时可与 owner 不同（同命名空间共享会话） */
  conversationOwnerAgentId?: number | null
}

export function buildAgentInvokeBodyFromDetail(
  detail: AgentDetailOut,
  userMessage: string,
  conversationSessionId: string | null,
  ws: ParsedInvokeWorkspace,
  extras?: BuildAgentInvokeBodyExtras,
): AgentInvokeRequest {
  const stopList = parseStopWordsForApi(ws.stopWords)
  const promptsPayload = buildPromptsPayload(ws.promptSystem)
  const sid = detail.sys_model_id != null ? String(detail.sys_model_id) : undefined
  if (!sid) {
    throw new Error('该 Agent 未绑定系统模型，请先在 Agent 中心配置并保存')
  }
  if (detail.agent_kind !== 'workbench' && !ws.promptSystem.trim()) {
    throw new Error('系统提示词为空，请先在 Agent 中心填写并保存')
  }

  return {
    agent_kind: detail.agent_kind,
    user_message: userMessage,
    // 不填 model_identity：系统模型以 config_id 为准（服务端 from_sys_model）；merge 仅接受 body 的 api_key / base_url_override
    hyperparameters: {
      temperature: ws.temperature,
      max_tokens: ws.maxTokens,
      top_p: ws.topP,
      ...(stopList ? { stop: stopList } : {}),
      ...(ws.seed != null ? { seed: ws.seed } : {}),
      frequency_penalty: ws.frequencyPenalty,
      presence_penalty: ws.presencePenalty,
    },
    response: {
      response_format: ws.responseFormat,
      ...(ws.responseFormat === 'json_schema' && ws.responseJsonSchema
        ? {
            response_json_schema: ws.responseJsonSchema,
            ...(ws.jsonSchemaId ? { json_schema_id: ws.jsonSchemaId } : {}),
          }
        : {}),
    },
    tool_choice: ws.useToolChoice
      ? {
          mode: ws.toolChoiceMode,
          forced_tool_name: ws.toolChoiceMode === 'specific' ? ws.forcedToolName.trim() || null : null,
        }
      : null,
    output: {
      strip_thinking_blocks: ws.stripThinking,
      include_tool_messages_in_raw: ws.includeToolMessagesInRaw,
    },
    config_id: sid,
    tool_names: ws.toolNames.length ? ws.toolNames : undefined,
    ...(promptsPayload ? { prompts: promptsPayload } : {}),
    agent_id: detail.id,
    ...(conversationSessionId ? { conversation_session_id: conversationSessionId } : {}),
    ...(extras?.conversationOwnerAgentId != null
      ? { conversation_owner_agent_id: extras.conversationOwnerAgentId }
      : {}),
    workspace_namespace: detail.workspace_namespace?.trim() || 'default',
  }
}
