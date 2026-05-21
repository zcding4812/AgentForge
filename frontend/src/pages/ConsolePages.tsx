import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Col,
  Collapse,
  Divider,
  Drawer,
  Dropdown,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  notification,
  Pagination,
  Popconfirm,
  Popover,
  Progress,
  Radio,
  Segmented,
  Spin,
  Statistic,
  Row,
  Select,
  Skeleton,
  Slider,
  Space,
  Steps,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from 'antd'
import type { MenuProps, UploadFile } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { TableColumnFilterHeader } from '../components/TableColumnFilterHeader'
import { readJsonBody } from '../api/parseEnvelope'
import {
  ArrowRightOutlined,
  CheckCircleOutlined,
  ClearOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  DownOutlined,
  EditOutlined,
  FileMarkdownOutlined,
  FileOutlined,
  FilePdfOutlined,
  FileTextOutlined,
  HistoryOutlined,
  InfoCircleOutlined,
  EyeOutlined,
  LoadingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
  RedoOutlined,
  ReloadOutlined,
  SearchOutlined,
  SlidersOutlined,
  UploadOutlined,
} from '@ant-design/icons'
import {
  formatAgentStreamProgressLabel,
  streamProgressPhaseBelongsToShellAgent,
} from '../utils/formatAgentStreamProgress'
import { workspaceMarkdownComponents } from '../components/chat/WorkspaceMarkdownMermaid'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent,
  type Key,
  type ReactNode,
  type SetStateAction,
} from 'react'
import type { StepsProps } from 'antd'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  createPersistedHttpTool,
  createPersistedMcpTool,
  deletePersistedHttpTool,
  fetchAgentById,
  fetchAgentsList,
  fetchAgentRegisteredTools,
  fetchPersistedHttpTools,
  updateAgent,
  updatePersistedHttpTool,
  updatePersistedMcpTool,
  probeMcpListTools,
  type AgentDetailOut,
  type AgentHttpToolPersistedOut,
  type AgentOut,
  type AgentRegisteredToolOut,
} from '../api/agentsApi'
import {
  fetchDefaultPromptByKind,
  WORKBENCH_FLOW_PROMPT_TEMPLATE,
  WORKSPACE_DEFAULT_PROMPT_MENU,
} from '../api/systemApi'
import {
  fetchMonitorDashboard,
  type InfraComponentStatus,
  type MonitorDashboardData,
} from '../api/monitorApi'
import {
  fetchAllChatLlmModels,
  fetchAllEmbeddingModels,
  type LlmModel,
} from '../api/providersApi'
import { isWorkbenchToolEnvelope } from '../utils/workbenchToolEnvelope'
import {
  WORKBENCH_STUDIO_CTA_EVENT,
  type WorkbenchStudioCtaDetail,
} from '../utils/workbenchStudioHeaderBridge'
import {
  WORKBENCH_OPEN_CONFIG_QUERY_PARAM,
  workbenchPathForNamespace,
} from '../utils/workbenchRoutes'
import {
  invokeAgent,
  invokeAgentStream,
  type AgentKind,
  type AgentProcessTraceStep,
  type AgentToolHistoryEntry,
  type KnowledgeInvokeCitation,
  type PromptEngineeringBody,
  type ResponseFormat,
} from '../api/agentApi'
import {
  getConversationMessageExecution,
  getConversationSessionDetail,
  listConversationSessions,
  type ConversationMessageExecutionOut,
  type ConversationMessageOut,
  type ConversationSessionOut,
  type PageMeta,
} from '../api/conversationApi'
import { useWorkbench } from '../contexts/useWorkbench'
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  fetchDocumentChunks,
  fetchKnowledgeBase,
  fetchKnowledgeBasesList,
  fetchKnowledgeDocuments,
  patchKnowledgeDocument,
  postBuildKnowledgeVectors,
  postRequeueDocumentIngest,
  searchKnowledgeInKb,
  updateKnowledgeBase,
  uploadKnowledgeDocument,
  type KnowledgeBaseOut,
  type KnowledgeUpdateBody,
  type KnowledgeChunkOut,
  type KnowledgeDocumentChunksData,
  type KnowledgeDocumentOut,
  type KnowledgeSearchData,
  type KnowledgeSearchHit,
} from '../api/knowledgeApi'
import './pages.css'
import './agentListPage.css'

export { AgentListPage } from './AgentListPage'

const { Title, Text, Paragraph } = Typography

const WORKSPACE_CONFIG_VERSION = 1

/** 与后端 ``config_json.memory`` / ``AgentMemorySettings`` 对齐 */
export type MemoryWorkspaceState = {
  maxHistoryRoundsCap: number
  summarizationSysModelId: number | undefined
  /** 存在滚动摘要时，注入模型保留最近若干轮原文（与后端 ``min_tail_raw_rounds`` 一致） */
  minTailRawRounds: number
  /** 链式摘要每批合并的 user 轮数（与后端 ``compress_batch_rounds`` 一致） */
  compressBatchRounds: number
  /** 与后端 ``summary_context_window_tokens`` 一致：用于历史总量 / 窗口 的比例估算 */
  summaryContextWindowTokens: number
  summaryContextRatioThreshold: number
  summaryContextRatioUrgent: number
  summaryMinRoundsSinceLast: number
  summaryMinTokensSinceLast: number
}

const MEMORY_WORKSPACE_DEFAULT: MemoryWorkspaceState = {
  maxHistoryRoundsCap: 20,
  summarizationSysModelId: undefined,
  minTailRawRounds: 6,
  compressBatchRounds: 3,
  summaryContextWindowTokens: 8192,
  summaryContextRatioThreshold: 0.4,
  summaryContextRatioUrgent: 0.7,
  summaryMinRoundsSinceLast: 20,
  summaryMinTokensSinceLast: 1000,
}

/** 与后端 ``config_json.knowledge`` / Agent 知识库绑定一致 */
export type KnowledgeWorkspaceState = {
  enabled: boolean
  /** 最多 8 个知识库 id */
  knowledgeBaseIds: number[]
  /** 每库召回条数，对应检索 ``limit`` */
  topK: number
  retrievalOverride: 'inherit' | 'keyword' | 'vector' | 'hybrid'
  /** 注入 context 时是否为每条分片追加 doc_id / chunk_index 等来源行 */
  showSources: boolean
}

const KNOWLEDGE_WORKSPACE_DEFAULT: KnowledgeWorkspaceState = {
  enabled: false,
  knowledgeBaseIds: [],
  topK: 8,
  retrievalOverride: 'inherit',
  showSources: false,
}

/** 与 backend ``config_json.input_filter`` / 内核 ``InputContentFilterConfig`` 对齐 */
export type InputFilterWorkspaceState = {
  enabled: boolean
  /** 多行编辑；落库时拆为 banned_keywords 数组 */
  banned_keywords_text: string
  banned_regex_text: string
  max_user_chars: number | null
  truncate_on_max: boolean
  reject_message: string
}

const INPUT_FILTER_WORKSPACE_DEFAULT: InputFilterWorkspaceState = {
  enabled: false,
  banned_keywords_text: '',
  banned_regex_text: '',
  max_user_chars: null,
  truncate_on_max: false,
  reject_message: '该输入未通过安全策略，请修改后重试。',
}

const AGENT_WORKSPACE_HISTORY_PAGE_SIZE = 15
/** 与会话详情 API 默认一致：最近一批消息条数 */
const AGENT_WORKSPACE_MESSAGE_PAGE_SIZE = 10

function buildAgentWorkspaceConfigJson(p: {
  temperature: number
  maxTokens: number
  topP: number
  stopWords: string
  /** 留空不传后端 */
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
  /** 第 1 类 System Prompt（与 prompts.system_prompt 对应） */
  promptSystem: string
  memory: MemoryWorkspaceState
  knowledge: KnowledgeWorkspaceState
  inputFilter: InputFilterWorkspaceState
}): Record<string, unknown> {
  return {
    v: WORKSPACE_CONFIG_VERSION,
    temperature: p.temperature,
    max_tokens: p.maxTokens,
    top_p: p.topP,
    stop_words: p.stopWords,
    seed: p.seed,
    frequency_penalty: p.frequencyPenalty,
    presence_penalty: p.presencePenalty,
    response_format: p.responseFormat,
    tool_names: p.toolNames,
    use_tool_choice: p.useToolChoice,
    tool_choice_mode: p.toolChoiceMode,
    forced_tool_name: p.forcedToolName,
    strip_thinking: p.stripThinking,
    stream_output: p.streamOutput,
    include_tool_messages_in_raw: p.includeToolMessagesInRaw,
    prompt_system: p.promptSystem,
    memory: {
      max_history_rounds_cap: p.memory.maxHistoryRoundsCap,
      summarization_sys_model_id: p.memory.summarizationSysModelId ?? null,
      min_tail_raw_rounds: p.memory.minTailRawRounds,
      compress_batch_rounds: p.memory.compressBatchRounds,
      summary_context_window_tokens: p.memory.summaryContextWindowTokens,
      summary_context_ratio_threshold: p.memory.summaryContextRatioThreshold,
      summary_context_ratio_urgent: p.memory.summaryContextRatioUrgent,
      summary_min_rounds_since_last: p.memory.summaryMinRoundsSinceLast,
      summary_min_tokens_since_last: p.memory.summaryMinTokensSinceLast,
    },
    knowledge: {
      enabled: p.knowledge.enabled,
      knowledge_base_ids: p.knowledge.knowledgeBaseIds,
      top_k: p.knowledge.topK,
      retrieval_override: p.knowledge.retrievalOverride,
      show_sources: p.knowledge.showSources,
    },
    input_filter: {
      enabled: p.inputFilter.enabled,
      banned_keywords: p.inputFilter.banned_keywords_text
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean),
      banned_regex: p.inputFilter.banned_regex_text
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean),
      max_user_chars: p.inputFilter.max_user_chars,
      truncate_on_max: p.inputFilter.truncate_on_max,
      reject_message: p.inputFilter.reject_message,
    },
  }
}

function applyAgentWorkspaceConfigJson(
  raw: AgentDetailOut['config_json'],
  setters: {
    setTemperature: (v: number) => void
    setMaxTokens: (v: number) => void
    setTopP: (v: number) => void
    setStopWords: (v: string) => void
    setSeed: (v: number | null) => void
    setFrequencyPenalty: (v: number) => void
    setPresencePenalty: (v: number) => void
    setResponseFormat: (v: ResponseFormat) => void
    setToolNames: (v: string[]) => void
    setUseToolChoice: (v: boolean) => void
    setToolChoiceMode: (v: 'auto' | 'none' | 'required' | 'specific') => void
    setForcedToolName: (v: string) => void
    setStripThinking: (v: boolean) => void
    setStreamOutput: (v: boolean) => void
    setIncludeToolMessagesInRaw: (v: boolean) => void
    setPromptSystem: (v: string) => void
    setMemoryWorkspace: Dispatch<SetStateAction<MemoryWorkspaceState>>
    setKnowledgeWorkspace: Dispatch<SetStateAction<KnowledgeWorkspaceState>>
    setInputFilterWorkspace: Dispatch<SetStateAction<InputFilterWorkspaceState>>
  },
): void {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) return
  const o = raw as Record<string, unknown>
  const num = (v: unknown, d: number) =>
    typeof v === 'number' && Number.isFinite(v) ? v : d
  const str = (v: unknown, d: string) => (typeof v === 'string' ? v : d)
  const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d)
  if ('temperature' in o) setters.setTemperature(num(o.temperature, 0.7))
  if ('max_tokens' in o) setters.setMaxTokens(Math.round(num(o.max_tokens, 4096)))
  if ('top_p' in o) setters.setTopP(num(o.top_p, 0.9))
  if ('stop_words' in o) setters.setStopWords(str(o.stop_words, ''))
  if ('seed' in o) {
    const v = o.seed
    if (v === null) setters.setSeed(null)
    else if (typeof v === 'number' && Number.isFinite(v)) setters.setSeed(Math.round(v))
  }
  if ('frequency_penalty' in o) {
    const v = o.frequency_penalty
    if (v === null || v === undefined) setters.setFrequencyPenalty(0)
    else if (typeof v === 'number' && Number.isFinite(v)) setters.setFrequencyPenalty(v)
  }
  if ('presence_penalty' in o) {
    const v = o.presence_penalty
    if (v === null || v === undefined) setters.setPresencePenalty(0)
    else if (typeof v === 'number' && Number.isFinite(v)) setters.setPresencePenalty(v)
  }
  if ('response_format' in o) {
    const rf = o.response_format
    if (rf === 'text' || rf === 'json_object' || rf === 'json_schema') {
      setters.setResponseFormat(rf)
    }
  }
  if ('tool_names' in o && Array.isArray(o.tool_names) && o.tool_names.every((x) => typeof x === 'string')) {
    setters.setToolNames(o.tool_names as string[])
  }
  if ('use_tool_choice' in o) setters.setUseToolChoice(bool(o.use_tool_choice, false))
  if ('tool_choice_mode' in o) {
    const m = o.tool_choice_mode
    if (m === 'auto' || m === 'none' || m === 'required' || m === 'specific') {
      setters.setToolChoiceMode(m)
    }
  }
  if ('forced_tool_name' in o) setters.setForcedToolName(str(o.forced_tool_name, ''))
  if ('strip_thinking' in o) setters.setStripThinking(bool(o.strip_thinking, true))
  if ('stream_output' in o) setters.setStreamOutput(bool(o.stream_output, false))
  if ('include_tool_messages_in_raw' in o) {
    setters.setIncludeToolMessagesInRaw(bool(o.include_tool_messages_in_raw, false))
  }
  if ('prompt_system_prompt' in o) {
    setters.setPromptSystem(str(o.prompt_system_prompt, ''))
  } else if ('prompt_system' in o || 'prompt_developer' in o || 'prompt_auxiliary' in o) {
    let sys = str(o.prompt_system, '')
    const dev = str(o.prompt_developer, '')
    const aux = str(o.prompt_auxiliary, '')
    if (dev) sys = sys ? `${sys}\n\n## 能力（补充）\n${dev}` : `## 能力（补充）\n${dev}`
    if (aux) sys = sys ? `${sys}\n\n## 约束与输出（补充）\n${aux}` : `## 约束与输出（补充）\n${aux}`
    setters.setPromptSystem(sys)
  }
  if ('memory' in o && o.memory != null && typeof o.memory === 'object' && !Array.isArray(o.memory)) {
    const m = o.memory as Record<string, unknown>
    setters.setMemoryWorkspace((prev) => {
      const cap =
        typeof m.max_history_rounds_cap === 'number' && Number.isFinite(m.max_history_rounds_cap)
          ? Math.max(1, Math.min(200, Math.round(m.max_history_rounds_cap)))
          : prev.maxHistoryRoundsCap
      let sumModel: number | undefined = prev.summarizationSysModelId
      if ('summarization_sys_model_id' in m) {
        const sid = m.summarization_sys_model_id
        if (sid === null || sid === undefined) sumModel = undefined
        else if (typeof sid === 'number' && Number.isFinite(sid) && sid >= 1) sumModel = Math.floor(sid)
      }
      const clampRatio = (v: unknown, d: number) => {
        if (typeof v !== 'number' || !Number.isFinite(v)) return d
        return Math.max(0.05, Math.min(0.95, v))
      }
      const clampInt = (v: unknown, d: number, lo: number, hi: number) => {
        if (typeof v !== 'number' || !Number.isFinite(v)) return d
        return Math.max(lo, Math.min(hi, Math.round(v)))
      }
      return {
        ...prev,
        maxHistoryRoundsCap: cap,
        summarizationSysModelId: sumModel,
        summaryContextWindowTokens: clampInt(
          m.summary_context_window_tokens,
          prev.summaryContextWindowTokens,
          512,
          2_000_000,
        ),
        summaryContextRatioThreshold: clampRatio(
          m.summary_context_ratio_threshold,
          prev.summaryContextRatioThreshold,
        ),
        summaryContextRatioUrgent: clampRatio(
          m.summary_context_ratio_urgent,
          prev.summaryContextRatioUrgent,
        ),
        summaryMinRoundsSinceLast: clampInt(
          m.summary_min_rounds_since_last,
          prev.summaryMinRoundsSinceLast,
          1,
          500,
        ),
        summaryMinTokensSinceLast: clampInt(
          m.summary_min_tokens_since_last,
          prev.summaryMinTokensSinceLast,
          100,
          500000,
        ),
        minTailRawRounds: clampInt(m.min_tail_raw_rounds, prev.minTailRawRounds, 1, 64),
        compressBatchRounds: clampInt(m.compress_batch_rounds, prev.compressBatchRounds, 1, 64),
      }
    })
  }
  if ('knowledge' in o && o.knowledge != null && typeof o.knowledge === 'object' && !Array.isArray(o.knowledge)) {
    const k = o.knowledge as Record<string, unknown>
    setters.setKnowledgeWorkspace((prev) => {
      const parseIds = (): number[] => {
        const raw = k.knowledge_base_ids
        if (!Array.isArray(raw)) return prev.knowledgeBaseIds
        const out: number[] = []
        for (const x of raw) {
          if (typeof x === 'number' && Number.isFinite(x) && x >= 1) out.push(Math.floor(x))
          else if (typeof x === 'string' && /^\d+$/.test(x)) out.push(Number(x))
          if (out.length >= 8) break
        }
        const seen = new Set<number>()
        return out.filter((id) => (seen.has(id) ? false : (seen.add(id), true)))
      }
      let topK = prev.topK
      if (typeof k.top_k === 'number' && Number.isFinite(k.top_k)) {
        topK = Math.max(1, Math.min(30, Math.round(k.top_k)))
      }
      let ro: KnowledgeWorkspaceState['retrievalOverride'] = prev.retrievalOverride
      if (typeof k.retrieval_override === 'string') {
        const s = k.retrieval_override.trim().toLowerCase()
        if (s === 'inherit' || s === 'keyword' || s === 'vector' || s === 'hybrid') ro = s
      }
      return {
        ...prev,
        enabled: typeof k.enabled === 'boolean' ? k.enabled : prev.enabled,
        knowledgeBaseIds: 'knowledge_base_ids' in k ? parseIds() : prev.knowledgeBaseIds,
        topK,
        retrievalOverride: ro,
        showSources: typeof k.show_sources === 'boolean' ? k.show_sources : prev.showSources,
      }
    })
  }
  if (
    'input_filter' in o &&
    o.input_filter != null &&
    typeof o.input_filter === 'object' &&
    !Array.isArray(o.input_filter)
  ) {
    const inf = o.input_filter as Record<string, unknown>
    setters.setInputFilterWorkspace((prev) => {
      const next: InputFilterWorkspaceState = { ...prev }
      if ('enabled' in inf) next.enabled = bool(inf.enabled, false)
      if (Array.isArray(inf.banned_keywords)) {
        const kws = (inf.banned_keywords as unknown[])
          .filter((x): x is string => typeof x === 'string')
          .map((s) => s.trim())
          .filter(Boolean)
        next.banned_keywords_text = kws.join('\n')
      }
      if (Array.isArray(inf.banned_regex)) {
        const pats = (inf.banned_regex as unknown[])
          .filter((x): x is string => typeof x === 'string')
          .map((s) => s.trim())
          .filter(Boolean)
        next.banned_regex_text = pats.join('\n')
      }
      if ('max_user_chars' in inf) {
        if (inf.max_user_chars === null || inf.max_user_chars === undefined) {
          next.max_user_chars = null
        } else if (typeof inf.max_user_chars === 'number' && Number.isFinite(inf.max_user_chars)) {
          const n = Math.round(inf.max_user_chars)
          next.max_user_chars = n >= 0 ? n : null
        }
      }
      if ('truncate_on_max' in inf) next.truncate_on_max = bool(inf.truncate_on_max, false)
      if (typeof inf.reject_message === 'string' && inf.reject_message.trim()) {
        next.reject_message = inf.reject_message.trim()
      }
      return next
    })
  } else {
    setters.setInputFilterWorkspace(INPUT_FILTER_WORKSPACE_DEFAULT)
  }
}

/** 仅传 system；多轮正文由服务端按 conversation_session_id 从 DB 注入，不拼 chat_history */
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

/** 仅标题 + key 旁「?」说明（如 Seed，控件在下方独占一行） */
function SamplingLabelWithHint({ title, hint }: { title: ReactNode; hint: ReactNode }) {
  return (
    <div className="agent-workspace-sampling-label-with-hint-only">
      <span className="agent-workspace-sampling-label-key">{title}</span>
      <Tooltip title={hint} overlayClassName="agent-workspace-sampling-hint-tooltip">
        <QuestionCircleOutlined
          className="agent-workspace-sampling-label-hint-icon"
          tabIndex={0}
          aria-label="说明"
        />
      </Tooltip>
    </div>
  )
}

/** 标签行：标题 +「?」说明 + 右侧可编辑数值（下方单独放 Slider） */
function SamplingFieldLabelWithValue({
  title,
  hint,
  min,
  max,
  step,
  value,
  onChange,
  fallbackOnInvalid,
  valueWidth = 56,
}: {
  title: ReactNode
  /** 悬停「?」展示的说明（原 Form.Item extra） */
  hint?: ReactNode
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
  fallbackOnInvalid: number
  /** InputNumber 宽度（px，宜紧凑；最大令牌数可略宽） */
  valueWidth?: number
}) {
  return (
    <div className="agent-workspace-sampling-label-with-value">
      <div className="agent-workspace-sampling-label-left">
        <span className="agent-workspace-sampling-label-key">{title}</span>
        {hint != null && (
          <Tooltip title={hint} overlayClassName="agent-workspace-sampling-hint-tooltip">
            <QuestionCircleOutlined
              className="agent-workspace-sampling-label-hint-icon"
              tabIndex={0}
              aria-label="说明"
            />
          </Tooltip>
        )}
      </div>
      <div className="agent-workspace-sampling-inline-value-wrap">
        <InputNumber
          variant="filled"
          size="small"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(v) => {
            if (typeof v === 'number' && !Number.isNaN(v)) onChange(v)
            else onChange(fallbackOnInvalid)
          }}
          controls={false}
          style={{ width: valueWidth }}
          className="agent-workspace-sampling-inline-value"
        />
      </div>
    </div>
  )
}

type BasicConfigBaseline = {
  agentDisplayName: string
  agentDescription: string
  agentKind: AgentKind
  systemModelId: string | undefined
  temperature: number
  maxTokens: number
  topP: number
  stopWords: string
  promptSystem: string
}

function defaultBasicBaseline(agentId: string | undefined): BasicConfigBaseline {
  return {
    agentDisplayName: agentId?.trim() ?? '',
    agentDescription: '',
    agentKind: 'simple_chat',
    systemModelId: undefined,
    temperature: 0.7,
    maxTokens: 4096,
    topP: 0.9,
    stopWords: '',
    promptSystem: '',
  }
}

function workspaceAgentKindLabel(k: AgentKind): string {
  const m: Record<AgentKind, string> = {
    simple_chat: '简单对话',
    react: 'ReAct',
    plan_execute: 'Plan & Execute',
    workbench: '工作区',
  }
  return m[k] ?? k
}

/** 基础配置：Form.Item 标签（标题 + 必填星 + 悬浮说明） */
function AgentWorkspaceFieldLabel({
  title,
  required,
  tip,
}: {
  title: string
  required?: boolean
  tip: string
}) {
  return (
    <Space size={4}>
      <span className="agent-workspace-basic-block__title agent-workspace-basic-block__title--form">
        {title}
        {required ? <span className="agent-workspace-required-mark">*</span> : null}
      </span>
      <Tooltip title={tip}>
        <QuestionCircleOutlined className="agent-workspace-field-help" aria-label={tip} />
      </Tooltip>
    </Space>
  )
}

/** 后端 ``knowledge_base.status`` → 列表展示文案 */
function knowledgeStatusLabel(status: string): string {
  const m: Record<string, string> = {
    empty: '空库',
    ready: '已就绪',
    processing: '待切块',
    draft: '草稿',
    failed: '失败',
  }
  return m[status] ?? status
}

function knowledgeStatusFilterToApi(label: string): string | undefined {
  if (label === '全部') return undefined
  if (label === '已就绪') return 'ready'
  if (label === '待切块') return 'processing'
  if (label === '空库') return 'empty'
  return undefined
}

/** 知识库详情步骤（四模块） */
const KNOWLEDGE_STEP_DEFINITIONS = [
  { title: '文档管理' },
  { title: '切块策略' },
  { title: '索引配置' },
  { title: '检索验证' },
] as const

/** 上传文件名允许的扩展名（与 Dragger accept 一致） */
const KNOWLEDGE_UPLOAD_EXT_RE = /\.(md|txt|pdf|docx|json|html|htm)$/i

/** Ant Design Upload 在部分环境下仅提供 `file` 而非 `originFileObj` */
function knowledgeUploadRawFile(f: UploadFile): File | undefined {
  if (f.originFileObj instanceof File) return f.originFileObj
  const alt = f as UploadFile & { file?: File }
  if (alt.file instanceof File) return alt.file
  return undefined
}

/** 从 ``config_json.retrieval.top_k`` 读取；与 ``POST .../search`` 的 ``limit`` 一致 */
function parseKnowledgeRetrievalTopK(
  configJson: Record<string, unknown> | null | undefined,
): number {
  const raw = configJson?.retrieval
  if (!raw || typeof raw !== 'object') return 20
  const top = Number((raw as Record<string, unknown>).top_k)
  if (Number.isFinite(top) && top >= 1 && top <= 100) return top
  return 20
}

/** 默认 RRF k（与后端 ``HYBRID_RRF_K`` 一致） */
function parseKnowledgeRetrievalRrf(
  configJson: Record<string, unknown> | null | undefined,
): number {
  const raw = configJson?.retrieval
  if (!raw || typeof raw !== 'object') return 60
  const k = Number((raw as Record<string, unknown>).rrf_k)
  if (Number.isFinite(k) && k >= 1 && k <= 200) return k
  return 60
}

function knowledgeDocFormatLabel(doc: KnowledgeDocumentOut): string {
  const m = (doc.mime ?? '').trim()
  if (m) {
    const short = m.split('/').pop() ?? m
    return short.length > 14 ? `${short.slice(0, 12)}…` : short
  }
  const p = doc.filename.lastIndexOf('.')
  return p >= 0 ? doc.filename.slice(p + 1).toLowerCase() : '—'
}

function renderSnippetWithHighlight(text: string | null, query: string): ReactNode {
  const t = text ?? ''
  const q = query.trim()
  if (!q || !t) return t || '—'
  const lower = t.toLowerCase()
  const qi = lower.indexOf(q.toLowerCase())
  if (qi < 0) return t
  const end = qi + q.length
  return (
    <>
      {t.slice(0, qi)}
      <mark className="knowledge-search-hit-mark">{t.slice(qi, end)}</mark>
      {t.slice(end)}
    </>
  )
}

/** 切块场景预设（引导参数，可再手动改） */
const CHUNK_SCENARIO_PRESETS: {
  key: string
  label: string
  chunk_method: string
  chunk_size: number
  chunk_overlap: number
}[] = [
  { key: 'general', label: '通用文本', chunk_method: 'markdown', chunk_size: 512, chunk_overlap: 50 },
  { key: 'code', label: '代码 / 文档', chunk_method: 'code', chunk_size: 1024, chunk_overlap: 100 },
  {
    key: 'long_text',
    label: '长文分段',
    chunk_method: 'char',
    chunk_size: 800,
    chunk_overlap: 80,
  },
]

/** 知识库存储 / 检索方式（与 ``KnowledgeUpdateBody`` 一致） */
const KNOWLEDGE_INDEX_MODE_OPTIONS: {
  label: string
  value: KnowledgeUpdateBody['storage_type']
  description: string
}[] = [
  {
    label: '关键词检索（精确匹配）',
    value: 'keyword',
    description: '基于分片文本匹配，适合专有名词、条款号等',
  },
  {
    label: '向量检索（语义匹配）',
    value: 'vector',
    description: '按语义相似度召回，需配置向量模型与 Milvus',
  },
  {
    label: '混合检索（推荐）',
    value: 'hybrid',
    description: '关键词 + 向量 RRF 融合，兼顾字面与语义',
  },
]

/** 与后端 ``CHUNK_METHOD_KEYS`` / ingest 策略对齐 */
const KNOWLEDGE_CHUNK_METHOD_BASE_OPTIONS: { label: string; value: string }[] = [
  { label: '按字符长度', value: 'char' },
  { label: '词元', value: 'token' },
  { label: '句边界', value: 'sentence' },
  { label: 'Markdown 标题分节', value: 'markdown' },
  { label: 'JSON 结构分节', value: 'json' },
  { label: 'HTML 标签分节', value: 'html' },
  { label: '代码 AST', value: 'code' },
]

function knowledgeChunkMethodHint(method: string): string {
  const hints: Record<string, string> = {
    length:
      '与「按字符长度」相同；历史数据可能仍存为 length，保存后会写为 char。',
    char:
      '按字符数切分：每块最多 chunk_size 个字符，相邻块共享 chunk_overlap 个字符；可先按分隔符拆段再窗口切分。',
    token: '块大小按约 token 理解；仍映射为后端 chunk_size 数值区间。',
    sentence: '优先在句读处断开，块大小为上限。',
    markdown: '按 Markdown 标题拆大段，过大时在块内再按窗口二次切分。（保存为 markdown 与历史数据 md 行为一致。）',
    json: '尽量在对象/数组边界保持结构完整，过大时二次切分。',
    html: '按 HTML 标签层级拆节，过大时二次切分。',
    code: '按扩展名推断语言走语法树；无扩展名时效果可能下降。',
  }
  return hints[method] ?? '保存后对新上传文档的 ingest 生效；已入库文档重切需后台任务。'
}

function knowledgeChunkSizeLabel(method: string): string {
  if (method === 'token' || method === 'sentence') return '目标块大小（上限）'
  if (method === 'markdown' || method === 'md' || method === 'json' || method === 'html') {
    return '大段内的二次切分窗口（字符）'
  }
  if (method === 'code') return '语法块后的二次切分窗口（字符）'
  return '目标块大小（字符）'
}

function knowledgeChunkOverlapLabel(method: string): string {
  if (method === 'token' || method === 'sentence') return '重叠（上限）'
  if (method === 'markdown' || method === 'md' || method === 'json' || method === 'html') {
    return '二次切分重叠（字符）'
  }
  if (method === 'code') return '二次切分重叠（字符）'
  return '块重叠（字符）'
}

function knowledgeChunkMethodShowsSeparator(method: string): boolean {
  return method === 'length' || method === 'char' || method === 'token' || method === 'sentence'
}

const knowledgeChunkPreviewColumns: ColumnsType<KnowledgeChunkOut> = [
  { title: '序号', dataIndex: 'chunk_index', key: 'chunk_index', width: 72 },
  {
    title: '文本',
    dataIndex: 'text',
    key: 'text',
    ellipsis: true,
    render: (t: string) => (t ? t : '—'),
  },
  {
    title: '创建时间',
    dataIndex: 'created_at',
    key: 'created_at',
    width: 180,
    render: (v: string | null) => (v ? v.replace('T', ' ').slice(0, 19) : '—'),
  },
]

type TraceListItem = {
  trace_name: string
  trace_id: string
  session_id: string
  agent_task_id: string
  duration_ms: number
  status: 'success' | 'failed' | 'running'
  executed_at: string
}

type TraceListStats = {
  total_count: number
  success_count: number
  failed_count: number
  running_count: number
  success_rate: number
  avg_duration_ms: number
  p95_duration_ms: number
}

type TraceListMeta = {
  page: number
  page_size: number
  total: number
}

type TraceListData = {
  items: TraceListItem[]
  stats: TraceListStats
  meta: TraceListMeta
}

type ChatRole = 'user' | 'assistant'

type ChatLine = {
  role: ChatRole
  content: string
  /** 与正文分离的思考文本（服务端解析或 metadata.thinking_text） */
  thinkingText?: string
  /** 多段助手输出回看（metadata.process_trace / invoke 响应） */
  processTrace?: AgentProcessTraceStep[]
  /** 消息时间（ISO 8601），用于列表展示 */
  at?: string
  /** 库消息 id；客户端临时插入的行可能无 */
  sourceMessageId?: number
  /** assistant：指向 user 消息 id（metadata.reply_message_id） */
  replyMessageId?: number | null
  /** user：metadata.user_display_name 或发送时客户端称呼 */
  userDisplayName?: string
  /** assistant：本条消息产生的 Agent（ConversationMessageOut.agent_id）；流式占位行可缺省 */
  responderAgentId?: number
  structured?: Record<string, unknown> | null
  /** 本轮助手消息对应的知识库检索来源（服务端注入上下文时的分片列表） */
  knowledgeCitations?: KnowledgeInvokeCitation[]
  /** 工具调用与 ToolMessage 结果（metadata.tool_history / invoke.tool_history） */
  toolHistory?: AgentToolHistoryEntry[]
}

function formatChatLineTime(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** 工作台 hub：按消息 agent_id 解析展示名；未在列表中则 Agent #id；无 id 时用当前页 Agent 名（流式占位） */
function resolveResponderAgentLabel(
  agentId: number | undefined,
  agentsInNs: AgentOut[],
  pageAgentDisplayName?: string,
): string {
  if (agentId != null && Number.isFinite(agentId) && agentId > 0) {
    const hit = agentsInNs.find((a) => a.id === agentId)
    const n = hit?.name?.trim()
    if (n) return n
    return `Agent #${agentId}`
  }
  const fb = pageAgentDisplayName?.trim()
  if (fb) return fb
  return '助手'
}

function parseUserDisplayNameFromMetadata(
  meta: Record<string, unknown> | null | undefined,
): string | undefined {
  if (!meta || typeof meta.user_display_name !== 'string') return undefined
  const s = meta.user_display_name.trim()
  return s || undefined
}

function chatLineTimestamp(): string {
  return new Date().toISOString()
}

function AssistantStructuredBlock({ data }: { data: Record<string, unknown> }) {
  if (isWorkbenchToolEnvelope(data)) {
    const ok = data.code === 'OK'
    return (
      <div className="agent-workspace-msg__structured agent-workspace-msg__envelope">
        <Space wrap size={8} style={{ marginBottom: 8 }}>
          <Tag color={ok ? 'success' : 'error'}>{data.code}</Tag>
          <Text type="secondary">{data.message}</Text>
        </Space>
        <pre className="agent-structured-json">{JSON.stringify(data.data, null, 2)}</pre>
      </div>
    )
  }
  return <pre className="agent-workspace-msg__structured agent-structured-json">{JSON.stringify(data, null, 2)}</pre>
}

/**
 * 是否再渲染主气泡下的「结构化」块。与正文为同一份 JSON 对象时只保留主文（默认一段），避免两段重复。
 * 工作台工具「信封」等仍保留下方特化区。
 */
function shouldShowAssistantStructuredBlock(m: ChatLine): boolean {
  if (m.role !== 'assistant' || m.structured == null || Object.keys(m.structured).length === 0) {
    return false
  }
  if (isWorkbenchToolEnvelope(m.structured as Record<string, unknown>)) {
    return true
  }
  const t = m.content?.trim() ?? ''
  if (t.length < 2 || t[0] !== '{') {
    return true
  }
  try {
    const a: unknown = JSON.parse(t)
    if (a !== null && typeof a === 'object' && !Array.isArray(a)) {
      if (JSON.stringify(a) === JSON.stringify(m.structured)) {
        return false
      }
    }
  } catch {
    return true
  }
  return true
}

function parseProcessTraceFromMetadata(
  meta: Record<string, unknown> | null | undefined,
): AgentProcessTraceStep[] | undefined {
  if (!meta || typeof meta !== 'object') return undefined
  const raw = meta.process_trace
  if (!Array.isArray(raw) || raw.length < 1) return undefined
  const out: AgentProcessTraceStep[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const o = item as Record<string, unknown>
    if (typeof o.seq !== 'number') continue
    const text =
      o.text == null ? '' : typeof o.text === 'string' ? o.text : String(o.text)
    const phase = o.phase === 'final' || o.phase === 'step' ? o.phase : 'step'
    const kind =
      o.kind === 'tool_call' || o.kind === 'tool_result' || o.kind === 'model'
        ? o.kind
        : 'model'
    const row: AgentProcessTraceStep = { seq: o.seq, text, phase, kind }
    if (o.tool_call_id != null) {
      row.tool_call_id =
        typeof o.tool_call_id === 'string' ? o.tool_call_id : String(o.tool_call_id)
    }
    if (o.name != null) {
      row.name = typeof o.name === 'string' ? o.name : String(o.name)
    }
    if (Object.prototype.hasOwnProperty.call(o, 'arguments')) {
      row.arguments = o.arguments
    }
    if (typeof o.content === 'string') {
      row.content = o.content
    }
    out.push(row)
  }
  return out.length >= 1 ? out : undefined
}

function parseToolHistoryFromMetadata(
  meta: Record<string, unknown> | null | undefined,
): AgentToolHistoryEntry[] | undefined {
  if (!meta || typeof meta !== 'object') return undefined
  const raw = meta.tool_history
  if (!Array.isArray(raw) || raw.length === 0) return undefined
  const out: AgentToolHistoryEntry[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue
    const o = item as Record<string, unknown>
    if (typeof o.seq !== 'number') continue
    if (o.phase !== 'call' && o.phase !== 'result') continue
    const toolCallId =
      o.tool_call_id == null
        ? null
        : typeof o.tool_call_id === 'string'
          ? o.tool_call_id
          : String(o.tool_call_id)
    const name =
      o.name == null ? null : typeof o.name === 'string' ? o.name : String(o.name)
    out.push({
      seq: o.seq,
      phase: o.phase,
      tool_call_id: toolCallId,
      name,
      ...(Object.prototype.hasOwnProperty.call(o, 'arguments') ? { arguments: o.arguments } : {}),
      ...(typeof o.content === 'string' ? { content: o.content } : {}),
    })
  }
  return out.length > 0 ? out : undefined
}

function parseExecutionApiToChatAux(
  data: ConversationMessageExecutionOut,
): {
  steps?: AgentProcessTraceStep[]
  toolEntries?: AgentToolHistoryEntry[]
} {
  const steps = parseProcessTraceFromMetadata({
    process_trace: data.process_trace as unknown,
  })
  const toolEntries = parseToolHistoryFromMetadata({
    tool_history: data.tool_history as unknown,
  })
  return {
    ...(steps != null && steps.length >= 1 ? { steps } : {}),
    ...(toolEntries != null && toolEntries.length > 0 ? { toolEntries } : {}),
  }
}

function storedMessageToChatLine(msg: ConversationMessageOut): ChatLine {
  const at = msg.created_at
  let structured: Record<string, unknown> | null | undefined
  let knowledgeCitations: KnowledgeInvokeCitation[] | undefined
  let thinkingText: string | undefined
  let processTrace: AgentProcessTraceStep[] | undefined
  let toolHistory: AgentToolHistoryEntry[] | undefined
  const meta = msg.metadata
  const metaObj = meta && typeof meta === 'object' ? (meta as Record<string, unknown>) : null
  if (metaObj) {
    if ('structured' in metaObj && metaObj.structured != null) {
      const s = metaObj.structured
      if (s && typeof s === 'object' && !Array.isArray(s)) {
        structured = s as Record<string, unknown>
      }
    }
    if ('knowledge_citations' in metaObj && Array.isArray(metaObj.knowledge_citations)) {
      knowledgeCitations = metaObj.knowledge_citations as KnowledgeInvokeCitation[]
    }
    if ('thinking_text' in metaObj && typeof metaObj.thinking_text === 'string') {
      const t = metaObj.thinking_text.trim()
      if (t) thinkingText = t
    }
    processTrace = parseProcessTraceFromMetadata(metaObj)
    toolHistory = parseToolHistoryFromMetadata(metaObj)
  }
  const userDisplayName =
    msg.role === 'user' ? parseUserDisplayNameFromMetadata(metaObj) : undefined

  if (msg.role === 'user') {
    return {
      role: 'user',
      content: msg.content,
      at,
      sourceMessageId: msg.id,
      ...(userDisplayName ? { userDisplayName } : {}),
    }
  }
  const responderAgentId =
    msg.agent_id != null && msg.agent_id > 0 ? msg.agent_id : undefined
  if (msg.role === 'system') {
    return {
      role: 'assistant',
      content: msg.content.trim() ? `【系统】${msg.content}` : msg.content,
      at,
      sourceMessageId: msg.id,
      ...(responderAgentId != null ? { responderAgentId } : {}),
    }
  }
  return {
    role: 'assistant',
    content: msg.content,
    at,
    sourceMessageId: msg.id,
    replyMessageId: msg.reply_message_id ?? null,
    ...(responderAgentId != null ? { responderAgentId } : {}),
    ...(structured != null ? { structured } : {}),
    ...(thinkingText != null ? { thinkingText } : {}),
    ...(processTrace != null && processTrace.length >= 1 ? { processTrace } : {}),
    ...(knowledgeCitations != null && knowledgeCitations.length > 0 ? { knowledgeCitations } : {}),
    ...(toolHistory != null && toolHistory.length > 0 ? { toolHistory } : {}),
  }
}

/** Agent 中心等：保持接口返回的时间顺序 */
function mapStoredMessagesToChatLines(raw: ConversationMessageOut[]): ChatLine[] {
  return raw.map(storedMessageToChatLine)
}

/**
 * 工作台：按用户提问（user 行 created_at / id）排序，其下紧跟 reply_message_id 指向该 user 的 assistant。
 */
function orderWorkbenchChatFromStored(raw: ConversationMessageOut[]): ChatLine[] {
  const users = raw
    .filter((m) => m.role === 'user')
    .sort((a, b) => {
      const t = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      if (t !== 0) return t
      return a.id - b.id
    })
  const systems = raw
    .filter((m) => m.role === 'system')
    .sort((a, b) => a.id - b.id)
  const usedAssistantIds = new Set<number>()
  const out: ChatLine[] = []
  for (const s of systems) {
    out.push(storedMessageToChatLine(s))
  }
  for (const u of users) {
    out.push(storedMessageToChatLine(u))
    const replies = raw
      .filter((m) => m.role === 'assistant' && m.reply_message_id === u.id)
      .sort((a, b) => {
        const t = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        if (t !== 0) return t
        return a.id - b.id
      })
    for (const r of replies) {
      usedAssistantIds.add(r.id)
      out.push(storedMessageToChatLine(r))
    }
  }
  const orphans = raw.filter((m) => m.role === 'assistant' && !usedAssistantIds.has(m.id))
  orphans.sort((a, b) => a.id - b.id)
  for (const o of orphans) {
    out.push(storedMessageToChatLine(o))
  }
  return out
}

function ChatMarkdownBody({ text }: { text: string }) {
  return (
    <div className="agent-workspace-msg__md">
      <Markdown remarkPlugins={[remarkGfm]} components={workspaceMarkdownComponents}>
        {text}
      </Markdown>
    </div>
  )
}

/** 与后端 `AgentOutputModule` 一致：剥离标签得到正文，并汇总 thinking 片段 */
function splitAssistantThinkingAndBody(raw: string): { thinking: string; body: string } {
  const blockRe =
    /<redacted_thinking>[\s\S]*?<\/redacted_thinking>|<thinking>[\s\S]*?<\/thinking>/gi
  const innerRe =
    /<redacted_thinking>([\s\S]*?)<\/redacted_thinking>|<thinking>([\s\S]*?)<\/thinking>/gi
  const parts: string[] = []
  let m: RegExpExecArray | null
  while ((m = innerRe.exec(raw)) !== null) {
    const inner = (m[1] ?? m[2] ?? '').trim()
    if (inner) parts.push(inner)
  }
  const body = raw.replace(blockRe, '').replace(/\n{3,}/g, '\n\n').trim()
  return { thinking: parts.join('\n\n'), body }
}

function AssistantBubbleMarkdown({
  text,
  thinkingText: thinkingFromApi,
}: {
  text: string
  /** 服务端已分离的 thinking；有则不再从 text 内解析标签 */
  thinkingText?: string
}) {
  const thinkingScrollRef = useRef<HTMLDivElement>(null)
  const { thinking, body } = useMemo(() => {
    if (thinkingFromApi != null && thinkingFromApi.trim() !== '') {
      return { thinking: thinkingFromApi.trim(), body: text }
    }
    return splitAssistantThinkingAndBody(text)
  }, [text, thinkingFromApi])
  const markdownSource = thinking ? body : text

  /** 可视区仅约 3 行高，内部可滚动；流式追加时保持滚到底部以始终看到最新片段 */
  useLayoutEffect(() => {
    if (!thinking) return
    const el = thinkingScrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [thinking])

  return (
    <>
      {thinking ? (
        <div className="agent-workspace-msg__thinking">
          <Text type="secondary" className="agent-workspace-msg__thinking-label">
            思考过程
          </Text>
          <div ref={thinkingScrollRef} className="agent-workspace-msg__thinking-viewport">
            <pre className="agent-workspace-msg__thinking-pre">{thinking}</pre>
          </div>
        </div>
      ) : null}
      <ChatMarkdownBody text={markdownSource} />
    </>
  )
}

type WorkspaceChatAuxDrawer =
  | {
      kind: 'execution_process'
      steps?: AgentProcessTraceStep[]
      toolEntries?: AgentToolHistoryEntry[]
      loading?: boolean
      error?: string | null
    }
  | { kind: 'knowledge'; citations: KnowledgeInvokeCitation[] }

function WorkspaceAuxDetailTag({ label, onOpen }: { label: ReactNode; onOpen: () => void }) {
  const onKeyDown = (e: KeyboardEvent<HTMLSpanElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onOpen()
    }
  }
  return (
    <Tag
      color="blue"
      className="agent-workspace-msg__aux-tag agent-workspace-msg__aux-tag--clickable"
      tabIndex={0}
      role="button"
      onClick={onOpen}
      onKeyDown={onKeyDown}
    >
      {label}
    </Tag>
  )
}

const PROCESS_TRACE_EMPTY_HINT =
  '本段无模型正文（常见于仅下发工具调用的中间轮）。下方「工具」步骤含调用名、入参与返回。'

function processTraceStepLabel(s: AgentProcessTraceStep): { label: string; isEmpty: boolean } {
  const body = (s.text ?? '').trim()
  const isEmpty = body.length === 0
  const stepNo = s.seq + 1
  const k = s.kind ?? 'model'
  if (k === 'tool_call') {
    const nm = s.name?.trim() || '工具'
    return {
      label: isEmpty ? `步骤 ${stepNo} · 工具调用 · ${nm} · 无正文` : `步骤 ${stepNo} · 工具调用 · ${nm}`,
      isEmpty,
    }
  }
  if (k === 'tool_result') {
    const nm = s.name?.trim() || '工具'
    return {
      label: isEmpty ? `步骤 ${stepNo} · 工具返回 · ${nm} · 无正文` : `步骤 ${stepNo} · 工具返回 · ${nm}`,
      isEmpty,
    }
  }
  const baseLabel =
    s.phase === 'final' ? `步骤 ${stepNo} · 模型输出（最终）` : `步骤 ${stepNo} · 模型输出`
  return {
    label: isEmpty ? `${baseLabel} · 无正文` : baseLabel,
    isEmpty,
  }
}

function ProcessTraceDrawerBody({ steps }: { steps: AgentProcessTraceStep[] }) {
  return (
    <Collapse
      bordered={false}
      size="small"
      className="agent-workspace-msg__process-trace-collapse agent-workspace-msg__process-trace-collapse--drawer"
      items={steps.map((s) => {
        const { label, isEmpty } = processTraceStepLabel(s)
        return {
          key: `pt-${s.seq}-${s.kind ?? 'model'}`,
          label,
          children: isEmpty ? (
            <Text type="secondary" className="agent-workspace-msg__process-trace-empty">
              {PROCESS_TRACE_EMPTY_HINT}
            </Text>
          ) : (
            <div className="agent-workspace-msg__md">
              <Markdown remarkPlugins={[remarkGfm]} components={workspaceMarkdownComponents}>
                {s.text}
              </Markdown>
            </div>
          ),
        }
      })}
    />
  )
}

function ExecutionProcessDrawerBody({
  steps,
  toolEntries,
  loading,
  error,
}: {
  steps?: AgentProcessTraceStep[]
  toolEntries?: AgentToolHistoryEntry[]
  loading?: boolean
  error?: string | null
}) {
  if (loading) {
    return (
      <Space direction="vertical" align="center" style={{ width: '100%', padding: '48px 0' }}>
        <Spin />
        <Text type="secondary">正在加载该条消息的执行过程…</Text>
      </Space>
    )
  }
  if (error) {
    return <Alert type="error" message={error} showIcon />
  }
  const hasPt = steps != null && steps.length >= 1
  const hasTh = toolEntries != null && toolEntries.length > 0
  const traceHasToolRows =
    steps?.some((s) => s.kind === 'tool_call' || s.kind === 'tool_result') ?? false
  const showToolSection = hasTh && !traceHasToolRows
  if (!hasPt && !hasTh) {
    return (
      <Empty
        description="该条助手消息暂无已落库的执行过程（process_trace / tool_history）"
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    )
  }
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {hasPt ? (
        <div>
          <Text strong className="agent-workspace-msg__execution-section-title">
            {traceHasToolRows ? '执行过程（模型与工具）' : '模型输出'}
          </Text>
          <ProcessTraceDrawerBody steps={steps!} />
        </div>
      ) : null}
      {showToolSection ? (
        <div>
          <Text strong className="agent-workspace-msg__execution-section-title">
            工具
          </Text>
          <ToolHistoryDrawerBody entries={toolEntries!} />
        </div>
      ) : null}
      {!hasPt && !hasTh ? <Empty description="暂无可展示的执行细节" /> : null}
    </Space>
  )
}

function KnowledgeCitationsDrawerBody({ citations }: { citations: KnowledgeInvokeCitation[] }) {
  return (
    <ul className="agent-workspace-msg__citation-list">
      {citations.map((c, idx) => (
        <li key={`${c.kb_id}-${c.doc_id}-${c.chunk_index ?? 'x'}-${idx}`}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {c.kb_name} · {c.filename?.trim() || `文档 #${c.doc_id}`}
            {c.chunk_index != null ? ` · 分片 #${c.chunk_index}` : ''}
          </Text>
          {c.text_snippet?.trim() ? (
            <Paragraph
              ellipsis={{ rows: 3, expandable: true }}
              style={{ marginBottom: 0, marginTop: 4, fontSize: 12 }}
            >
              {c.text_snippet.trim()}
            </Paragraph>
          ) : null}
        </li>
      ))}
    </ul>
  )
}

type ToolHistoryPairedRow =
  | {
      kind: 'invocation'
      step: number
      call: AgentToolHistoryEntry
      result: AgentToolHistoryEntry | null
    }
  | { kind: 'result_only'; step: number; result: AgentToolHistoryEntry }

/** 将时间序的 call/result 合并为「一步 · 工具 · 入参+结果」，便于与「步骤 N · 模型输出」对照 */
function pairToolHistoryForDisplay(entries: AgentToolHistoryEntry[]): ToolHistoryPairedRow[] {
  const sorted = [...entries].sort((a, b) => a.seq - b.seq)
  const n = sorted.length
  const used = new Set<number>()
  const out: ToolHistoryPairedRow[] = []
  let toolStep = 0

  for (let i = 0; i < n; i++) {
    if (used.has(i)) continue
    const e = sorted[i]
    if (e.phase === 'call') {
      toolStep += 1
      const tid =
        e.tool_call_id != null && String(e.tool_call_id).trim() !== ''
          ? String(e.tool_call_id)
          : null
      let pairJ = -1
      for (let j = i + 1; j < n; j++) {
        if (used.has(j)) continue
        const r = sorted[j]
        if (r.phase !== 'result') continue
        const rtid =
          r.tool_call_id != null && String(r.tool_call_id).trim() !== ''
            ? String(r.tool_call_id)
            : null
        if (tid != null && rtid != null && rtid !== tid) continue
        pairJ = j
        break
      }
      if (pairJ >= 0) {
        used.add(i)
        used.add(pairJ)
        out.push({ kind: 'invocation', step: toolStep, call: e, result: sorted[pairJ] })
      } else {
        used.add(i)
        out.push({ kind: 'invocation', step: toolStep, call: e, result: null })
      }
    } else {
      toolStep += 1
      used.add(i)
      out.push({ kind: 'result_only', step: toolStep, result: e })
    }
  }
  return out
}

function formatToolSubAgentSuffix(
  call: AgentToolHistoryEntry,
  result: AgentToolHistoryEntry | null,
): string {
  const sid = call.sub_agent_id ?? result?.sub_agent_id
  const nest = call.nesting ?? result?.nesting
  if (nest === 'sub_agent' && sid != null && Number.isFinite(Number(sid))) {
    return ` · 子 #${sid}`
  }
  return ''
}

function ToolHistoryDrawerBody({ entries }: { entries: AgentToolHistoryEntry[] }) {
  const rows = pairToolHistoryForDisplay(entries)
  return (
    <Collapse
      bordered={false}
      size="small"
      className="agent-workspace-msg__process-trace-collapse agent-workspace-msg__process-trace-collapse--drawer"
      items={rows.map((row, idx) => {
        if (row.kind === 'result_only') {
          const e = row.result
          const name = e.name?.trim() || e.tool_call_id || '工具'
          const sub = formatToolSubAgentSuffix(e, null)
          return {
            key: `th-ro-${e.seq}-${idx}`,
            label: `步骤 ${row.step} · ${name}${sub}`,
            children: (
              <div>
                <Text type="secondary" className="agent-workspace-msg__tool-section-label">
                  执行结果
                </Text>
                <pre
                  className="agent-structured-json agent-workspace-msg__tool-history-result"
                  style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                >
                  {e.content ?? '（空）'}
                </pre>
              </div>
            ),
          }
        }
        const { call, result, step } = row
        const name = call.name?.trim() || call.tool_call_id || '工具'
        const sub = formatToolSubAgentSuffix(call, result)
        return {
          key: `th-${call.seq}-${result?.seq ?? 'x'}-${idx}`,
          label: `步骤 ${step} · 调用 ${name}${sub}`,
          children: (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <div>
                <Text type="secondary" className="agent-workspace-msg__tool-section-label">
                  入参
                </Text>
                <pre className="agent-structured-json">
                  {JSON.stringify(call.arguments ?? {}, null, 2)}
                </pre>
              </div>
              {result ? (
                <div>
                  <Text type="secondary" className="agent-workspace-msg__tool-section-label">
                    执行结果
                  </Text>
                  <pre
                    className="agent-structured-json agent-workspace-msg__tool-history-result"
                    style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                  >
                    {result.content ?? '（空）'}
                  </pre>
                </div>
              ) : (
                <Text type="secondary" className="agent-workspace-msg__tool-section-label">
                  （尚无与此调用对应的返回记录）
                </Text>
              )}
            </Space>
          ),
        }
      })}
    />
  )
}

export type AgentWorkspacePageCoreProps = {
  /** ``/agents/:id`` 与 ``/workbench/:namespace`` 两套入口，避免同一路由组件混用。 */
  workspaceEntry: 'agent-by-id' | 'workbench-by-namespace'
}

export function AgentWorkspacePageCore({ workspaceEntry }: AgentWorkspacePageCoreProps) {
  const params = useParams<{ agentId?: string; workspaceNamespace?: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { workbenchAgents, loading: workbenchAgentsLoading, selectWorkbenchAgent } = useWorkbench()

  const agentId = useMemo(() => {
    if (workspaceEntry === 'agent-by-id') {
      const p = params.agentId?.trim()
      return p && /^\d+$/.test(p) ? p : ''
    }
    const nsRaw = params.workspaceNamespace
    if (nsRaw != null && nsRaw !== '') {
      let decoded = nsRaw
      try {
        decoded = decodeURIComponent(nsRaw)
      } catch {
        decoded = nsRaw
      }
      const target = decoded.trim()
      const hit = workbenchAgents.find(
        (a) => (a.workspace_namespace ?? 'default').trim() === target,
      )
      return hit ? String(hit.id) : ''
    }
    return ''
  }, [workspaceEntry, params.agentId, params.workspaceNamespace, workbenchAgents])

  const invalidWorkbenchNamespace = Boolean(
    workspaceEntry === 'workbench-by-namespace' &&
      params.workspaceNamespace != null &&
      params.workspaceNamespace !== '' &&
      !workbenchAgentsLoading &&
      agentId === '',
  )

  const [collapsedConfig, setCollapsedConfig] = useState(false)
  const workbenchListOpenConfigFlag = searchParams.get(WORKBENCH_OPEN_CONFIG_QUERY_PARAM)

  /** 工作区列表齿轮：仅 ``/workbench/:ns`` 解读 ``wb_config=1``，展开侧栏后移除参数（与 ``/agents/:id`` 不复用）。 */
  useEffect(() => {
    if (workspaceEntry !== 'workbench-by-namespace') return
    if (workbenchListOpenConfigFlag !== '1') return
    setCollapsedConfig(false)
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.delete(WORKBENCH_OPEN_CONFIG_QUERY_PARAM)
        return next
      },
      { replace: true },
    )
  }, [workspaceEntry, workbenchListOpenConfigFlag, setSearchParams])

  /** 配置列内容区宽度（不含拖拽条）；最窄 480，最宽为工作区宽度一半减去拖拽条 */
  const AGENT_WORKSPACE_CONFIG_RESIZER_W = 6
  const AGENT_WORKSPACE_CONFIG_SIDEBAR_MIN_PX = 480
  const [configSidebarInnerPx, setConfigSidebarInnerPx] = useState(AGENT_WORKSPACE_CONFIG_SIDEBAR_MIN_PX)
  const agentWorkspaceShellRef = useRef<HTMLDivElement>(null)
  /** 数字 ID 路由下从 GET /api/agents/{id} 拉取元数据时为 true */
  const [agentMetaLoading, setAgentMetaLoading] = useState(false)
  const [agentDisplayName, setAgentDisplayName] = useState('')
  const [agentDescription, setAgentDescription] = useState('')
  const [agentKind, setAgentKind] = useState<AgentKind>('simple_chat')
  /** 与入库 `agent_entity.workspace_namespace` 一致；invoke 时传入 `workspace_namespace` */
  const [workspaceNamespace, setWorkspaceNamespace] = useState('default')

  /** 系统模型：传 config_id=sys_model.id，后端从库加载 endpoint/凭据 */
  const [systemModelId, setSystemModelId] = useState<string | undefined>(undefined)
  const [llmModels, setLlmModels] = useState<LlmModel[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState('')

  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [topP, setTopP] = useState(0.9)
  const [stopWords, setStopWords] = useState('')
  /** 为 null 时不随请求发送（与 OpenAPI hyperparameters 可选字段一致） */
  const [seed, setSeed] = useState<number | null>(null)
  const [frequencyPenalty, setFrequencyPenalty] = useState(0)
  const [presencePenalty, setPresencePenalty] = useState(0)
  const [samplingPopoverOpen, setSamplingPopoverOpen] = useState(false)
  const [responseFormat, setResponseFormat] = useState<ResponseFormat>('text')
  const [toolNames, setToolNames] = useState<string[]>([])
  const [registeredToolOptions, setRegisteredToolOptions] = useState<{ label: string; value: string }[]>(
    [],
  )
  const [toolChoiceMode, setToolChoiceMode] = useState<'auto' | 'none' | 'required' | 'specific'>('auto')
  const [forcedToolName, setForcedToolName] = useState('')
  const [useToolChoice, setUseToolChoice] = useState(false)
  const [stripThinking, setStripThinking] = useState(true)
  const [streamOutput, setStreamOutput] = useState(false)
  const [includeToolMessagesInRaw, setIncludeToolMessagesInRaw] = useState(false)

  const [promptSystem, setPromptSystem] = useState('')
  const [memoryWorkspace, setMemoryWorkspace] =
    useState<MemoryWorkspaceState>(MEMORY_WORKSPACE_DEFAULT)
  const [knowledgeWorkspace, setKnowledgeWorkspace] =
    useState<KnowledgeWorkspaceState>(KNOWLEDGE_WORKSPACE_DEFAULT)
  const [inputFilterWorkspace, setInputFilterWorkspace] = useState<InputFilterWorkspaceState>(
    INPUT_FILTER_WORKSPACE_DEFAULT,
  )
  const [agentKbOptions, setAgentKbOptions] = useState<{ label: string; value: number }[]>([])
  const [agentKbListLoading, setAgentKbListLoading] = useState(false)

  const [messages, setMessages] = useState<ChatLine[]>([])
  /** 与后端对话持久化对齐：首轮由 start/invoke 响应写入，后续请求带回 */
  const [conversationSessionId, setConversationSessionId] = useState<string | null>(null)
  /** 当前会话标题（历史选中或后续可由服务端更新） */
  const [conversationSessionTitle, setConversationSessionTitle] = useState<string | null>(null)
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false)
  /** 执行过程（模型分段 + 工具）/ 知识库来源：可点标签打开右侧抽屉 */
  const [chatAuxDrawer, setChatAuxDrawer] = useState<WorkspaceChatAuxDrawer | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historySessions, setHistorySessions] = useState<ConversationSessionOut[]>([])
  const [historyPage, setHistoryPage] = useState(1)
  const [historyTotal, setHistoryTotal] = useState(0)
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)
  /** 当前会话从服务端分页拉取的消息元数据（用于「加载更早」）；清空对话时置空 */
  const [sessionMessagesMeta, setSessionMessagesMeta] = useState<PageMeta | null>(null)
  const [sessionMessagesPage, setSessionMessagesPage] = useState(1)
  /** 工作台 hub：用于「加载更早」后与整段消息重排 Q-A 顺序 */
  const [workbenchSessionMessagesRaw, setWorkbenchSessionMessagesRaw] = useState<
    ConversationMessageOut[] | null
  >(null)
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [saveSubmitting, setSaveSubmitting] = useState(false)
  const invokeAbortRef = useRef<AbortController | null>(null)
  const chatMessagesScrollRef = useRef<HTMLDivElement>(null)
  /** 为 true 时在消息变化后滚到底部（用户上翻阅读时为 false，避免「一跳一跳」） */
  const chatStickBottomRef = useRef(true)
  const streamFlushRafRef = useRef<number | null>(null)
  const streamPendingStreamRef = useRef<{ content: string; exploring: string } | null>(null)
  const [streamExploringText, setStreamExploringText] = useState('')
  const [streamExploringPhase, setStreamExploringPhase] = useState('')
  const streamExploringScrollRef = useRef<HTMLPreElement>(null)
  /** 工作区编排：进入页面时为当前 `agent_id` 自动拉取「最新一条」历史会话一次（`agentId` 变则重置） */
  const workbenchLatestHistoryBootstrappedRef = useRef<string | null>(null)

  const CHAT_SCROLL_BOTTOM_THRESHOLD_PX = 80

  const handleChatMessagesScroll = useCallback(() => {
    const el = chatMessagesScrollRef.current
    if (!el) return
    const { scrollTop, scrollHeight, clientHeight } = el
    chatStickBottomRef.current =
      scrollHeight - scrollTop - clientHeight <= CHAT_SCROLL_BOTTOM_THRESHOLD_PX
  }, [])

  const clampConfigSidebarInnerPx = useCallback((w: number) => {
    const shell = agentWorkspaceShellRef.current
    const halfInner = shell
      ? Math.floor(shell.getBoundingClientRect().width / 2) - AGENT_WORKSPACE_CONFIG_RESIZER_W
      : w
    /** 主区一半不足 480 时仍保持最窄 480（允许略超「一半」），否则上限为一半减拖拽条 */
    const upper = Math.max(AGENT_WORKSPACE_CONFIG_SIDEBAR_MIN_PX, halfInner)
    return Math.min(Math.max(AGENT_WORKSPACE_CONFIG_SIDEBAR_MIN_PX, w), upper)
  }, [])

  useEffect(() => {
    const shell = agentWorkspaceShellRef.current
    if (!shell || collapsedConfig) return
    const ro = new ResizeObserver(() => {
      setConfigSidebarInnerPx((prev) => clampConfigSidebarInnerPx(prev))
    })
    ro.observe(shell)
    return () => ro.disconnect()
  }, [collapsedConfig, clampConfigSidebarInnerPx])

  const onConfigSidebarResizePointerDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return
      e.preventDefault()
      const shell = agentWorkspaceShellRef.current
      if (!shell) return
      const startX = e.clientX
      const startW = configSidebarInnerPx
      const onMove = (ev: MouseEvent) => {
        const delta = ev.clientX - startX
        setConfigSidebarInnerPx(clampConfigSidebarInnerPx(startW + delta))
      }
      const onUp = () => {
        document.removeEventListener('mousemove', onMove)
        document.removeEventListener('mouseup', onUp)
        document.body.style.removeProperty('cursor')
        document.body.style.removeProperty('user-select')
      }
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
    },
    [configSidebarInnerPx, clampConfigSidebarInnerPx],
  )

  useLayoutEffect(() => {
    const el = chatMessagesScrollRef.current
    if (!el || !chatStickBottomRef.current) return
    el.scrollTop = el.scrollHeight
  }, [messages, loading])

  useLayoutEffect(() => {
    const el = streamExploringScrollRef.current
    if (!el || !loading || !streamOutput) return
    el.scrollTop = el.scrollHeight
  }, [streamExploringText, loading, streamOutput])

  const uniqueLlmModels = useMemo(() => {
    const seen = new Set<string>()
    return llmModels.filter((m) => {
      if (seen.has(m.id)) return false
      seen.add(m.id)
      return true
    })
  }, [llmModels])

  const isNumericAgentRoute = useMemo(() => {
    const raw = agentId?.trim() ?? ''
    return raw.length > 0 && /^\d+$/.test(raw)
  }, [agentId])

  const isWorkbenchWorkspace = useMemo(
    () => isNumericAgentRoute && agentKind === 'workbench',
    [isNumericAgentRoute, agentKind],
  )

  useEffect(() => {
    const show = workspaceEntry === 'workbench-by-namespace' && isWorkbenchWorkspace
    const ns = workspaceNamespace.trim() || 'default'
    const detail: WorkbenchStudioCtaDetail = { show, namespace: ns }
    window.dispatchEvent(new CustomEvent(WORKBENCH_STUDIO_CTA_EVENT, { detail }))
    return () => {
      window.dispatchEvent(
        new CustomEvent(WORKBENCH_STUDIO_CTA_EVENT, {
          detail: { show: false, namespace: '' },
        }),
      )
    }
  }, [workspaceEntry, isWorkbenchWorkspace, workspaceNamespace])

  /** 仅 ``/workbench/:workspaceNamespace`` 入口为 true（与 Agent 中心路由分离） */
  const isWorkbenchHubRoute = useMemo(
    () =>
      workspaceEntry === 'workbench-by-namespace' &&
      params.workspaceNamespace != null &&
      String(params.workspaceNamespace).trim() !== '',
    [workspaceEntry, params.workspaceNamespace],
  )

  const hubNamespaceFromRoute = useMemo(() => {
    if (!isWorkbenchHubRoute) return ''
    const raw = params.workspaceNamespace ?? ''
    try {
      return decodeURIComponent(raw).trim()
    } catch {
      return raw.trim()
    }
  }, [isWorkbenchHubRoute, params.workspaceNamespace])

  /** 当前 URL 命名空间解析到的工作台 orchestrator（列表就绪后即可判定，无需等详情） */
  const hubResolvedWorkbenchAgent = useMemo(() => {
    if (!isWorkbenchHubRoute) return null
    const target = hubNamespaceFromRoute || 'default'
    return workbenchAgents.find((a) => (a.workspace_namespace ?? 'default').trim() === target) ?? null
  }, [hubNamespaceFromRoute, isWorkbenchHubRoute, workbenchAgents])

  const hubRouteTargetsOrchestrator = useMemo(
    () =>
      workspaceEntry === 'workbench-by-namespace' &&
      isWorkbenchHubRoute &&
      isNumericAgentRoute &&
      hubResolvedWorkbenchAgent != null &&
      Number(agentId) === hubResolvedWorkbenchAgent.id,
    [agentId, hubResolvedWorkbenchAgent, isNumericAgentRoute, isWorkbenchHubRoute, workspaceEntry],
  )

  /**
   * 与后端 ``GET .../sessions/:id`` 的 ``messages_agent_id`` 一致：当前会话 session_id 下，
   * 只拉取 ``conversation_message.agent_id`` 等于 URL 当前 Agent 的行（共享会话时与编排器/子 Agent 视图对齐）。
   */
  const conversationDetailMessagesAgentId = useMemo(() => {
    if (!isNumericAgentRoute) return undefined
    const n = Number(agentId?.trim() ?? '')
    return Number.isFinite(n) ? n : undefined
  }, [isNumericAgentRoute, agentId])

  const [workspaceAgentsInNsLoading, setWorkspaceAgentsInNsLoading] = useState(false)
  const [workspaceAgentsInNsError, setWorkspaceAgentsInNsError] = useState<string | null>(null)
  const [workspaceAgentsInNs, setWorkspaceAgentsInNs] = useState<AgentOut[]>([])

  useEffect(() => {
    if (!isWorkbenchHubRoute) return
    const ns = hubNamespaceFromRoute || 'default'
    setWorkspaceNamespace(ns)
  }, [hubNamespaceFromRoute, isWorkbenchHubRoute])

  useEffect(() => {
    if (!hubRouteTargetsOrchestrator) {
      setWorkspaceAgentsInNs([])
      setWorkspaceAgentsInNsError(null)
      setWorkspaceAgentsInNsLoading(false)
      return
    }
    const ns = hubNamespaceFromRoute || 'default'
    let cancelled = false
    setWorkspaceAgentsInNsLoading(true)
    setWorkspaceAgentsInNsError(null)
    void fetchAgentsList({
      workspace_namespace: ns,
      page: 1,
      page_size: 100,
    })
      .then((d) => {
        if (cancelled) return
        setWorkspaceAgentsInNs(d.items ?? [])
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setWorkspaceAgentsInNs([])
          setWorkspaceAgentsInNsError(e instanceof Error ? e.message : '加载失败')
        }
      })
      .finally(() => {
        if (!cancelled) setWorkspaceAgentsInNsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [hubNamespaceFromRoute, hubRouteTargetsOrchestrator])

  const workbenchHubBusinessAgents = useMemo(
    () => workspaceAgentsInNs.filter((a) => a.agent_kind !== 'workbench'),
    [workspaceAgentsInNs],
  )

  const workbenchHubNamespaceRows = useMemo(() => {
    const byNs = new Map<string, { namespace: string; name: string }>()
    for (const a of workbenchAgents) {
      const ns = (a.workspace_namespace ?? 'default').trim() || 'default'
      const name = a.name.trim() || '工作区'
      if (!byNs.has(ns)) {
        byNs.set(ns, { namespace: ns, name })
      }
    }
    return Array.from(byNs.values()).sort((x, y) =>
      x.namespace.localeCompare(y.namespace, undefined, { sensitivity: 'base' }),
    )
  }, [workbenchAgents])

  const workbenchHubNamespaceSelectOptions = useMemo(
    () =>
      workbenchHubNamespaceRows.map((r) => ({
        value: r.namespace,
        label: <span className="agent-workspace-hub-ns-label">{r.namespace}</span>,
        title: r.name !== r.namespace ? `${r.name} · ${r.namespace}` : r.namespace,
      })),
    [workbenchHubNamespaceRows],
  )

  const onWorkbenchHubSwitchNamespace = useCallback(
    (namespace: string) => {
      const ns = (namespace || 'default').trim() || 'default'
      const hit = workbenchAgents.find((a) => (a.workspace_namespace ?? 'default').trim() === ns)
      if (hit) selectWorkbenchAgent(hit.id)
      navigate(workbenchPathForNamespace(ns), { replace: true })
    },
    [navigate, selectWorkbenchAgent, workbenchAgents],
  )

  useEffect(() => {
    if (!isNumericAgentRoute) {
      setAgentKbOptions([])
      return
    }
    let cancelled = false
    setAgentKbListLoading(true)
    void fetchKnowledgeBasesList({ page: 1, page_size: 100 })
      .then((d) => {
        if (cancelled) return
        setAgentKbOptions(
          d.items.map((kb) => ({
            label: `${kb.name}（#${kb.id} · ${knowledgeStatusLabel(kb.status)}）`,
            value: kb.id,
          })),
        )
      })
      .catch(() => {
        if (!cancelled) setAgentKbOptions([])
      })
      .finally(() => {
        if (!cancelled) setAgentKbListLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [isNumericAgentRoute])

  useEffect(() => {
    invokeAbortRef.current?.abort()
    setConversationSessionId(null)
    setConversationSessionTitle(null)
    setSessionMessagesMeta(null)
    setSessionMessagesPage(1)
    setMessages([])
    setWorkbenchSessionMessagesRaw(null)
    setStreamExploringPhase('')
    setStreamExploringText('')
    setLoading(false)
    setChatAuxDrawer(null)
    workbenchLatestHistoryBootstrappedRef.current = null
  }, [agentId])

  useEffect(() => {
    let cancelled = false
    const ns = workspaceNamespace.trim() || 'default'
    void fetchAgentRegisteredTools({ namespace: ns })
      .then((d) => {
        if (cancelled) return
        setRegisteredToolOptions(
          d.items.map((t) => ({
            label: `${t.name}（${
              t.origin === 'builtin'
                ? '内置'
                : t.origin === 'http'
                  ? 'HTTP'
                  : t.origin === 'mcp'
                    ? 'MCP'
                    : '其他'
            }）`,
            value: t.name,
          })),
        )
      })
      .catch(() => {
        if (!cancelled) setRegisteredToolOptions([])
      })
    return () => {
      cancelled = true
    }
  }, [workspaceNamespace])

  const loadHistoryPage = useCallback(
    async (page: number) => {
      if (!isNumericAgentRoute || !agentId?.trim()) return
      setHistoryLoading(true)
      try {
        const data = await listConversationSessions({
          page,
          page_size: AGENT_WORKSPACE_HISTORY_PAGE_SIZE,
          agent_id: Number(agentId.trim()),
        })
        setHistorySessions(data.items)
        setHistoryTotal(data.meta.total)
        setHistoryPage(data.meta.page)
      } catch (e: unknown) {
        void message.error(e instanceof Error ? e.message : '加载历史会话失败')
      } finally {
        setHistoryLoading(false)
      }
    },
    [agentId, isNumericAgentRoute],
  )

  const openHistoryDrawer = useCallback(() => {
    if (!isNumericAgentRoute) {
      void message.warning('请先进入已入库 Agent 后再查看历史对话')
      return
    }
    setHistoryDrawerOpen(true)
    void loadHistoryPage(1)
  }, [isNumericAgentRoute, loadHistoryPage])

  const onSelectHistorySession = useCallback(
    async (session: ConversationSessionOut) => {
      setHistoryDetailLoading(true)
      try {
        const detail = await getConversationSessionDetail(session.session_id, {
          page: 1,
          page_size: AGENT_WORKSPACE_MESSAGE_PAGE_SIZE,
          ...(conversationDetailMessagesAgentId != null
            ? { messages_agent_id: conversationDetailMessagesAgentId }
            : {}),
        })
        setConversationSessionId(detail.session.session_id)
        setConversationSessionTitle(detail.session.title?.trim() || '未命名对话')
        setSessionMessagesMeta(detail.messages_meta)
        setSessionMessagesPage(1)
        if (hubRouteTargetsOrchestrator) {
          setWorkbenchSessionMessagesRaw(detail.messages)
          setMessages(orderWorkbenchChatFromStored(detail.messages))
        } else {
          setWorkbenchSessionMessagesRaw(null)
          setMessages(mapStoredMessagesToChatLines(detail.messages))
        }
        chatStickBottomRef.current = true
        setHistoryDrawerOpen(false)
      } catch (e: unknown) {
        void message.error(e instanceof Error ? e.message : '加载对话详情失败')
      } finally {
        setHistoryDetailLoading(false)
      }
    },
    [hubRouteTargetsOrchestrator, conversationDetailMessagesAgentId],
  )

  const shouldAutoBootstrapWorkbenchLatestHistory = Boolean(
    isNumericAgentRoute &&
      agentId?.trim() &&
      conversationDetailMessagesAgentId != null &&
      (isWorkbenchHubRoute || agentKind === 'workbench'),
  )

  useEffect(() => {
    if (!shouldAutoBootstrapWorkbenchLatestHistory) return
    const aid = agentId!.trim()
    if (workbenchLatestHistoryBootstrappedRef.current === aid) return

    let cancelled = false
    void (async () => {
      try {
        const data = await listConversationSessions({
          page: 1,
          page_size: 1,
          agent_id: Number(aid),
        })
        if (cancelled) return
        if (data.items.length === 0) {
          workbenchLatestHistoryBootstrappedRef.current = aid
          return
        }
        await onSelectHistorySession(data.items[0])
        if (!cancelled) workbenchLatestHistoryBootstrappedRef.current = aid
      } catch (e: unknown) {
        if (!cancelled) {
          workbenchLatestHistoryBootstrappedRef.current = aid
          void message.warning(e instanceof Error ? e.message : '加载最近对话失败')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [shouldAutoBootstrapWorkbenchLatestHistory, agentId, onSelectHistorySession])

  const loadOlderSessionMessages = useCallback(async () => {
    if (!conversationSessionId || !sessionMessagesMeta) return
    const loadedSoFar = sessionMessagesPage * sessionMessagesMeta.page_size
    if (loadedSoFar >= sessionMessagesMeta.total) return
    setHistoryDetailLoading(true)
    try {
      const nextPage = sessionMessagesPage + 1
      const detail = await getConversationSessionDetail(conversationSessionId, {
        page: nextPage,
        page_size: sessionMessagesMeta.page_size,
        ...(conversationDetailMessagesAgentId != null
          ? { messages_agent_id: conversationDetailMessagesAgentId }
          : {}),
      })
      setSessionMessagesMeta(detail.messages_meta)
      setSessionMessagesPage(nextPage)
      if (hubRouteTargetsOrchestrator) {
        const merged = [...detail.messages, ...(workbenchSessionMessagesRaw ?? [])]
        setWorkbenchSessionMessagesRaw(merged)
        setMessages(orderWorkbenchChatFromStored(merged))
      } else {
        setMessages((prev) => [...mapStoredMessagesToChatLines(detail.messages), ...prev])
      }
    } catch (e: unknown) {
      void message.error(e instanceof Error ? e.message : '加载更早消息失败')
    } finally {
      setHistoryDetailLoading(false)
    }
  }, [
    conversationSessionId,
    sessionMessagesMeta,
    sessionMessagesPage,
    hubRouteTargetsOrchestrator,
    workbenchSessionMessagesRaw,
    conversationDetailMessagesAgentId,
  ])

  const defaultPromptDropdownMenu: MenuProps['items'] = useMemo(
    () =>
      WORKSPACE_DEFAULT_PROMPT_MENU.map((m) => ({
        key: m.id,
        label: m.label,
        title: m.description,
      })),
    [],
  )

  const onDefaultPromptMenuClick = useCallback<NonNullable<MenuProps['onClick']>>(
    async (info) => {
      const kind = String(info.key)
      try {
        const d = await fetchDefaultPromptByKind(kind)
        setPromptSystem(d.text)
        void message.success(`已载入：${d.label}`)
      } catch (e: unknown) {
        void message.error(e instanceof Error ? e.message : '加载失败')
      }
    },
    [],
  )

  const applySavedWorkspaceConfig = useCallback((cfg: AgentDetailOut['config_json']) => {
    applyAgentWorkspaceConfigJson(cfg, {
      setTemperature,
      setMaxTokens,
      setTopP,
      setStopWords,
      setSeed,
      setFrequencyPenalty,
      setPresencePenalty,
      setResponseFormat,
      setToolNames,
      setUseToolChoice,
      setToolChoiceMode,
      setForcedToolName,
      setStripThinking,
      setStreamOutput,
      setIncludeToolMessagesInRaw,
      setPromptSystem,
      setMemoryWorkspace,
      setKnowledgeWorkspace,
      setInputFilterWorkspace,
    })
  }, [])

  /** 加载详情：先应用 config_json，再用库表 ``system_prompt`` 列覆盖（非空时优先） */
  const applyAgentDetailToWorkspace = useCallback(
    (detail: AgentDetailOut) => {
      setWorkspaceNamespace(detail.workspace_namespace?.trim() || 'default')
      applySavedWorkspaceConfig(detail.config_json)
      const col = detail.system_prompt?.trim()
      if (col) {
        setPromptSystem(col)
      }
    },
    [applySavedWorkspaceConfig],
  )

  const buildWorkspaceConfigSnapshot = useCallback(
    () =>
      buildAgentWorkspaceConfigJson({
        temperature,
        maxTokens,
        topP,
        stopWords,
        seed,
        frequencyPenalty,
        presencePenalty,
        responseFormat,
        toolNames,
        useToolChoice,
        toolChoiceMode,
        forcedToolName,
        stripThinking,
        streamOutput,
        includeToolMessagesInRaw,
        promptSystem,
        memory: memoryWorkspace,
        knowledge: knowledgeWorkspace,
        inputFilter: inputFilterWorkspace,
      }),
    [
      temperature,
      maxTokens,
      topP,
      stopWords,
      seed,
      frequencyPenalty,
      presencePenalty,
      responseFormat,
      toolNames,
      useToolChoice,
      toolChoiceMode,
      forcedToolName,
      stripThinking,
      streamOutput,
      includeToolMessagesInRaw,
      promptSystem,
      memoryWorkspace,
      knowledgeWorkspace,
      inputFilterWorkspace,
    ],
  )

  const loadLlmModels = useCallback(async () => {
    setModelsLoading(true)
    setModelsError('')
    try {
      const rows = await fetchAllChatLlmModels()
      setLlmModels(rows)
    } catch (e: unknown) {
      setModelsError(e instanceof Error ? e.message : '加载模型列表失败')
    } finally {
      setModelsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadLlmModels()
  }, [loadLlmModels])

  /** 顶栏「刷新」：仅重新拉取 Agent 详情，不重置左侧表单草稿（与切换路由时的 effect 区分） */
  const reloadAgentDetailFromApi = useCallback(async () => {
    const raw = agentId?.trim() ?? ''
    const isNumericId = raw.length > 0 && /^\d+$/.test(raw)
    if (!isNumericId) return
    setAgentMetaLoading(true)
    try {
      const detail = await fetchAgentById(Number(raw))
      setAgentDisplayName(detail.name)
      setAgentDescription(detail.description ?? '')
      setAgentKind(detail.agent_kind)
      setSystemModelId(detail.sys_model_id != null ? String(detail.sys_model_id) : undefined)
      applyAgentDetailToWorkspace(detail)
    } catch {
      void message.warning('加载 Agent 失败，请检查网络或 ID 是否存在')
      setAgentDisplayName(raw)
    } finally {
      setAgentMetaLoading(false)
    }
  }, [agentId, applyAgentDetailToWorkspace])

  const handleWorkspaceAppRefresh = useCallback(() => {
    void loadLlmModels()
    void reloadAgentDetailFromApi()
  }, [loadLlmModels, reloadAgentDetailFromApi])

  useEffect(() => {
    window.addEventListener('app:refresh', handleWorkspaceAppRefresh)
    return () => window.removeEventListener('app:refresh', handleWorkspaceAppRefresh)
  }, [handleWorkspaceAppRefresh])

  useEffect(() => {
    const b = defaultBasicBaseline(agentId)
    const raw = agentId?.trim() ?? ''
    const isNumericId = raw.length > 0 && /^\d+$/.test(raw)

    setTemperature(b.temperature)
    setMaxTokens(b.maxTokens)
    setTopP(b.topP)
    setStopWords(b.stopWords)
    setPromptSystem(b.promptSystem)
    setSamplingPopoverOpen(false)

    if (!isNumericId) {
      setAgentMetaLoading(false)
      setAgentDisplayName(b.agentDisplayName)
      setAgentDescription(b.agentDescription)
      setAgentKind(b.agentKind)
      setSystemModelId(b.systemModelId)
      setWorkspaceNamespace('default')
      setKnowledgeWorkspace(KNOWLEDGE_WORKSPACE_DEFAULT)
      setInputFilterWorkspace(INPUT_FILTER_WORKSPACE_DEFAULT)
      return
    }

    setAgentMetaLoading(true)
    setAgentDisplayName('')
    setAgentDescription('')
    setAgentKind('simple_chat')
    setSystemModelId(undefined)

    let cancelled = false
    void fetchAgentById(Number(raw))
      .then((detail) => {
        if (cancelled) return
        setAgentDisplayName(detail.name)
        setAgentDescription(detail.description ?? '')
        setAgentKind(detail.agent_kind)
        setSystemModelId(detail.sys_model_id != null ? String(detail.sys_model_id) : undefined)
        applyAgentDetailToWorkspace(detail)
      })
      .catch(() => {
        if (cancelled) return
        void message.warning('加载 Agent 失败，请检查网络或 ID 是否存在')
        setAgentDisplayName(raw)
      })
      .finally(() => {
        if (!cancelled) setAgentMetaLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [agentId, applyAgentDetailToWorkspace])

  const stopInvoke = useCallback(() => {
    invokeAbortRef.current?.abort()
  }, [])

  const clearChat = useCallback(() => {
    setMessages([])
    setConversationSessionId(null)
    setConversationSessionTitle(null)
    setSessionMessagesMeta(null)
    setSessionMessagesPage(1)
    setWorkbenchSessionMessagesRaw(null)
    setStreamExploringPhase('')
    setStreamExploringText('')
  }, [])

  const isChatEmpty = messages.length === 0 && !loading

  const canLoadOlderSessionMessages =
    sessionMessagesMeta !== null &&
    sessionMessagesPage * sessionMessagesMeta.page_size < sessionMessagesMeta.total

  const saveBasicConfigDraft = useCallback(async () => {
    if (!isNumericAgentRoute || !agentId?.trim()) {
      void message.warning('请先通过 Agent 中心创建并进入已入库 Agent')
      return
    }
    if (!systemModelId) {
      void message.warning('请选择系统模型')
      return
    }
    if (agentKind !== 'workbench' && !promptSystem.trim()) {
      void message.warning('请填写系统提示词')
      return
    }
    setSaveSubmitting(true)
    try {
      await updateAgent(Number(agentId), {
        sys_model_id: Number(systemModelId),
        system_prompt: promptSystem.trim() || null,
        config_json: buildWorkspaceConfigSnapshot(),
      })
      void message.success('配置已保存')
    } catch (e: unknown) {
      void message.error(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaveSubmitting(false)
    }
  }, [
    agentId,
    agentKind,
    isNumericAgentRoute,
    systemModelId,
    promptSystem,
    buildWorkspaceConfigSnapshot,
  ])

  const sendMessage = async () => {
    const text = draft.trim()
    if (!text || loading) return
    if (!agentDisplayName.trim()) {
      void message.warning('请填写名称')
      return
    }
    if (!systemModelId) {
      void message.warning('请先选择系统模型')
      return
    }
    if (agentKind !== 'workbench' && !promptSystem.trim()) {
      void message.warning('请填写系统提示词')
      return
    }
    invokeAbortRef.current?.abort()
    const ac = new AbortController()
    invokeAbortRef.current = ac

    chatStickBottomRef.current = true

    const userAt = chatLineTimestamp()
    const assistantAt = chatLineTimestamp()
    setStreamExploringPhase('')
    setStreamExploringText('')
    setDraft('')
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text, at: userAt },
      ...(streamOutput ? [{ role: 'assistant' as const, content: '', at: assistantAt }] : []),
    ])
    setLoading(true)
    try {
      const promptsPayload = buildPromptsPayload(promptSystem)
      const stopList = parseStopWordsForApi(stopWords)

      const responseBlock: { response_format: ResponseFormat } = { response_format: responseFormat }

      const invokeBody = {
        agent_kind: agentKind,
        user_message: text,
        // 系统模型以 config_id 为准，勿传占位的 model_identity（见 buildAgentInvokeBodyFromDetail 注释）
        hyperparameters: {
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          ...(stopList ? { stop: stopList } : {}),
          ...(seed != null ? { seed } : {}),
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
        },
        response: responseBlock,
        tool_choice: useToolChoice
          ? {
              mode: toolChoiceMode,
              forced_tool_name: toolChoiceMode === 'specific' ? forcedToolName.trim() || null : null,
            }
          : null,
        output: {
          strip_thinking_blocks: stripThinking,
          include_tool_messages_in_raw: includeToolMessagesInRaw,
        },
        config_id: systemModelId,
        tool_names: toolNames.length ? toolNames : undefined,
        ...(promptsPayload ? { prompts: promptsPayload } : {}),
        ...(isNumericAgentRoute && agentId?.trim()
          ? { agent_id: Number(agentId.trim()) }
          : {}),
        ...(conversationSessionId ? { conversation_session_id: conversationSessionId } : {}),
        workspace_namespace: workspaceNamespace.trim() || 'default',
      }

      const primaryShellAgentId =
        isNumericAgentRoute && agentId?.trim() ? Number(agentId.trim()) : null

      const data = streamOutput
        ? await invokeAgentStream(invokeBody, {
            signal: ac.signal,
            onStart: (ev) => {
              if (ev.conversation_session_id) {
                setConversationSessionId(ev.conversation_session_id)
              }
            },
            onProgress: (ev) => {
              const label = formatAgentStreamProgressLabel(ev).trim()
              if (label && streamProgressPhaseBelongsToShellAgent(ev, primaryShellAgentId)) {
                setStreamExploringPhase(label)
              }
            },
            onDelta: (contentAcc, exploringAcc) => {
              streamPendingStreamRef.current = { content: contentAcc, exploring: exploringAcc }
              if (streamFlushRafRef.current != null) return
              streamFlushRafRef.current = requestAnimationFrame(() => {
                streamFlushRafRef.current = null
                const pair = streamPendingStreamRef.current
                if (pair == null) return
                setStreamExploringText(pair.exploring)
                setMessages((prev) =>
                  prev.map((m) =>
                    m.role === 'assistant' && m.at === assistantAt
                      ? { ...m, content: pair.content }
                      : m,
                  ),
                )
              })
            },
          })
        : await invokeAgent(invokeBody, { signal: ac.signal })

      if (data.conversation_session_id) {
        setConversationSessionId(data.conversation_session_id)
      }

      if (data.warnings?.length) {
        notification.warning({
          message: '调用提示',
          description: data.warnings.join('\n'),
          duration: 8,
        })
      }

      const reply = data.assistant_text?.trim()
      const think = data.thinking_text?.trim()
      if (!reply && !data.structured && !think) {
        throw new Error('响应缺少 data.assistant_text / structured / thinking_text')
      }
      const processTrace =
        data.process_trace && data.process_trace.length >= 1 ? data.process_trace : undefined
      const toolHistory =
        data.tool_history && data.tool_history.length > 0 ? data.tool_history : undefined
      setMessages((prev) => {
        const kc = data.knowledge_citations?.length ? data.knowledge_citations : undefined
        if (streamOutput) {
          return prev.map((m) =>
            m.role === 'assistant' && m.at === assistantAt
              ? {
                  ...m,
                  content: reply ?? '',
                  structured: data.structured ?? null,
                  ...(think ? { thinkingText: think } : {}),
                  ...(processTrace ? { processTrace } : {}),
                  ...(kc ? { knowledgeCitations: kc } : {}),
                  ...(toolHistory ? { toolHistory } : {}),
                }
              : m,
          )
        }
        return [
          ...prev,
          {
            role: 'assistant',
            at: chatLineTimestamp(),
            content: reply ?? '',
            structured: data.structured ?? null,
            ...(think ? { thinkingText: think } : {}),
            ...(processTrace ? { processTrace } : {}),
            ...(kc ? { knowledgeCitations: kc } : {}),
            ...(toolHistory ? { toolHistory } : {}),
          },
        ]
      })
    } catch (e: unknown) {
      const aborted =
        ac.signal.aborted ||
        (e instanceof DOMException && e.name === 'AbortError') ||
        (e instanceof Error && e.name === 'AbortError')
      if (aborted) {
        setMessages((prev) => {
          if (streamOutput) {
            return prev.map((m) =>
              m.role === 'assistant' && m.at === assistantAt && !m.content.trim()
                ? { ...m, content: '（已停止）' }
                : m,
            )
          }
          return [...prev, { role: 'assistant', at: chatLineTimestamp(), content: '（已停止）' }]
        })
        return
      }
      const errText = e instanceof Error ? e.message : '请求失败'
      notification.error({
        key: 'agent-workspace-invoke-error',
        message: '对话请求失败',
        description: errText,
        placement: 'top',
        duration: 10,
        style: { maxWidth: 560 },
      })
    } finally {
      if (streamFlushRafRef.current != null) {
        cancelAnimationFrame(streamFlushRafRef.current)
        streamFlushRafRef.current = null
      }
      streamPendingStreamRef.current = null
      setStreamExploringText('')
      setStreamExploringPhase('')
      if (invokeAbortRef.current === ac) {
        invokeAbortRef.current = null
      }
      setLoading(false)
    }
  }

  if (invalidWorkbenchNamespace) {
    return (
      <div className="page" style={{ padding: 24, textAlign: 'center' }}>
        <Empty description="未找到该命名空间下的工作区" />
        <Button type="link" onClick={() => navigate('/workbench')}>
          返回工作区
        </Button>
      </div>
    )
  }

  return (
    <div className="agent-list-page agent-workspace-page">
      <div className="agent-list-table-shell">
        <div
          ref={agentWorkspaceShellRef}
          className={`agent-workspace${
            !collapsedConfig ? ' agent-workspace--with-config' : ' agent-workspace--with-expand-rail'
          }`}
        >
          {!collapsedConfig ? (
            <>
              <div
                className="agent-workspace-config-aside"
                style={{
                  width: configSidebarInnerPx + AGENT_WORKSPACE_CONFIG_RESIZER_W,
                }}
              >
                <div className="agent-workspace-config-column">
              <div className="agent-workspace-config-sticky-head">
                <div className="agent-workspace-config-save-bar">
                  <Row gutter={8} align="middle" wrap={false} className="agent-workspace-config-save-bar__row">
                    <Col flex="none">
                      <Tooltip title="收起配置">
                        <Button
                          type="text"
                          className="agent-workspace-config-collapse-only-icon"
                          icon={<MenuFoldOutlined />}
                          aria-label="收起配置"
                          onClick={() => setCollapsedConfig(true)}
                        />
                      </Tooltip>
                    </Col>
                    <Col flex="auto" style={{ minWidth: 0 }}>
                      {isNumericAgentRoute ? (
                        <div className="agent-workspace-config-head-meta">
                          <div className="agent-workspace-config-head-title-group">
                            <div className="agent-workspace-config-head-name-wrap">
                              <Tooltip
                                title={
                                  agentMetaLoading
                                    ? '加载中…'
                                    : agentDescription.trim()
                                      ? agentDescription
                                      : '暂无描述'
                                }
                                placement="bottomLeft"
                              >
                                <span className="agent-workspace-config-head-name">
                                  {agentMetaLoading ? '…' : agentDisplayName.trim() || '—'}
                                </span>
                              </Tooltip>
                            </div>
                            <Tag bordered={false} color="blue" className="agent-workspace-config-head-kind-tag">
                              {agentMetaLoading ? '…' : workspaceAgentKindLabel(agentKind)}
                            </Tag>
                          </div>
                        </div>
                      ) : (
                        <Text type="secondary" className="agent-workspace-config-head-draft" ellipsis>
                          未入库会话
                        </Text>
                      )}
                    </Col>
                    <Col flex="none">
                      <Space size={8} wrap align="center">
                        {hubRouteTargetsOrchestrator ? (
                          <Tooltip title="切换命名空间；悬停选项可见工作区编排名称。">
                            <Select
                              size="small"
                              className="agent-workspace-hub-ns-switch__select"
                              popupClassName="agent-workspace-hub-ns-dropdown"
                              popupMatchSelectWidth={false}
                              styles={{ popup: { root: { minWidth: 140 } } }}
                              style={{ minWidth: 112, maxWidth: 220 }}
                              value={hubNamespaceFromRoute || 'default'}
                              loading={workbenchAgentsLoading}
                              options={
                                workbenchHubNamespaceSelectOptions.length > 0
                                  ? workbenchHubNamespaceSelectOptions
                                  : [
                                      {
                                        value: hubNamespaceFromRoute || 'default',
                                        label: (
                                          <span className="agent-workspace-hub-ns-label">
                                            {hubNamespaceFromRoute || 'default'}
                                          </span>
                                        ),
                                        title:
                                          hubResolvedWorkbenchAgent &&
                                          hubResolvedWorkbenchAgent.name.trim()
                                            ? `${hubResolvedWorkbenchAgent.name.trim()} · ${hubNamespaceFromRoute || 'default'}`
                                            : (hubNamespaceFromRoute || 'default'),
                                      },
                                    ]
                              }
                              onChange={onWorkbenchHubSwitchNamespace}
                              showSearch
                              filterOption={(input, option) => {
                                const q = input.trim().toLowerCase()
                                if (!q) return true
                                const v = option?.value
                                const row = workbenchHubNamespaceRows.find((r) => r.namespace === v)
                                if (!row) return String(v ?? '').toLowerCase().includes(q)
                                return (
                                  row.name.toLowerCase().includes(q) ||
                                  row.namespace.toLowerCase().includes(q)
                                )
                              }}
                              aria-label="切换命名空间"
                            />
                          </Tooltip>
                        ) : (
                          <Tooltip title="将系统模型与左侧表单配置保存到服务端。">
                            <Button
                              type="primary"
                              size="small"
                              loading={saveSubmitting}
                              disabled={saveSubmitting}
                              onClick={() => void saveBasicConfigDraft()}
                            >
                              保存配置
                            </Button>
                          </Tooltip>
                        )}
                      </Space>
                    </Col>
                  </Row>
                </div>
              </div>
              <div className="agent-workspace-config-scroll">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {hubRouteTargetsOrchestrator ? (
                    <Spin spinning={workspaceAgentsInNsLoading}>
                      {workspaceAgentsInNsError ? (
                        <Alert type="error" message={workspaceAgentsInNsError} showIcon />
                      ) : (
                        <>
                          <Card
                            size="small"
                            className="agent-workbench-hub-meta-card"
                            title={
                              <span className="agent-workbench-hub-meta-card__title">
                                工作区基本信息
                              </span>
                            }
                          >
                            <div className="agent-workbench-hub-meta">
                              <section className="agent-workbench-hub-meta__block">
                                <span className="agent-workbench-hub-meta__label">名称</span>
                                {agentMetaLoading ? (
                                  <Text type="secondary">…</Text>
                                ) : (
                                  <div className="agent-workbench-hub-meta__title">
                                    {agentDisplayName.trim() || '—'}
                                  </div>
                                )}
                              </section>
                              <div
                                className="agent-workbench-hub-meta__grid"
                                aria-label="工作区标识"
                              >
                                <div className="agent-workbench-hub-meta__tile">
                                  <span className="agent-workbench-hub-meta__label">Agent ID</span>
                                  <Text code className="agent-workbench-hub-meta__code">
                                    {agentId?.trim() || '—'}
                                  </Text>
                                </div>
                                <div className="agent-workbench-hub-meta__tile">
                                  <span className="agent-workbench-hub-meta__label">命名空间</span>
                                  <Text code className="agent-workbench-hub-meta__code">
                                    {hubNamespaceFromRoute || 'default'}
                                  </Text>
                                </div>
                              </div>
                              <section className="agent-workbench-hub-meta__block agent-workbench-hub-meta__block--desc">
                                <span className="agent-workbench-hub-meta__label">描述</span>
                                {agentMetaLoading ? (
                                  <Text type="secondary">…</Text>
                                ) : agentDescription.trim() ? (
                                  <Paragraph
                                    ellipsis={{ rows: 3 }}
                                    className="agent-workbench-hub-meta__desc-text"
                                  >
                                    {agentDescription.trim()}
                                  </Paragraph>
                                ) : (
                                  <Text type="secondary" className="agent-workbench-hub-meta__desc-empty">
                                    暂无
                                  </Text>
                                )}
                              </section>
                            </div>
                          </Card>
                          <div className="agent-workbench-hub-agents">
                            <div className="agent-workbench-hub-agents__header">
                              <span className="agent-workbench-hub-agents__title">业务 Agent</span>
                              <span className="agent-workbench-hub-agents__count">
                                {workbenchHubBusinessAgents.length}
                              </span>
                            </div>
                            {workbenchHubBusinessAgents.length === 0 ? (
                              <div className="agent-workbench-hub-agents__empty">
                                <Empty
                                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                                  description="暂无业务 Agent"
                                />
                              </div>
                            ) : (
                              <div className="agent-workbench-hub-agents__list" role="list">
                                {workbenchHubBusinessAgents.map((item) => (
                                  <Link
                                    key={item.id}
                                    to={`/agents/${item.id}`}
                                    className="agent-workbench-hub-agents__row"
                                    role="listitem"
                                  >
                                    <Tag
                                      color="blue"
                                      className="agent-workbench-hub-agents__kind"
                                    >
                                      {workspaceAgentKindLabel(item.agent_kind)}
                                    </Tag>
                                    <span className="agent-workbench-hub-agents__name">
                                      {item.name.trim() || `Agent #${item.id}`}
                                    </span>
                                    <span className="agent-workbench-hub-agents__id">#{item.id}</span>
                                    <ArrowRightOutlined
                                      className="agent-workbench-hub-agents__arrow"
                                      aria-hidden
                                    />
                                  </Link>
                                ))}
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </Spin>
                  ) : null}
                  {!hubRouteTargetsOrchestrator ? (
                  <Collapse
                  bordered={false}
                  ghost
                  className="agent-workspace-fold-cards"
                  defaultActiveKey={['basic']}
                  expandIconPosition="end"
                  expandIcon={({ isActive }) => (
                    <span
                      className="agent-workspace-fold-cards__icon"
                      style={{
                        transform: isActive ? 'rotate(0deg)' : 'rotate(-90deg)',
                      }}
                    >
                      <DownOutlined />
                    </span>
                  )}
                  items={[
                    {
                      key: 'basic',
                      label: '基础配置',
                      children: (
                        <Form
                          layout="vertical"
                          requiredMark={false}
                          colon={false}
                          size="middle"
                          className="agent-workspace-basic-form agent-workspace-basic-form--compact"
                        >
                          {!isNumericAgentRoute ? (
                            <>
                              <Alert
                                type="info"
                                showIcon
                                message="未入库会话"
                                description="名称、编排类型与描述请在 Agent 中心通过侧栏「新建 Agent」创建后再进入工作区。"
                                className="agent-workspace-agent-meta-alert"
                              />
                              <Divider className="agent-workspace-weak-divider" />
                            </>
                          ) : null}

                          {isNumericAgentRoute ? (
                            <Form.Item
                              label={
                                <AgentWorkspaceFieldLabel
                                  title="workspace_namespace"
                                  tip="与入库 Agent 一致；多 Agent / 知识库资源按此隔离。调用 invoke 时随请求传给服务端。"
                                />
                              }
                            >
                              <Input readOnly value={workspaceNamespace} variant="filled" />
                            </Form.Item>
                          ) : null}

                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                required
                                title="模型与参数"
                                tip="选择系统已挂载的对话模型；同一模型 ID 仅保留一条配置。"
                              />
                            }
                          >
                            <div>
                            <Row gutter={8} align="middle" wrap={false}>
                              <Col flex="auto" style={{ minWidth: 0 }}>
                                <Select
                                  showSearch
                                  size="middle"
                                  loading={modelsLoading || agentMetaLoading}
                                  placeholder="选择已挂载的对话模型"
                                  style={{ width: '100%' }}
                                  value={systemModelId}
                                  disabled={agentMetaLoading}
                                  onChange={(v) => {
                                    setSystemModelId(v)
                                  }}
                                  optionFilterProp="label"
                                  options={uniqueLlmModels.map((m) => ({
                                    value: m.id,
                                    label: `${m.model_code} · ${m.provider_name}`,
                                  }))}
                                />
                              </Col>
                              <Col flex="none">
                                <Popover
                                  trigger="click"
                                  open={samplingPopoverOpen}
                                  onOpenChange={setSamplingPopoverOpen}
                                  placement="bottomRight"
                                  overlayClassName="agent-workspace-sampling-popover"
                                  title={
                                    <div className="agent-workspace-sampling-popover-head">
                                      <Text strong>采样参数</Text>
                                      <Button
                                        type="link"
                                        size="small"
                                        icon={<ReloadOutlined />}
                                        onClick={(e) => {
                                          e.preventDefault()
                                          e.stopPropagation()
                                          setTemperature(0.7)
                                          setMaxTokens(4096)
                                          setTopP(0.9)
                                          setStopWords('')
                                          setSeed(null)
                                          setFrequencyPenalty(0)
                                          setPresencePenalty(0)
                                        }}
                                      >
                                        恢复默认
                                      </Button>
                                    </div>
                                  }
                                  content={
                                    <Form
                                      layout="vertical"
                                      requiredMark={false}
                                      colon={false}
                                      size="small"
                                      className="agent-workspace-sampling-popover-form"
                                      component="div"
                                    >
                                      <Form.Item
                                        label={
                                          <SamplingFieldLabelWithValue
                                            title="🔹 温度"
                                            hint={
                                              <>
                                                <div>取值范围：0～2</div>
                                                <div>默认：0.7</div>
                                              </>
                                            }
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            value={temperature}
                                            onChange={setTemperature}
                                            fallbackOnInvalid={0.7}
                                          />
                                        }
                                        className="agent-workspace-sampling-form-item-slider agent-workspace-sampling-form-item--label-value"
                                      >
                                        <div className="agent-workspace-sampling-slider-only">
                                          <Slider
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            value={temperature}
                                            onChange={setTemperature}
                                            tooltip={{ open: false }}
                                          />
                                        </div>
                                      </Form.Item>
                                      <Form.Item
                                        label={
                                          <SamplingFieldLabelWithValue
                                            title="🔹 最大令牌数"
                                            hint={
                                              <>
                                                <div>取值范围：256～128000</div>
                                                <div>默认：4096</div>
                                                <div style={{ marginTop: 6 }}>
                                                  限制本次模型回复生成的 token 数；用量统计里的 total 仍含系统提示与上下文。
                                                </div>
                                              </>
                                            }
                                            min={256}
                                            max={128000}
                                            step={256}
                                            value={maxTokens}
                                            onChange={setMaxTokens}
                                            fallbackOnInvalid={4096}
                                            valueWidth={70}
                                          />
                                        }
                                        className="agent-workspace-sampling-form-item-slider agent-workspace-sampling-form-item--label-value"
                                      >
                                        <div className="agent-workspace-sampling-slider-only">
                                          <Slider
                                            min={256}
                                            max={128000}
                                            step={256}
                                            value={maxTokens}
                                            onChange={setMaxTokens}
                                            tooltip={{ open: false }}
                                          />
                                        </div>
                                      </Form.Item>
                                      <Form.Item
                                        label={
                                          <SamplingFieldLabelWithValue
                                            title="🔹 Top P"
                                            hint={
                                              <>
                                                <div>取值范围：0～1</div>
                                                <div>默认：0.9</div>
                                              </>
                                            }
                                            min={0}
                                            max={1}
                                            step={0.01}
                                            value={topP}
                                            onChange={setTopP}
                                            fallbackOnInvalid={0.9}
                                          />
                                        }
                                        className="agent-workspace-sampling-form-item-slider agent-workspace-sampling-form-item--label-value"
                                      >
                                        <div className="agent-workspace-sampling-slider-only">
                                          <Slider
                                            min={0}
                                            max={1}
                                            step={0.01}
                                            value={topP}
                                            onChange={setTopP}
                                            tooltip={{ open: false }}
                                          />
                                        </div>
                                      </Form.Item>
                                      <Collapse
                                        bordered={false}
                                        size="small"
                                        defaultActiveKey={[]}
                                        className="agent-workspace-sampling-more-collapse"
                                        style={{ width: '100%' }}
                                        items={[
                                          {
                                            key: 'more-hparams',
                                            label: (
                                              <Text type="secondary" style={{ fontSize: 13 }}>
                                                更多：Seed、惩罚项、停止词
                                              </Text>
                                            ),
                                            children: (
                                              <div style={{ paddingTop: 4 }}>
                                                <Form.Item
                                                  label={
                                                    <SamplingLabelWithHint
                                                      title="🔹 Seed"
                                                      hint={
                                                        <>
                                                          <div>取值范围：非负整数；留空则不传该参数</div>
                                                          <div>默认：不传（留空）</div>
                                                          <div style={{ marginTop: 6 }}>
                                                            部分模型支持确定性采样。保存配置会写入 Agent。
                                                          </div>
                                                        </>
                                                      }
                                                    />
                                                  }
                                                  className="agent-workspace-sampling-form-item--label-value"
                                                >
                                                  <InputNumber
                                                    size="small"
                                                    min={0}
                                                    step={1}
                                                    placeholder="留空不传"
                                                    style={{ width: '100%' }}
                                                    value={seed ?? undefined}
                                                    onChange={(v) =>
                                                      setSeed(v == null || Number.isNaN(v) ? null : Math.round(v))
                                                    }
                                                  />
                                                </Form.Item>
                                                <Form.Item
                                                  label={
                                                    <SamplingFieldLabelWithValue
                                                      title="🔹 Frequency penalty"
                                                      hint={
                                                        <>
                                                          <div>取值范围：-2～2</div>
                                                          <div>默认：0</div>
                                                          <div style={{ marginTop: 6 }}>与温度相同可拖动调节。</div>
                                                        </>
                                                      }
                                                      min={-2}
                                                      max={2}
                                                      step={0.1}
                                                      value={frequencyPenalty}
                                                      onChange={setFrequencyPenalty}
                                                      fallbackOnInvalid={0}
                                                    />
                                                  }
                                                  className="agent-workspace-sampling-form-item-slider agent-workspace-sampling-form-item--label-value"
                                                >
                                                  <div className="agent-workspace-sampling-slider-only">
                                                    <Slider
                                                      min={-2}
                                                      max={2}
                                                      step={0.1}
                                                      value={frequencyPenalty}
                                                      onChange={setFrequencyPenalty}
                                                      tooltip={{ open: false }}
                                                    />
                                                  </div>
                                                </Form.Item>
                                                <Form.Item
                                                  label={
                                                    <SamplingFieldLabelWithValue
                                                      title="🔹 Presence penalty"
                                                      hint={
                                                        <>
                                                          <div>取值范围：-2～2</div>
                                                          <div>默认：0</div>
                                                          <div style={{ marginTop: 6 }}>与温度相同可拖动调节。</div>
                                                        </>
                                                      }
                                                      min={-2}
                                                      max={2}
                                                      step={0.1}
                                                      value={presencePenalty}
                                                      onChange={setPresencePenalty}
                                                      fallbackOnInvalid={0}
                                                    />
                                                  }
                                                  className="agent-workspace-sampling-form-item-slider agent-workspace-sampling-form-item--label-value"
                                                >
                                                  <div className="agent-workspace-sampling-slider-only">
                                                    <Slider
                                                      min={-2}
                                                      max={2}
                                                      step={0.1}
                                                      value={presencePenalty}
                                                      onChange={setPresencePenalty}
                                                      tooltip={{ open: false }}
                                                    />
                                                  </div>
                                                </Form.Item>
                                                <Form.Item
                                                  label={
                                                    <SamplingLabelWithHint
                                                      title="🔹 停止词"
                                                      hint={
                                                        <>
                                                          <div>可选；每行一条。</div>
                                                          <div>默认：空（不传）</div>
                                                        </>
                                                      }
                                                    />
                                                  }
                                                  className="agent-workspace-sampling-form-item--label-value"
                                                >
                                                  <Input.TextArea
                                                    rows={3}
                                                    placeholder="每行一条，可选"
                                                    value={stopWords}
                                                    onChange={(e) => setStopWords(e.target.value)}
                                                    className="agent-workspace-sampling-stopwords"
                                                  />
                                                </Form.Item>
                                              </div>
                                            ),
                                          },
                                        ]}
                                      />
                                    </Form>
                                  }
                                >
                                  <Tooltip title="采样参数（悬浮层）">
                                    <Button
                                      type="text"
                                      icon={<SlidersOutlined style={{ fontSize: 16 }} />}
                                      aria-label="打开采样参数"
                                      className={
                                        samplingPopoverOpen
                                          ? 'agent-workspace-model-tune-btn agent-workspace-model-tune-btn--active'
                                          : 'agent-workspace-model-tune-btn'
                                      }
                                      style={{
                                        height: 32,
                                        width: 32,
                                        padding: 0,
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                      }}
                                    />
                                  </Tooltip>
                                </Popover>
                              </Col>
                            </Row>
                            {modelsError ? (
                              <Text type="danger" style={{ display: 'block', marginTop: 4 }}>
                                {modelsError}
                              </Text>
                            ) : null}
                            </div>
                          </Form.Item>

                          <Divider className="agent-workspace-weak-divider" />

                          <Form.Item
                            className="agent-workspace-system-prompt-item"
                            label={
                              <div className="agent-workspace-system-prompt-label-row">
                                <div className="agent-workspace-system-prompt-label-row__title">
                                  <AgentWorkspaceFieldLabel
                                    required={agentKind !== 'workbench'}
                                    title={
                                      agentKind === 'workbench' ? '流程提示词' : '系统提示词'
                                    }
                                    tip={
                                      agentKind === 'workbench'
                                        ? '可选。补充本工作台的委派策略、汇总风格、领域术语等；基础编排职责与 workbench_* 边界由服务端固定注入，无需在此重复。'
                                        : '第 1 类 System Prompt（prompts.system_prompt）；与每次用户消息一并发送。建议写清角色、能力、约束、输出格式与安全边界。'
                                    }
                                  />
                                </div>
                                {agentKind === 'workbench' ? (
                                  <Button
                                    type="link"
                                    size="small"
                                    className="agent-workspace-default-prompt-btn"
                                    onClick={() => {
                                      setPromptSystem(WORKBENCH_FLOW_PROMPT_TEMPLATE)
                                      void message.success('已填入流程提示词模板')
                                    }}
                                  >
                                    流程模板
                                  </Button>
                                ) : (
                                  <Dropdown
                                    trigger={['click']}
                                    menu={{
                                      items: defaultPromptDropdownMenu,
                                      onClick: onDefaultPromptMenuClick,
                                    }}
                                  >
                                    <Button
                                      type="link"
                                      size="small"
                                      className="agent-workspace-default-prompt-btn"
                                      icon={<DownOutlined style={{ fontSize: 11 }} />}
                                      iconPosition="end"
                                    >
                                      默认提示词
                                    </Button>
                                  </Dropdown>
                                )}
                              </div>
                            }
                          >
                            <Input.TextArea
                              placeholder={
                                agentKind === 'workbench'
                                  ? '可选：流程与风格补充（留空则仅使用平台基础提示词）'
                                  : '角色、能力与安全边界'
                              }
                              value={promptSystem}
                              onChange={(e) => setPromptSystem(e.target.value)}
                              className="agent-prompt-field__input agent-workspace-system-prompt-input"
                            />
                          </Form.Item>
                        </Form>
                      ),
                    },
                    {
                      key: 'tools',
                      label: '工具',
                      children: (
                        <Space direction="vertical" size={10} style={{ width: '100%' }}>
                          <div>
                            <Tooltip title="对应请求体 tool_names 与 config_json.tool_names；仅绑定已在服务端注册的工具。">
                              <Text type="secondary">工具名称</Text>
                            </Tooltip>
                            <Select
                              mode="tags"
                              style={{ width: '100%', marginTop: 4 }}
                              placeholder="可从下拉选已注册名，或输入后回车添加"
                              options={registeredToolOptions}
                              value={toolNames}
                              onChange={setToolNames}
                              tokenSeparators={[',']}
                            />
                          </div>
                          <Divider className="agent-workspace-weak-divider" />
                          <Space direction="vertical" style={{ width: '100%' }} size={8}>
                            <div className="agent-workspace-switch-row">
                              <Text type="secondary">启用自定义策略</Text>
                              <Switch checked={useToolChoice} onChange={setUseToolChoice} />
                            </div>
                            {useToolChoice ? (
                              <>
                                <Select
                                  style={{ width: '100%' }}
                                  value={toolChoiceMode}
                                  onChange={setToolChoiceMode}
                                  options={[
                                    { label: 'auto', value: 'auto' },
                                    { label: 'none', value: 'none' },
                                    { label: 'required', value: 'required' },
                                    { label: 'specific', value: 'specific' },
                                  ]}
                                />
                                {toolChoiceMode === 'specific' ? (
                                  <Input
                                    placeholder="指定要强制的工具名"
                                    value={forcedToolName}
                                    onChange={(e) => setForcedToolName(e.target.value)}
                                  />
                                ) : null}
                              </>
                            ) : null}
                          </Space>
                        </Space>
                      ),
                    },
                    {
                      key: 'memory',
                      label: '记忆配置',
                      children: (
                        <Space
                          className="agent-workspace-memory-stack"
                          direction="vertical"
                          size={12}
                          style={{ width: '100%' }}
                        >
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="注入历史轮数上限"
                                tip="与会话消息组装为 HistoryTurnSlot 时，最多保留最近若干轮（user+assistant 为一轮）；范围 1～200。"
                              />
                            }
                          >
                            <InputNumber
                              min={1}
                              max={200}
                              value={memoryWorkspace.maxHistoryRoundsCap}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  maxHistoryRoundsCap:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(1, Math.min(200, Math.round(v)))
                                      : prev.maxHistoryRoundsCap,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                          <Text strong className="agent-workspace-memory-section-title">
                            分层滚动摘要
                          </Text>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="尾窗原文轮数"
                                tip="会话已有非空滚动摘要时，仅将最近若干轮 HistoryTurnSlot 注入模型，与摘要块拼接；与后端 min_tail_raw_rounds 一致（默认 6）。"
                              />
                            }
                          >
                            <InputNumber
                              min={1}
                              max={64}
                              value={memoryWorkspace.minTailRawRounds}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  minTailRawRounds:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(1, Math.min(64, Math.round(v)))
                                      : prev.minTailRawRounds,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="每批压缩轮数"
                                tip="链式摘要每次将「上一摘要 + 随后若干 user 轮」合并；与后端 compress_batch_rounds 一致（默认 3）。"
                              />
                            }
                          >
                            <InputNumber
                              min={1}
                              max={64}
                              value={memoryWorkspace.compressBatchRounds}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  compressBatchRounds:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(1, Math.min(64, Math.round(v)))
                                      : prev.compressBatchRounds,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="摘要专用模型（可选）"
                                tip="滚动摘要 LLM 的 sys_model.id；未选时由服务端默认或与主模型一致（待接入）。"
                              />
                            }
                          >
                            <Select
                              showSearch
                              allowClear
                              loading={modelsLoading}
                              placeholder="不指定则走服务端默认"
                              style={{ width: '100%' }}
                              value={memoryWorkspace.summarizationSysModelId}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summarizationSysModelId:
                                    v === undefined || v === null ? undefined : Number(v),
                                }))
                              }
                              optionFilterProp="label"
                              options={uniqueLlmModels.map((m) => ({
                                value: m.id,
                                label: `${m.model_code} · ${m.provider_name}`,
                              }))}
                            />
                          </Form.Item>
                          <Text strong className="agent-workspace-memory-section-title">
                            滚动摘要触发（相对主模型上下文窗口的比例）
                          </Text>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="主模型上下文窗口（Token）"
                                tip="用于估算「历史总 token ÷ 窗口」；与挂载主模型真实窗口不一致时请在此覆盖（与后端 summary_context_window_tokens 一致）。"
                              />
                            }
                          >
                            <InputNumber
                              min={512}
                              max={2_000_000}
                              step={1024}
                              value={memoryWorkspace.summaryContextWindowTokens}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summaryContextWindowTokens:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(512, Math.min(2_000_000, Math.round(v)))
                                      : prev.summaryContextWindowTokens,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="常规触发阈值"
                                tip="历史总 token 占比超过该值时，可触发摘要任务（与新增轮次/token 条件配合，由服务端策略实现）。"
                              />
                            }
                          >
                            <Slider
                              min={0.05}
                              max={0.95}
                              step={0.05}
                              value={memoryWorkspace.summaryContextRatioThreshold}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summaryContextRatioThreshold: v,
                                }))
                              }
                              tooltip={{ formatter: (x) => (x != null ? `${Math.round(x * 100)}%` : '') }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="紧急触发阈值"
                                tip="超过该比例时按「紧急」优先级调度摘要，避免撑满上下文。"
                              />
                            }
                          >
                            <Slider
                              min={0.05}
                              max={0.95}
                              step={0.05}
                              value={memoryWorkspace.summaryContextRatioUrgent}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summaryContextRatioUrgent: v,
                                }))
                              }
                              tooltip={{ formatter: (x) => (x != null ? `${Math.round(x * 100)}%` : '') }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="距上次摘要最少新增轮数"
                                tip="自上次摘要成功以来，至少新增的完整对话轮数（与 token 条件二选一或组合由服务端决定）。"
                              />
                            }
                          >
                            <InputNumber
                              min={1}
                              max={500}
                              value={memoryWorkspace.summaryMinRoundsSinceLast}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summaryMinRoundsSinceLast:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(1, Math.min(500, Math.round(v)))
                                      : prev.summaryMinRoundsSinceLast,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="距上次摘要最少新增 Token"
                                tip="自上次摘要成功以来，至少新增的估算 token 数。"
                              />
                            }
                          >
                            <InputNumber
                              min={100}
                              max={500000}
                              step={100}
                              value={memoryWorkspace.summaryMinTokensSinceLast}
                              onChange={(v) =>
                                setMemoryWorkspace((prev) => ({
                                  ...prev,
                                  summaryMinTokensSinceLast:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(100, Math.min(500000, Math.round(v)))
                                      : prev.summaryMinTokensSinceLast,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Space>
                      ),
                    },
                    {
                      key: 'kb',
                      label: '知识库',
                      children: (
                        <Space
                          className="agent-workspace-knowledge-stack"
                          direction="vertical"
                          size={12}
                          style={{ width: '100%' }}
                        >
                          <div className="agent-workspace-switch-row">
                            <Text type="secondary">启用自动检索</Text>
                            <Switch
                              checked={knowledgeWorkspace.enabled}
                              onChange={(v) =>
                                setKnowledgeWorkspace((prev) => ({ ...prev, enabled: v }))
                              }
                            />
                          </div>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="绑定知识库"
                                tip="可多选，最多 8 个；每库独立检索后再合并为参考上下文。"
                              />
                            }
                          >
                            <Select
                              mode="multiple"
                              allowClear
                              loading={agentKbListLoading}
                              placeholder="选择知识库"
                              options={agentKbOptions}
                              value={knowledgeWorkspace.knowledgeBaseIds}
                              onChange={(ids) =>
                                setKnowledgeWorkspace((prev) => ({
                                  ...prev,
                                  knowledgeBaseIds: (ids as number[]).slice(0, 8),
                                }))
                              }
                              style={{ width: '100%' }}
                              disabled={!knowledgeWorkspace.enabled}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="每库召回条数（Top K）"
                                tip="与 POST /api/knowledge/{id}/search 的 limit 一致，单库 1～30。"
                              />
                            }
                          >
                            <InputNumber
                              min={1}
                              max={30}
                              value={knowledgeWorkspace.topK}
                              onChange={(v) =>
                                setKnowledgeWorkspace((prev) => ({
                                  ...prev,
                                  topK:
                                    typeof v === 'number' && !Number.isNaN(v)
                                      ? Math.max(1, Math.min(30, Math.round(v)))
                                      : prev.topK,
                                }))
                              }
                              style={{ width: '100%' }}
                              disabled={!knowledgeWorkspace.enabled}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <AgentWorkspaceFieldLabel
                                title="检索方式覆盖"
                                tip="inherit：与每个知识库自身 retrieval 配置一致；其余为单次调用覆盖。"
                              />
                            }
                          >
                            <Select<KnowledgeWorkspaceState['retrievalOverride']>
                              style={{ width: '100%' }}
                              value={knowledgeWorkspace.retrievalOverride}
                              onChange={(v) =>
                                setKnowledgeWorkspace((prev) => ({ ...prev, retrievalOverride: v }))
                              }
                              disabled={!knowledgeWorkspace.enabled}
                              options={[
                                { value: 'inherit', label: '与知识库配置一致（inherit）' },
                                { value: 'keyword', label: '关键词' },
                                { value: 'vector', label: '向量' },
                                { value: 'hybrid', label: '混合' },
                              ]}
                            />
                          </Form.Item>
                          <div className="agent-workspace-switch-row">
                            <Text type="secondary">在上下文中标注片段来源</Text>
                            <Tooltip title="开启后，每条检索分片下方追加知识库 id、文档 id、分片序号，便于模型在回答中引用来源。">
                              <span>
                                <Switch
                                  checked={knowledgeWorkspace.showSources}
                                  onChange={(v) =>
                                    setKnowledgeWorkspace((prev) => ({ ...prev, showSources: v }))
                                  }
                                  disabled={!knowledgeWorkspace.enabled}
                                />
                              </span>
                            </Tooltip>
                          </div>
                        </Space>
                      ),
                    },
                    {
                      key: 'advanced',
                      label: '高级',
                      children: (
                        <Space direction="vertical" size={12} style={{ width: '100%' }}>
                        <div>
                          <Text strong style={{ display: 'block', marginBottom: 4 }}>
                            输入 / 输出链
                          </Text>
                          <Text
                            type="secondary"
                            style={{ display: 'block', marginBottom: 8, fontSize: 12, lineHeight: 1.5 }}
                          >
                            与侧栏「保存配置」一起写入 <Text code>config_json.input_filter</Text>。开启后由后端在请求进入模型/工具前检查末条用户消息（禁词/正则/长度等，见下）。
                          </Text>
                          <Space direction="vertical" style={{ width: '100%' }} size={6}>
                            <div className="agent-workspace-switch-row">
                              <Text type="secondary">输入过滤</Text>
                              <Tooltip
                                title={
                                  isNumericAgentRoute
                                    ? '对应 input_filter.enabled；点「保存配置」后由后端在编排前执行。'
                                    : '请先进入已入库 Agent 路由后再保存。'
                                }
                              >
                                <span>
                                  <Switch
                                    checked={inputFilterWorkspace.enabled}
                                    onChange={(v) =>
                                      setInputFilterWorkspace((prev) => ({ ...prev, enabled: v }))
                                    }
                                    disabled={!isNumericAgentRoute}
                                  />
                                </span>
                              </Tooltip>
                            </div>
                            {isNumericAgentRoute ? (
                              <Space
                                direction="vertical"
                                style={{ width: '100%', marginTop: 2 }}
                                size={8}
                              >
                                <div>
                                  <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                                    禁词（每行一条，子串匹配，不区分大小写）
                                  </Text>
                                  <Input.TextArea
                                    className="agent-prompt-field__input"
                                    rows={3}
                                    value={inputFilterWorkspace.banned_keywords_text}
                                    placeholder="例如：某品牌名、敏感词"
                                    disabled={!isNumericAgentRoute}
                                    onChange={(e) =>
                                      setInputFilterWorkspace((p) => ({
                                        ...p,
                                        banned_keywords_text: e.target.value,
                                      }))
                                    }
                                  />
                                </div>
                                <div>
                                  <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                                    禁则（正则，每行一条，忽略大小写；无效行保存时由后端忽略）
                                  </Text>
                                  <Input.TextArea
                                    className="agent-prompt-field__input"
                                    rows={2}
                                    value={inputFilterWorkspace.banned_regex_text}
                                    placeholder="例如：password\\s*=\\s*\\S+"
                                    disabled={!isNumericAgentRoute}
                                    onChange={(e) =>
                                      setInputFilterWorkspace((p) => ({
                                        ...p,
                                        banned_regex_text: e.target.value,
                                      }))
                                    }
                                  />
                                </div>
                                <div className="agent-workspace-switch-row" style={{ alignItems: 'center' }}>
                                  <Text type="secondary">单条用户消息最大字符</Text>
                                  <InputNumber
                                    size="small"
                                    min={0}
                                    max={1_000_000}
                                    value={inputFilterWorkspace.max_user_chars ?? undefined}
                                    placeholder="不限制"
                                    disabled={!isNumericAgentRoute}
                                    onChange={(v) =>
                                      setInputFilterWorkspace((p) => ({
                                        ...p,
                                        max_user_chars:
                                          v == null || (typeof v === 'number' && Number.isNaN(v))
                                            ? null
                                            : Math.max(0, Math.round(v)),
                                      }))
                                    }
                                  />
                                </div>
                                <div className="agent-workspace-switch-row">
                                  <Text type="secondary">超长时截断末条用户</Text>
                                  <Tooltip title="需消息带 id 以便替换；未带 id 时与超长拒绝相同。关闭则超长直接拒绝。">
                                    <span>
                                      <Switch
                                        checked={inputFilterWorkspace.truncate_on_max}
                                        onChange={(v) =>
                                          setInputFilterWorkspace((p) => ({ ...p, truncate_on_max: v }))
                                        }
                                        disabled={!isNumericAgentRoute}
                                      />
                                    </span>
                                  </Tooltip>
                                </div>
                                <div>
                                  <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                                    拒答提示（未通过时给用户的助理回复）
                                  </Text>
                                  <Input
                                    className="agent-prompt-field__input"
                                    value={inputFilterWorkspace.reject_message}
                                    onChange={(e) =>
                                      setInputFilterWorkspace((p) => ({
                                        ...p,
                                        reject_message: e.target.value,
                                      }))
                                    }
                                    disabled={!isNumericAgentRoute}
                                  />
                                </div>
                              </Space>
                            ) : null}
                            <div className="agent-workspace-switch-row">
                              <Text type="secondary">输出干预</Text>
                              <Tooltip title="待接入中间件">
                                <span>
                                  <Switch disabled checked={false} />
                                </span>
                              </Tooltip>
                            </div>
                          </Space>
                        </div>
                        <Divider style={{ margin: '4px 0' }} />
                        <div>
                          <Text strong style={{ display: 'block', marginBottom: 6 }}>
                            结构化输出
                          </Text>
                          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
                            response.response_format
                          </Text>
                          <Select<ResponseFormat>
                            style={{ width: '100%' }}
                            value={responseFormat}
                            onChange={setResponseFormat}
                            options={[
                              { label: 'text', value: 'text' },
                              { label: 'json_object', value: 'json_object' },
                              { label: 'json_schema', value: 'json_schema' },
                            ]}
                          />
                        </div>
                        <Divider style={{ margin: '4px 0' }} />
                        <Space direction="vertical" style={{ width: '100%' }} size={6}>
                          <div className="agent-workspace-switch-row">
                            <Text type="secondary">显示工具调用</Text>
                            <Switch
                              checked={includeToolMessagesInRaw}
                              onChange={setIncludeToolMessagesInRaw}
                            />
                          </div>
                          <div className="agent-workspace-switch-row">
                            <Text type="secondary">Human-in-the-loop</Text>
                            <Tooltip title="待接入中断与确认流">
                              <span>
                                <Switch disabled checked={false} />
                              </span>
                            </Tooltip>
                          </div>
                        </Space>
                        <Divider style={{ margin: '4px 0' }} />
                        <div>
                          <Text strong style={{ display: 'block', marginBottom: 8 }}>
                            output
                          </Text>
                          <Space direction="vertical" style={{ width: '100%' }} size={8}>
                            <div className="agent-workspace-switch-row">
                              <Text type="secondary">剥离 thinking 块</Text>
                              <Switch checked={stripThinking} onChange={setStripThinking} />
                            </div>
                          </Space>
                        </div>
                      </Space>
                      ),
                    },
                  ]}
                />
                  ) : null}
                </Space>
              </div>
                </div>
                <div
                  className="agent-workspace-config-resizer"
                  onMouseDown={onConfigSidebarResizePointerDown}
                  role="separator"
                  aria-orientation="vertical"
                  aria-label="拖拽调整配置区宽度"
                />
              </div>
            </>
          ) : (
            <div className="agent-workspace-expand-rail">
              <Tooltip title="展开配置">
                <Button
                  type="text"
                  className="agent-workspace-expand-rail__btn"
                  icon={<MenuUnfoldOutlined />}
                  aria-label="展开配置"
                  onClick={() => setCollapsedConfig(false)}
                />
              </Tooltip>
            </div>
          )}
          <Card
            className="chat-panel agent-workspace-chat-panel"
            styles={{ header: { display: 'none' }, body: { padding: 0 } }}
          >
            <div className="agent-workspace-chat-panel-inner">
              <div className="agent-workspace-chat-topbar agent-workspace-chat-topbar--with-title">
                <div className="agent-workspace-chat-topbar__title-wrap">
                  <Tooltip
                    title={conversationSessionTitle ?? '对话'}
                    placement="bottomLeft"
                  >
                    <span className="agent-workspace-chat-topbar__title-text">
                      {conversationSessionTitle ?? '对话'}
                    </span>
                  </Tooltip>
                </div>
                <div className="agent-workspace-chat-topbar__actions">
                  <Tooltip title={isNumericAgentRoute ? '历史对话' : '请先进入已入库 Agent'}>
                    <Button
                      type="text"
                      size="small"
                      icon={<HistoryOutlined />}
                      aria-label="历史对话"
                      className="agent-workspace-chat-history-btn"
                      disabled={!isNumericAgentRoute}
                      onClick={openHistoryDrawer}
                    />
                  </Tooltip>
                  <Tooltip title="清空对话">
                    <Button
                      type="text"
                      size="small"
                      icon={<ClearOutlined />}
                      aria-label="清空对话"
                      className="agent-workspace-chat-clear-btn"
                      onClick={clearChat}
                    />
                  </Tooltip>
                </div>
              </div>

              <Drawer
                title="历史对话"
                placement="right"
                width={560}
                open={historyDrawerOpen}
                onClose={() => setHistoryDrawerOpen(false)}
                destroyOnClose
                styles={{ body: { paddingTop: 8 } }}
              >
                <Spin spinning={historyLoading || historyDetailLoading}>
                  {!historyLoading && historySessions.length === 0 ? (
                    <Empty description="暂无历史会话" />
                  ) : (
                    <>
                      <List
                        bordered={false}
                        className="agent-workspace-history-list"
                        dataSource={historySessions}
                        renderItem={(item) => (
                          <List.Item className="agent-workspace-history-list__item">
                            <button
                              type="button"
                              className="agent-workspace-history-list__btn"
                              disabled={historyDetailLoading}
                              onClick={() => void onSelectHistorySession(item)}
                            >
                              <span className="agent-workspace-history-list__title">
                                {item.title?.trim() || '未命名对话'}
                              </span>
                              <span className="agent-workspace-history-list__meta">
                                {new Date(item.updated_at).toLocaleString('zh-CN', {
                                  month: '2-digit',
                                  day: '2-digit',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })}
                              </span>
                            </button>
                          </List.Item>
                        )}
                      />
                      {historyTotal > AGENT_WORKSPACE_HISTORY_PAGE_SIZE ? (
                        <Pagination
                          className="agent-workspace-history-pagination"
                          size="small"
                          current={historyPage}
                          total={historyTotal}
                          pageSize={AGENT_WORKSPACE_HISTORY_PAGE_SIZE}
                          showSizeChanger={false}
                          onChange={(p) => void loadHistoryPage(p)}
                        />
                      ) : null}
                    </>
                  )}
                </Spin>
              </Drawer>

              <Drawer
                title={
                  chatAuxDrawer?.kind === 'execution_process'
                    ? '执行过程'
                    : chatAuxDrawer?.kind === 'knowledge'
                      ? '知识库来源'
                      : undefined
                }
                placement="right"
                width={560}
                open={chatAuxDrawer != null}
                onClose={() => setChatAuxDrawer(null)}
                destroyOnClose
                styles={{ body: { paddingTop: 12 } }}
              >
                {chatAuxDrawer?.kind === 'execution_process' ? (
                  <ExecutionProcessDrawerBody
                    steps={chatAuxDrawer.steps}
                    toolEntries={chatAuxDrawer.toolEntries}
                    loading={chatAuxDrawer.loading}
                    error={chatAuxDrawer.error}
                  />
                ) : chatAuxDrawer?.kind === 'knowledge' ? (
                  <KnowledgeCitationsDrawerBody citations={chatAuxDrawer.citations} />
                ) : null}
              </Drawer>

              <div
                ref={chatMessagesScrollRef}
                className="agent-workspace-chat-messages-scroll"
                onScroll={handleChatMessagesScroll}
              >
                {loading && streamOutput ? (
                  <div
                    className="agent-workspace-stream-exploring"
                    aria-busy="true"
                    aria-label="流式探索与编排输出"
                  >
                    <div className="agent-workspace-stream-exploring__head">
                      <span className="agent-workspace-stream-exploring__title">Exploring</span>
                      {streamExploringPhase ? (
                        <Text
                          type="secondary"
                          className="agent-workspace-stream-exploring__phase"
                          ellipsis={{ tooltip: streamExploringPhase }}
                          aria-label="流式流程状态"
                        >
                          {streamExploringPhase}
                        </Text>
                      ) : (
                        <Text type="secondary" className="agent-workspace-stream-exploring__phase-placeholder">
                          等待流式阶段…
                        </Text>
                      )}
                    </div>
                    <pre
                      ref={streamExploringScrollRef}
                      className="agent-workspace-stream-exploring__viewport"
                      aria-live="polite"
                    >
                      {streamExploringText.trim() ? streamExploringText : '…'}
                    </pre>
                  </div>
                ) : null}
                <div
                  className={`agent-workspace-messages-inner${isChatEmpty ? ' agent-workspace-messages-inner--empty' : ''}`}
                >
                  {canLoadOlderSessionMessages ? (
                    <div className="agent-workspace-load-older">
                      <Button
                        type="link"
                        size="small"
                        loading={historyDetailLoading}
                        onClick={() => void loadOlderSessionMessages()}
                      >
                        加载更早消息
                      </Button>
                    </div>
                  ) : null}
                  <div className="messages agent-workspace-messages">
                    {messages.map((m, i) => (
                      <div
                        key={`msg-${m.sourceMessageId ?? 'c'}-${m.at ?? i}-${m.role}-${i}`}
                        className={
                          m.role === 'user'
                            ? `agent-workspace-msg-row agent-workspace-msg-row--user${
                                hubRouteTargetsOrchestrator ? ' agent-workspace-msg-row--user--hub-meta' : ''
                              }`
                            : 'agent-workspace-msg-row agent-workspace-msg-row--assistant'
                        }
                      >
                        {hubRouteTargetsOrchestrator && m.role === 'user' ? (
                          <div className="agent-workspace-msg__user-meta agent-workspace-msg__user-meta--outside-bubble">
                            <Text type="secondary">
                              {m.userDisplayName?.trim()
                                ? `${m.userDisplayName.trim()}${m.at ? ` · ${formatChatLineTime(m.at)}` : ''}`
                                : m.at
                                  ? `提问 · ${formatChatLineTime(m.at)}`
                                  : '提问'}
                            </Text>
                          </div>
                        ) : null}
                        {hubRouteTargetsOrchestrator && m.role === 'assistant' && m.at ? (
                          <div className="agent-workspace-msg__assistant-meta agent-workspace-msg__assistant-meta--outside-bubble">
                            <Text type="secondary">
                              {resolveResponderAgentLabel(
                                m.responderAgentId ??
                                  (isNumericAgentRoute &&
                                  agentId?.trim() &&
                                  /^\d+$/.test(agentId.trim())
                                    ? Number(agentId.trim())
                                    : undefined),
                                workspaceAgentsInNs,
                                agentDisplayName,
                              ) + ` · ${formatChatLineTime(m.at)}`}
                            </Text>
                          </div>
                        ) : null}
                        <div
                          className={
                            m.role === 'user'
                              ? 'agent-workspace-msg-bubble agent-workspace-msg-bubble--user'
                              : 'agent-workspace-msg-bubble agent-workspace-msg-bubble--assistant'
                          }
                        >
                          <div className="agent-workspace-msg__body agent-workspace-msg__body--md">
                            {m.role === 'assistant' &&
                            loading &&
                            streamOutput &&
                            !m.content.trim() ? (
                              <div className="agent-workspace-msg__first-wait" aria-busy="true">
                                <Spin size="small" />
                              </div>
                            ) : m.role === 'assistant' ? (
                              <AssistantBubbleMarkdown text={m.content} thinkingText={m.thinkingText} />
                            ) : (
                              <ChatMarkdownBody text={m.content} />
                            )}
                          </div>
                          {shouldShowAssistantStructuredBlock(m) ? (
                            <AssistantStructuredBlock data={m.structured!} />
                          ) : null}
                        </div>
                        {m.role === 'assistant' &&
                        ((m.processTrace && m.processTrace.length >= 1) ||
                          (m.toolHistory && m.toolHistory.length > 0) ||
                          (conversationSessionId != null && m.sourceMessageId != null) ||
                          (m.knowledgeCitations && m.knowledgeCitations.length > 0)) ? (
                          <div className="agent-workspace-msg__aux-tags-wrap">
                            <Space wrap size={8}>
                              {(m.processTrace && m.processTrace.length >= 1) ||
                              (m.toolHistory && m.toolHistory.length > 0) ||
                              (conversationSessionId != null && m.sourceMessageId != null) ? (
                                <WorkspaceAuxDetailTag
                                  label="执行过程"
                                  onOpen={() => {
                                    const sid = conversationSessionId
                                    const mid = m.sourceMessageId
                                    if (sid && mid != null) {
                                      setChatAuxDrawer({ kind: 'execution_process', loading: true })
                                      void getConversationMessageExecution(sid, mid)
                                        .then((data) => {
                                          const parsed = parseExecutionApiToChatAux(data)
                                          setChatAuxDrawer({
                                            kind: 'execution_process',
                                            ...parsed,
                                          })
                                        })
                                        .catch((e: unknown) => {
                                          setChatAuxDrawer({
                                            kind: 'execution_process',
                                            loading: false,
                                            error:
                                              e instanceof Error
                                                ? e.message
                                                : typeof e === 'string'
                                                  ? e
                                                  : '加载执行过程失败',
                                          })
                                        })
                                      return
                                    }
                                    setChatAuxDrawer({
                                      kind: 'execution_process',
                                      ...(m.processTrace && m.processTrace.length >= 1
                                        ? { steps: m.processTrace }
                                        : {}),
                                      ...(m.toolHistory && m.toolHistory.length > 0
                                        ? { toolEntries: m.toolHistory }
                                        : {}),
                                    })
                                  }}
                                />
                              ) : null}
                              {m.knowledgeCitations && m.knowledgeCitations.length > 0 ? (
                                <WorkspaceAuxDetailTag
                                  label={`知识库来源（${m.knowledgeCitations.length}）`}
                                  onOpen={() =>
                                    setChatAuxDrawer({
                                      kind: 'knowledge',
                                      citations: m.knowledgeCitations!,
                                    })
                                  }
                                />
                              ) : null}
                            </Space>
                          </div>
                        ) : null}
                      </div>
                    ))}
                    {loading && !streamOutput ? (
                      <div className="agent-workspace-msg-row agent-workspace-msg-row--assistant">
                        <div className="agent-workspace-msg-bubble agent-workspace-msg-bubble--assistant agent-workspace-msg-bubble--loading">
                          <div className="agent-workspace-msg__first-wait" aria-busy="true">
                            <Spin size="small" />
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                  {isChatEmpty ? (
                    <p className="agent-workspace-chat-empty-hint">输入消息开始对话...</p>
                  ) : null}
                </div>
              </div>

              <div className="agent-workspace-chat-toolbar-inner">
                <Space size={8} wrap className="agent-workspace-chat-toolbar-switches">
                  <Tooltip title="开启后展示模型思考过程">
                    <Space size={6} align="center">
                      <Text type="secondary" className="agent-workspace-chat-toolbar__label">
                        思考
                      </Text>
                      <Switch
                        size="small"
                        checked={!stripThinking}
                        onChange={(on) => setStripThinking(!on)}
                        aria-label="思考模式"
                      />
                    </Space>
                  </Tooltip>
                  <Tooltip title="开启后通过 SSE 边收边展示回复">
                    <Space size={6} align="center">
                      <Text type="secondary" className="agent-workspace-chat-toolbar__label">
                        流式
                      </Text>
                      <Switch
                        size="small"
                        checked={streamOutput}
                        onChange={setStreamOutput}
                        aria-label="流式输出"
                      />
                    </Space>
                  </Tooltip>
                </Space>
              </div>

              <div className="agent-workspace-chat-footer">
                <div className="agent-workspace-chat-composer">
                  <Input.TextArea
                    className="agent-workspace-chat-composer__textarea"
                    placeholder="输入消息..."
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onPressEnter={(e) => {
                      if (!e.shiftKey) {
                        e.preventDefault()
                        void sendMessage()
                      }
                    }}
                  />
                  <div className="agent-workspace-chat-footer__actions">
                    {loading ? (
                      <Button
                        type="primary"
                        danger
                        size="small"
                        className="agent-workspace-chat-btn-stop"
                        onClick={stopInvoke}
                      >
                        停止
                      </Button>
                    ) : (
                      <Button
                        type="primary"
                        size="small"
                        className="agent-workspace-chat-btn-send"
                        onClick={() => void sendMessage()}
                      >
                        发送
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

const KNOWLEDGE_LIST_PAGE_SIZE = 10

export function KnowledgeListPage() {
  const navigate = useNavigate()
  const [statusFilter, setStatusFilter] = useState('全部')
  const [searchDraft, setSearchDraft] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [listError, setListError] = useState('')
  const [items, setItems] = useState<KnowledgeBaseOut[]>([])
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [createSubmitting, setCreateSubmitting] = useState(false)
  const [createForm] = Form.useForm<{ name: string }>()

  useEffect(() => {
    const t = window.setTimeout(() => {
      const next = searchDraft.trim()
      setDebouncedSearch((prev) => {
        if (prev !== next) setPage(1)
        return next
      })
    }, 350)
    return () => window.clearTimeout(t)
  }, [searchDraft])

  useEffect(() => {
    setPage(1)
  }, [statusFilter])

  const load = useCallback(async () => {
    setLoading(true)
    setListError('')
    try {
      const st = knowledgeStatusFilterToApi(statusFilter)
      const data = await fetchKnowledgeBasesList({
        page,
        page_size: KNOWLEDGE_LIST_PAGE_SIZE,
        q: debouncedSearch || undefined,
        status: st,
      })
      setItems(data.items)
      setTotal(data.total)
    } catch (e) {
      const msg = e instanceof Error ? e.message : '加载知识库列表失败'
      setListError(msg)
      setItems([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [debouncedSearch, page, statusFilter])

  useEffect(() => {
    void load()
  }, [load])

  const knowledgeTableColumns: ColumnsType<KnowledgeBaseOut> = useMemo(
    () => [
      {
        title: '名称',
        dataIndex: 'name',
        ellipsis: true,
        render: (name: string) => (
          <Tooltip title={name}>
            <strong className="knowledge-list-table-name">{name}</strong>
          </Tooltip>
        ),
      },
      {
        title: '说明',
        dataIndex: 'description',
        ellipsis: true,
        render: (text: string | null) => {
          const t = text || '—'
          return (
            <Tooltip title={t}>
              <span className="knowledge-list-table-desc">{t}</span>
            </Tooltip>
          )
        },
      },
      {
        title: 'slug',
        dataIndex: 'slug',
        width: 160,
        ellipsis: true,
        render: (v: string) => (
          <Tooltip title={v}>
            <span className="agent-tool-name-chip">{v}</span>
          </Tooltip>
        ),
      },
      {
        title: '检索 / 存储',
        key: 'retrieval',
        width: 148,
        ellipsis: true,
        render: (_: unknown, r: KnowledgeBaseOut) => (
          <Tooltip title={`${r.retrieval_type} · ${r.storage_type}`}>
            <span className="knowledge-list-table-meta">
              {r.retrieval_type} / {r.storage_type}
            </span>
          </Tooltip>
        ),
      },
      {
        title: (
          <TableColumnFilterHeader
            label="状态"
            tooltip="按生命周期筛选"
            active={statusFilter !== '全部'}
          >
            <Radio.Group
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value as string)
              }}
            >
              <Space direction="vertical" size={6}>
                <Radio value="全部">全部</Radio>
                <Radio value="已就绪">已就绪</Radio>
                <Radio value="待切块">待切块</Radio>
                <Radio value="空库">空库</Radio>
              </Space>
            </Radio.Group>
          </TableColumnFilterHeader>
        ),
        dataIndex: 'status',
        width: 112,
        render: (v: string) => {
          const color =
            v === 'ready' ? 'success' : v === 'processing' ? 'processing' : v === 'failed' ? 'error' : 'blue'
          return <Tag color={color}>{knowledgeStatusLabel(v)}</Tag>
        },
      },
      {
        title: 'ID',
        dataIndex: 'id',
        width: 88,
        align: 'center',
        render: (id: number) => <span className="knowledge-list-table-meta">{id}</span>,
      },
      {
        title: '操作',
        key: 'actions',
        width: 112,
        align: 'center',
        render: (_: unknown, row: KnowledgeBaseOut) => (
          <Space size={4} className="agent-list-actions-cell">
            <Tooltip title="进入详情">
              <Button
                type="text"
                className="agent-list-row-action"
                icon={<EyeOutlined />}
                aria-label="进入详情"
                onClick={() => navigate(`/resources/knowledge/${row.id}?step=1`)}
              />
            </Tooltip>
            <Popconfirm
              title="删除知识库"
              description="将软删该库及文档目录；对象存储与向量侧清理由后续任务完成。"
              okText="删除"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={async () => {
                try {
                  await deleteKnowledgeBase(row.id)
                  message.success('已删除')
                  await load()
                } catch (e: unknown) {
                  message.error(e instanceof Error ? e.message : '删除失败')
                }
              }}
            >
              <Tooltip title="删除">
                <span>
                  <Button
                    type="text"
                    danger
                    className="agent-list-row-action"
                    icon={<DeleteOutlined />}
                    aria-label="删除"
                  />
                </span>
              </Tooltip>
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [load, navigate, statusFilter],
  )

  return (
    <div className="agent-list-page knowledge-page knowledge-list-page">
      <div className="agent-list-table-shell knowledge-list-shell">
        <div className="agent-list-toolbar">
          <div className="agent-list-toolbar-left">
            <Space size={8} wrap>
              <Button
                type="primary"
                className="agent-list-action-primary"
                icon={<PlusOutlined />}
                onClick={() => setIsCreateOpen(true)}
              >
                新建知识库
              </Button>
            </Space>
          </div>
          <div className="agent-list-toolbar-right">
            <div className="agent-list-search-wrap tools-page-search">
              <Input
                className="agent-list-search-input"
                prefix={<SearchOutlined className="agent-list-search-icon" />}
                placeholder="搜索名称、slug、说明…"
                allowClear
                value={searchDraft}
                onChange={(e) => setSearchDraft(e.target.value)}
              />
            </div>
          </div>
        </div>

        {listError ? (
          <div className="agent-list-table-wrap" style={{ padding: 16 }}>
            <Alert type="error" message={listError} showIcon />
          </div>
        ) : loading ? (
          <div className="agent-list-table-wrap" style={{ padding: 24 }}>
            <Skeleton active paragraph={{ rows: 8 }} />
          </div>
        ) : (
          <div className="agent-list-table-wrap">
            <div className="agent-list-table-inner">
              <Table<KnowledgeBaseOut>
                className="agent-list-table"
                rowKey="id"
                size="small"
                columns={knowledgeTableColumns}
                dataSource={items}
                pagination={false}
                locale={{
                  emptyText: (
                    <div style={{ padding: '40px 0', textAlign: 'center' }}>
                      <Text type="secondary">暂无知识库数据</Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={() => setIsCreateOpen(true)}
                        >
                          新建知识库
                        </Button>
                      </div>
                    </div>
                  ),
                }}
              />
            </div>
            <div className="agent-list-pagination-bar">
              <Pagination
                size="small"
                current={page}
                pageSize={KNOWLEDGE_LIST_PAGE_SIZE}
                total={total}
                showTotal={(t) => `共 ${t} 条`}
                onChange={(p) => setPage(p)}
              />
            </div>
          </div>
        )}
      </div>

      <Modal
        title="新建知识库"
        open={isCreateOpen}
        okText="创建并进入详情"
        cancelText="取消"
        confirmLoading={createSubmitting}
        onCancel={() => setIsCreateOpen(false)}
        onOk={async () => {
          const values = await createForm.validateFields()
          setCreateSubmitting(true)
          try {
            const row = await createKnowledgeBase({ name: values.name })
            message.success('已创建')
            setIsCreateOpen(false)
            createForm.resetFields()
            navigate(`/resources/knowledge/${row.id}?step=1`)
          } catch (e) {
            message.error(e instanceof Error ? e.message : '创建失败')
          } finally {
            setCreateSubmitting(false)
          }
        }}
      >
        <Form form={createForm} layout="vertical">
          <Form.Item
            label="知识库名称"
            name="name"
            rules={[{ required: true, message: '请输入知识库名称' }]}
          >
            <Input placeholder="例如：产品知识库" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

function formatDocBytes(n: number | null): string {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function knowledgeRetrievalTypeLabel(t: KnowledgeSearchData['configured_retrieval']): string {
  const m: Record<string, string> = { keyword: '关键词', vector: '向量', hybrid: '混合' }
  return m[t] ?? t
}

function buildKnowledgeSearchHitColumns(query: string): ColumnsType<KnowledgeSearchHit> {
  const q = query.trim()
  return [
    { title: '文档 ID', dataIndex: 'doc_id', key: 'doc_id', width: 88 },
    {
      title: '分片',
      dataIndex: 'chunk_index',
      key: 'chunk_index',
      width: 72,
      render: (v: number | null) => (v != null ? String(v) : '—'),
    },
    { title: '文件名', dataIndex: 'filename', key: 'filename', ellipsis: true },
    {
      title: '片段预览',
      dataIndex: 'text_snippet',
      key: 'text_snippet',
      ellipsis: true,
      render: (v: string | null) => renderSnippetWithHighlight(v, q),
    },
    {
      title: 'RRF 分',
      dataIndex: 'score',
      key: 'score',
      width: 100,
      render: (v: number | null | undefined) =>
        typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : '—',
    },
    { title: '匹配', dataIndex: 'match_type', key: 'match_type', width: 100 },
  ]
}

function knowledgeDocFileIcon(filename: string): ReactNode {
  const ext = (filename.split('.').pop() || '').toLowerCase()
  if (ext === 'pdf')
    return <FilePdfOutlined className="knowledge-doc-icon knowledge-doc-icon--pdf" aria-hidden />
  if (ext === 'md' || ext === 'markdown')
    return <FileMarkdownOutlined className="knowledge-doc-icon knowledge-doc-icon--md" aria-hidden />
  if (ext === 'html' || ext === 'htm')
    return <FileTextOutlined className="knowledge-doc-icon knowledge-doc-icon--html" aria-hidden />
  if (ext === 'json')
    return <FileOutlined className="knowledge-doc-icon knowledge-doc-icon--json" aria-hidden />
  return <FileTextOutlined className="knowledge-doc-icon knowledge-doc-icon--txt" aria-hidden />
}

function knowledgeDocStatusDisplay(status: string): { color: string; label: string; icon: ReactNode } {
  const map: Record<string, { color: string; label: string; icon: ReactNode }> = {
    indexed: {
      color: 'success',
      label: '已完成',
      icon: <CheckCircleOutlined aria-hidden />,
    },
    processing: {
      color: 'processing',
      label: '处理中',
      icon: <LoadingOutlined aria-hidden />,
    },
    pending: {
      color: 'default',
      label: '待处理',
      icon: <ClockCircleOutlined aria-hidden />,
    },
    failed: { color: 'error', label: '失败', icon: <CloseCircleOutlined aria-hidden /> },
  }
  return map[status] ?? { color: 'default', label: status, icon: <FileOutlined aria-hidden /> }
}

/**
 * 切块第 2 步右侧「各文件当前策略」：仅展示「已 indexed 且已保存文档级切块覆盖」的文档，
 * 与左侧 ``chunk_left_panel`` 列表互斥（同一文件不会同时出现在两侧）。
 */
function knowledgeDocEligibleForChunkStrategyTable(doc: KnowledgeDocumentOut): boolean {
  if (doc.status !== 'indexed') return false
  const m = (doc.chunk_method ?? '').trim()
  return m.length > 0
}

function formatDocTime(iso: string): string {
  const s = iso.replace('T', ' ')
  if (s.length >= 19) return s.slice(0, 19)
  return s
}

function getKnowledgeDocColumns(opts: {
  kbId: number
  deletingId: number | null
  reingestingId: number | null
  setDeletingId: (id: number | null) => void
  onDeleted: () => Promise<void>
  onPreview: (doc: KnowledgeDocumentOut) => void
  /** 重新入队 ingest（失败为「重试」，其余为「重建索引」） */
  onReingest: (doc: KnowledgeDocumentOut) => void
  /** 文档管理步：点击文件名优先打开侧栏预览，不弹元数据 Modal */
  previewSelect?: (doc: KnowledgeDocumentOut) => void
}): ColumnsType<KnowledgeDocumentOut> {
  return [
    {
      title: '文件名',
      key: 'filename',
      ellipsis: true,
      render: (_, r) => (
        <Space size={8} className="knowledge-doc-name-cell">
          {knowledgeDocFileIcon(r.filename)}
          <Tooltip title={r.object_key ? `存储键：${r.object_key}` : r.filename}>
            <Button
              type="link"
              size="small"
              className="knowledge-doc-name-link"
              onClick={() =>
                opts.previewSelect ? opts.previewSelect(r) : opts.onPreview(r)
              }
            >
              {r.filename}
            </Button>
          </Tooltip>
        </Space>
      ),
    },
    {
      title: '格式',
      key: 'format',
      width: 100,
      ellipsis: true,
      render: (_, r) => (
        <Tooltip title={r.mime ?? '—'}>
          <span className="knowledge-doc-format">{knowledgeDocFormatLabel(r)}</span>
        </Tooltip>
      ),
    },
    {
      title: '大小',
      key: 'size',
      width: 108,
      align: 'right',
      render: (_, r) => (
        <span className="knowledge-doc-size">{formatDocBytes(r.size_bytes)}</span>
      ),
    },
    {
      title: '处理状态',
      key: 'status',
      width: 200,
      render: (_, r) => {
        const d = knowledgeDocStatusDisplay(r.status)
        return (
          <div className="knowledge-doc-status-cell">
            <Tag icon={d.icon} color={d.color}>
              {d.label}
            </Tag>
            {r.status === 'processing' ? (
              <Progress
                percent={38}
                size="small"
                showInfo={false}
                status="active"
                className="knowledge-doc-status-progress"
              />
            ) : null}
          </div>
        )
      },
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 188,
      render: (v: string) => formatDocTime(v),
    },
    {
      title: '操作',
      key: 'action',
      width: 168,
      render: (_, r) => (
        <Space size={4} wrap>
          {r.status === 'processing' ? (
            <Text type="secondary" style={{ fontSize: 12 }}>
              索引中…
            </Text>
          ) : (
            <Button
              type="link"
              size="small"
              icon={r.status === 'failed' ? <RedoOutlined /> : undefined}
              loading={opts.reingestingId === r.id}
              disabled={opts.deletingId != null}
              onClick={() => opts.onReingest(r)}
            >
              {r.status === 'failed' ? '重试' : '重建索引'}
            </Button>
          )}
          <Popconfirm
            title="确定删除该文档？"
            description="将软删记录、清理已写入的分片，并尽力删除对象存储中的文件。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ loading: opts.deletingId === r.id }}
            onConfirm={async () => {
              opts.setDeletingId(r.id)
              try {
                await deleteKnowledgeDocument(opts.kbId, r.id)
                message.success('已删除')
                await opts.onDeleted()
              } catch (e: unknown) {
                message.error(e instanceof Error ? e.message : '删除失败')
              } finally {
                opts.setDeletingId(null)
              }
            }}
          >
            <Button type="link" danger size="small" disabled={opts.deletingId != null}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]
}

export function KnowledgeDetailPage() {
  const { kbId } = useParams()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [maxStep, setMaxStep] = useState(1)
  const [kbLoading, setKbLoading] = useState(true)
  const [kbDetail, setKbDetail] = useState<KnowledgeBaseOut | null>(null)
  const [kbError, setKbError] = useState<string | null>(null)
  const [docPage, setDocPage] = useState(1)
  const [docPageSize] = useState(20)
  const [docsLoading, setDocsLoading] = useState(false)
  const [docsError, setDocsError] = useState<string | null>(null)
  const [docsItems, setDocsItems] = useState<KnowledgeDocumentOut[]>([])
  const [docsTotal, setDocsTotal] = useState(0)
  /** 第 2 步左侧「待配置」：不含已 indexed（与全量 docsItems 分页独立） */
  const [chunkPendingPage, setChunkPendingPage] = useState(1)
  const [chunkPendingItems, setChunkPendingItems] = useState<KnowledgeDocumentOut[]>([])
  const [chunkPendingTotal, setChunkPendingTotal] = useState(0)
  const [chunkPendingLoading, setChunkPendingLoading] = useState(false)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [chunkSaveLoading, setChunkSaveLoading] = useState(false)
  const [chunkSelectedRowKeys, setChunkSelectedRowKeys] = useState<Key[]>([])
  const [docChunkRevertId, setDocChunkRevertId] = useState<number | null>(null)
  const [docDeletingId, setDocDeletingId] = useState<number | null>(null)
  const [docPreviewOpen, setDocPreviewOpen] = useState(false)
  const [previewDoc, setPreviewDoc] = useState<KnowledgeDocumentOut | null>(null)
  /** 文档列表 / 切块预览弹窗：重新入队 ingest 的 loading */
  const [docReingestingId, setDocReingestingId] = useState<number | null>(null)
  const [uploadPercent, setUploadPercent] = useState<number | null>(null)
  /** 第 1 步上传：受控列表，配合 onChange 防抖触发批量上传（避免仅依赖 beforeUpload 末项 uid 导致不触发） */
  const [step1UploadFileList, setStep1UploadFileList] = useState<UploadFile[]>([])
  const step1UploadDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const step1UploadPendingRef = useRef<UploadFile[]>([])
  const [chunksPreviewOpen, setChunksPreviewOpen] = useState(false)
  const [chunksPreviewDocId, setChunksPreviewDocId] = useState<number | null>(null)
  const [chunksPreviewLoading, setChunksPreviewLoading] = useState(false)
  const [chunksPreviewData, setChunksPreviewData] = useState<KnowledgeDocumentChunksData | null>(
    null,
  )
  const [chunksPreviewPage, setChunksPreviewPage] = useState(1)
  const [chunksPreviewPageSize] = useState(10)
  /** 第 4 步「索引验证」：与 POST …/search 一致的试检索（向量/混合/关键词） */
  const [kbSearchDraft, setKbSearchDraft] = useState('')
  const [kbSearchLoading, setKbSearchLoading] = useState(false)
  const [kbSearchHits, setKbSearchHits] = useState<KnowledgeSearchHit[]>([])
  const [kbSearchMeta, setKbSearchMeta] = useState<Pick<
    KnowledgeSearchData,
    'configured_retrieval' | 'applied_retrieval' | 'note'
  > | null>(null)
  const [kbSearchRetrievalMode, setKbSearchRetrievalMode] = useState<
    'inherit' | 'keyword' | 'vector' | 'hybrid'
  >('inherit')
  const [kbSearchLimit, setKbSearchLimit] = useState(20)
  const [kbSearchRecallLimit, setKbSearchRecallLimit] = useState<number | null>(null)
  /** 混合 RRF 常数 k；null 表示用服务端默认（60） */
  const [kbSearchRrfK, setKbSearchRrfK] = useState<number | null>(null)
  /** 模块1：表格多选、右侧预览 */
  const [docManageSelectedKeys, setDocManageSelectedKeys] = useState<Key[]>([])
  const [m1PreviewDoc, setM1PreviewDoc] = useState<KnowledgeDocumentOut | null>(null)
  const [m1PreviewChunks, setM1PreviewChunks] = useState<KnowledgeDocumentChunksData | null>(null)
  const [m1PreviewLoading, setM1PreviewLoading] = useState(false)
  const [uploadFailureHint, setUploadFailureHint] = useState<string | null>(null)
  const [chunkScenarioKey, setChunkScenarioKey] = useState<string | null>(null)
  const [searchQueryHistory, setSearchQueryHistory] = useState<string[]>([])
  const [chunkForm] = Form.useForm<{
    chunk_size: number
    chunk_overlap: number
    chunk_method: string
    chunk_separator?: string | null
  }>()
  const [indexConfigForm] = Form.useForm<{
    /** 与后端 storage_type / retrieval_type 同时写入同一值 */
    index_mode: NonNullable<KnowledgeUpdateBody['storage_type']>
    embedding_model_config_id: number | null
    /** 写入 ``config_json.retrieval.top_k``，对应 ``/search`` 的 ``limit`` */
    retrieval_top_k: number
    /** 写入 ``config_json.retrieval.rrf_k``，混合检索默认 RRF 常数 */
    retrieval_rrf_k: number
  }>()
  const [saveAndBuildIndexLoading, setSaveAndBuildIndexLoading] = useState(false)
  const [embeddingModelsLoading, setEmbeddingModelsLoading] = useState(false)
  const [embeddingModels, setEmbeddingModels] = useState<LlmModel[]>([])
  const watchedChunkMethod = Form.useWatch('chunk_method', chunkForm)
  const watchedIndexMode = Form.useWatch('index_mode', indexConfigForm)
  const watchedRetrievalTopK = Form.useWatch('retrieval_top_k', indexConfigForm)
  const watchedRetrievalRrfK = Form.useWatch('retrieval_rrf_k', indexConfigForm)
  const effectiveChunkMethod = (() => {
    const raw = String(watchedChunkMethod ?? kbDetail?.chunk_method ?? 'char')
    const m = raw === 'md' ? 'markdown' : raw
    return m === 'length' ? 'char' : m
  })()
  const retrievalTopKLimit = useMemo(
    () => parseKnowledgeRetrievalTopK(kbDetail?.config_json ?? null),
    [kbDetail?.config_json],
  )
  useEffect(() => {
    setKbSearchLimit(retrievalTopKLimit)
  }, [retrievalTopKLimit])
  const currentStep = useMemo(() => Number(searchParams.get('step') ?? 1), [searchParams])

  /** 流程为 4 步：旧链接 step>4 时收敛到第 4 步 */
  useEffect(() => {
    if (!Number.isFinite(currentStep) || currentStep <= 4) return
    setSearchParams({ step: '4' }, { replace: true })
  }, [currentStep, setSearchParams])

  /** 与 URL `step` 对齐，避免深链或刷新后卡在不可达步骤 */
  useEffect(() => {
    setMaxStep((s) => Math.max(s, currentStep))
  }, [currentStep])

  /** 已有文档时解锁第 2 步（上传成功会刷新列表，亦依赖此项） */
  useEffect(() => {
    if (docsTotal > 0) {
      setMaxStep((s) => Math.max(s, 2))
    }
  }, [docsTotal])

  const kbNumericId = useMemo(() => {
    if (!kbId) return null
    const id = Number(kbId)
    return Number.isFinite(id) ? id : null
  }, [kbId])

  useEffect(() => {
    if (!kbId) {
      setKbDetail(null)
      setKbError('缺少知识库 ID')
      setKbLoading(false)
      return
    }
    const id = Number(kbId)
    if (!Number.isFinite(id)) {
      setKbDetail(null)
      setKbError('知识库 ID 须为数字')
      setKbLoading(false)
      return
    }
    setKbLoading(true)
    setKbError(null)
    void fetchKnowledgeBase(id)
      .then((row) => setKbDetail(row))
      .catch((e: unknown) =>
        setKbError(e instanceof Error ? e.message : '加载知识库失败'),
      )
      .finally(() => setKbLoading(false))
  }, [kbId])

  const loadDocuments = useCallback(
    (page: number) => {
      if (kbNumericId == null || kbError) return
      setDocsLoading(true)
      setDocsError(null)
      void fetchKnowledgeDocuments({
        kbId: kbNumericId,
        page,
        page_size: docPageSize,
      })
        .then((data) => {
          setDocsItems(data.items)
          setDocsTotal(data.total)
        })
        .catch((e: unknown) =>
          setDocsError(e instanceof Error ? e.message : '加载文档列表失败'),
        )
        .finally(() => setDocsLoading(false))
    },
    [kbNumericId, kbError, docPageSize],
  )

  const loadChunkPendingDocuments = useCallback(
    async (page: number) => {
      if (kbNumericId == null || kbError) return
      setChunkPendingLoading(true)
      try {
        let p = page
        let data = await fetchKnowledgeDocuments({
          kbId: kbNumericId,
          page: p,
          page_size: docPageSize,
          chunk_left_panel: true,
        })
        if (data.items.length === 0 && data.total > 0 && p > 1) {
          p = 1
          setChunkPendingPage(1)
          data = await fetchKnowledgeDocuments({
            kbId: kbNumericId,
            page: 1,
            page_size: docPageSize,
            chunk_left_panel: true,
          })
        }
        setChunkPendingItems(data.items)
        setChunkPendingTotal(data.total)
        setChunkSelectedRowKeys((keys) =>
          keys.filter((k) => data.items.some((doc) => doc.id === Number(k))),
        )
      } catch (e: unknown) {
        setDocsError(e instanceof Error ? e.message : '加载待配置列表失败')
      } finally {
        setChunkPendingLoading(false)
      }
    },
    [kbNumericId, kbError, docPageSize],
  )

  const reloadDocumentsKeepPagination = useCallback(async () => {
    if (kbNumericId == null || kbError) return
    setDocsLoading(true)
    setDocsError(null)
    try {
      let page = docPage
      let data = await fetchKnowledgeDocuments({
        kbId: kbNumericId,
        page,
        page_size: docPageSize,
      })
      if (data.items.length === 0 && data.total > 0 && page > 1) {
        page = 1
        setDocPage(1)
        data = await fetchKnowledgeDocuments({
          kbId: kbNumericId,
          page: 1,
          page_size: docPageSize,
        })
      }
      setDocsItems(data.items)
      setDocsTotal(data.total)
    } catch (e: unknown) {
      setDocsError(e instanceof Error ? e.message : '加载文档列表失败')
    } finally {
      setDocsLoading(false)
    }
    if (currentStep === 2) {
      void loadChunkPendingDocuments(chunkPendingPage)
    }
  }, [kbNumericId, kbError, docPage, docPageSize, currentStep, chunkPendingPage, loadChunkPendingDocuments])

  const runBatchUpload = useCallback(
    async (fileList: UploadFile[]) => {
      if (kbNumericId == null) return
      setUploadLoading(true)
      setUploadPercent(0)
      setUploadFailureHint(null)
      const failures: string[] = []
      const uploadedInBatch = new Set<string>()
      try {
        const list = fileList as UploadFile[]
        const n = list.length
        let done = 0
        let skipped = 0
        let processed = 0
        for (const f of list) {
          const raw = knowledgeUploadRawFile(f)
          if (!(raw instanceof File)) {
            failures.push(`${f.name || '未命名'}：无法读取本地文件，请重试`)
            continue
          }
          processed += 1
          if (!KNOWLEDGE_UPLOAD_EXT_RE.test(raw.name)) {
            failures.push(`${raw.name}：不支持的格式（仅 .md/.txt/.pdf/.docx/.json/.html）`)
            setUploadPercent(Math.round((processed / n) * 100))
            continue
          }
          const dup =
            docsItems.some((d) => d.filename === raw.name) || uploadedInBatch.has(raw.name)
          if (dup) {
            const go = await new Promise<boolean>((resolve) => {
              Modal.confirm({
                title: '同名文件已存在',
                content: `「${raw.name}」已在库中或本批已传，仍要继续上传吗？`,
                okText: '继续上传',
                cancelText: '跳过',
                onOk: () => resolve(true),
                onCancel: () => resolve(false),
              })
            })
            if (!go) {
              skipped += 1
              setUploadPercent(Math.round((processed / n) * 100))
              continue
            }
          }
          try {
            await uploadKnowledgeDocument(kbNumericId, raw)
            uploadedInBatch.add(raw.name)
            done += 1
            setUploadPercent(Math.round((processed / n) * 100))
          } catch (e: unknown) {
            failures.push(`${raw.name}：${e instanceof Error ? e.message : '上传失败'}`)
            setUploadPercent(Math.round((processed / n) * 100))
          }
        }
        if (done > 0) {
          message.success(`已成功上传 ${done} 个文件`)
          setMaxStep((s) => Math.max(s, 2))
          setDocPage(1)
          await loadDocuments(1)
        }
        if (failures.length > 0) {
          setUploadFailureHint(failures.slice(0, 8).join('\n'))
          message.warning(`部分文件未上传（${failures.length} 个）`)
        }
      } finally {
        setUploadLoading(false)
        setUploadPercent(null)
      }
    },
    [kbNumericId, loadDocuments, docsItems],
  )

  const onStep1UploadChange = useCallback(
    (info: { fileList: UploadFile[] }) => {
      setStep1UploadFileList(info.fileList)
      step1UploadPendingRef.current = info.fileList
      if (step1UploadDebounceRef.current) clearTimeout(step1UploadDebounceRef.current)
      step1UploadDebounceRef.current = setTimeout(() => {
        step1UploadDebounceRef.current = null
        const list = step1UploadPendingRef.current
        if (list.length === 0) return
        void runBatchUpload(list).finally(() => {
          setStep1UploadFileList([])
          step1UploadPendingRef.current = []
        })
      }, 100)
    },
    [runBatchUpload],
  )

  useEffect(() => {
    return () => {
      if (step1UploadDebounceRef.current) clearTimeout(step1UploadDebounceRef.current)
    }
  }, [])

  const loadM1DocPreviewChunks = useCallback(
    async (doc: KnowledgeDocumentOut) => {
      if (kbNumericId == null) return
      setM1PreviewLoading(true)
      setM1PreviewChunks(null)
      try {
        const data = await fetchDocumentChunks({
          kbId: kbNumericId,
          docId: doc.id,
          page: 1,
          page_size: 40,
        })
        setM1PreviewChunks(data)
      } catch {
        setM1PreviewChunks(null)
      } finally {
        setM1PreviewLoading(false)
      }
    },
    [kbNumericId],
  )

  const batchDeleteSelectedDocs = useCallback(async () => {
    if (kbNumericId == null || docManageSelectedKeys.length === 0) return
    const ids = docManageSelectedKeys.map((k) => Number(k)).filter((x) => Number.isFinite(x))
    Modal.confirm({
      title: '批量删除文档',
      content: `确定删除选中的 ${ids.length} 个文档吗？`,
      okText: '删除',
      okType: 'danger',
      onOk: async () => {
        let ok = 0
        for (const id of ids) {
          try {
            await deleteKnowledgeDocument(kbNumericId, id)
            ok += 1
          } catch {
            /* 单条失败继续 */
          }
        }
        message.success(`已删除 ${ok}/${ids.length} 个`)
        setDocManageSelectedKeys([])
        setM1PreviewDoc(null)
        setM1PreviewChunks(null)
        await reloadDocumentsKeepPagination()
      },
    })
  }, [kbNumericId, docManageSelectedKeys, reloadDocumentsKeepPagination])

  const loadChunkPreviewPage = useCallback(
    async (docId: number, page: number) => {
      if (kbNumericId == null) return
      setChunksPreviewLoading(true)
      try {
        const data = await fetchDocumentChunks({
          kbId: kbNumericId,
          docId,
          page,
          page_size: chunksPreviewPageSize,
        })
        setChunksPreviewData(data)
        setChunksPreviewDocId(docId)
        setChunksPreviewPage(page)
      } catch (e: unknown) {
        message.error(e instanceof Error ? e.message : '加载分片失败')
        setChunksPreviewData(null)
      } finally {
        setChunksPreviewLoading(false)
      }
    },
    [kbNumericId, chunksPreviewPageSize],
  )

  const reingestDocumentById = useCallback(
    async (docId: number) => {
      if (kbNumericId == null) return
      const row = docsItems.find((d) => d.id === docId)
      if (row?.status === 'processing') {
        message.warning('该文档正在索引中，请稍候再试')
        return
      }
      setDocReingestingId(docId)
      try {
        await postRequeueDocumentIngest(kbNumericId, docId)
        message.success('已重新入队索引任务；完成后将更新分片，并在配置 Milvus 时写入向量')
        await reloadDocumentsKeepPagination()
        if (chunksPreviewDocId === docId) {
          await loadChunkPreviewPage(docId, 1)
        }
      } catch (e: unknown) {
        message.error(e instanceof Error ? e.message : '重新入队失败')
      } finally {
        setDocReingestingId(null)
      }
    },
    [
      kbNumericId,
      docsItems,
      reloadDocumentsKeepPagination,
      chunksPreviewDocId,
      loadChunkPreviewPage,
    ],
  )

  const reingestDocument = useCallback(
    (doc: KnowledgeDocumentOut) => {
      void reingestDocumentById(doc.id)
    },
    [reingestDocumentById],
  )

  const openDocPreview = useCallback((doc: KnowledgeDocumentOut) => {
    setPreviewDoc(doc)
    setDocPreviewOpen(true)
  }, [])

  const saveChunkStrategyAndMaybePreview = useCallback(async () => {
    if (kbNumericId == null) return
    if (chunkSelectedRowKeys.length === 0) {
      message.warning('请先在左侧表格中勾选至少一个文档')
      return
    }
    try {
      const vals = await chunkForm.validateFields()
      setChunkSaveLoading(true)
      const sep = (vals.chunk_separator ?? '').trim()
      const payload = {
        chunk_size: vals.chunk_size,
        chunk_overlap: vals.chunk_overlap,
        chunk_method: vals.chunk_method,
        chunk_separator: sep === '' ? null : sep,
      }
      const ids = chunkSelectedRowKeys
        .map((k) => Number(k))
        .filter((n) => Number.isFinite(n))
      await Promise.all(
        ids.map((docId) => patchKnowledgeDocument(kbNumericId, docId, payload)),
      )
      message.success(`已写入 ${ids.length} 个文档的切块覆盖`)
      const fresh = await fetchKnowledgeDocuments({
        kbId: kbNumericId,
        page: docPage,
        page_size: docPageSize,
      })
      setDocsItems(fresh.items)
      setDocsTotal(fresh.total)
      void loadChunkPendingDocuments(chunkPendingPage)
      if (fresh.total === 0) {
        message.warning('请先上传至少一个文档；索引完成后方能在弹窗中看到分片。')
        return
      }
      // 预览文档须在「已写入 Mongo 分片」的文档中选：仅用当前页会误选到同页全部待处理文档，导致表格为空
      const wide = await fetchKnowledgeDocuments({
        kbId: kbNumericId,
        page: 1,
        page_size: 100,
      })
      const pool = wide.items
      const selectedInPool = pool.filter((x) => ids.includes(x.id))
      const preferred =
        selectedInPool.find((x) => x.status === 'indexed' && (x.chunk_count ?? 0) > 0) ??
        pool.find((x) => x.status === 'indexed' && (x.chunk_count ?? 0) > 0) ??
        selectedInPool.find((x) => x.status === 'indexed') ??
        pool.find((x) => x.status === 'indexed') ??
        selectedInPool[0] ??
        pool[0]
      setChunksPreviewOpen(true)
      await loadChunkPreviewPage(preferred.id, 1)
    } catch (e: unknown) {
      if (
        e &&
        typeof e === 'object' &&
        'errorFields' in e &&
        Array.isArray((e as { errorFields: unknown }).errorFields)
      ) {
        return
      }
      message.error(e instanceof Error ? e.message : '保存失败')
    } finally {
      setChunkSaveLoading(false)
    }
  }, [
    kbNumericId,
    chunkForm,
    docPageSize,
    docPage,
    chunkSelectedRowKeys,
    loadChunkPreviewPage,
    loadChunkPendingDocuments,
    chunkPendingPage,
  ])

  const knowledgeDocColumns = useMemo(
    () =>
      kbNumericId == null
        ? ([] as ColumnsType<KnowledgeDocumentOut>)
        : getKnowledgeDocColumns({
            kbId: kbNumericId,
            deletingId: docDeletingId,
            reingestingId: docReingestingId,
            setDeletingId: setDocDeletingId,
            onDeleted: reloadDocumentsKeepPagination,
            onPreview: openDocPreview,
            onReingest: reingestDocument,
            previewSelect:
              currentStep === 1 ? (doc: KnowledgeDocumentOut) => setM1PreviewDoc(doc) : undefined,
          }),
    [
      kbNumericId,
      docDeletingId,
      docReingestingId,
      reloadDocumentsKeepPagination,
      openDocPreview,
      reingestDocument,
      currentStep,
    ],
  )

  const knowledgeStepsItems: StepsProps['items'] = useMemo(() => {
    const si = Math.max(0, Math.min(currentStep - 1, 3))
    const nIndexed = docsItems.filter((d) => d.status === 'indexed').length
    const nProcessing = docsItems.filter((d) => d.status === 'processing').length
    return KNOWLEDGE_STEP_DEFINITIONS.map((def, i) => {
      let icon: ReactNode | undefined
      if (i < si) {
        icon = <CheckCircleOutlined className="knowledge-step-icon-done" aria-hidden />
      } else if (i === si) {
        if (i === 0 && uploadLoading && uploadPercent != null) {
          icon = (
            <Progress
              type="circle"
              percent={uploadPercent}
              size={32}
              showInfo={false}
              strokeWidth={10}
              status="active"
            />
          )
        } else if (
          i === 1 &&
          docsTotal > 0 &&
          (nProcessing > 0 || nIndexed < docsTotal)
        ) {
          const pct = Math.min(
            100,
            Math.max(0, Math.round((nIndexed / Math.max(docsTotal, 1)) * 100)),
          )
          icon = (
            <Progress
              type="circle"
              percent={nProcessing > 0 ? Math.max(pct, 12) : pct}
              size={32}
              showInfo={false}
              strokeWidth={10}
              status={nProcessing > 0 ? 'active' : 'success'}
            />
          )
        } else if (i === 2 && kbDetail != null) {
          const st = kbDetail.status
          if (st === 'failed') {
            icon = (
              <Progress
                type="circle"
                percent={100}
                size={32}
                showInfo={false}
                strokeWidth={10}
                status="exception"
              />
            )
          } else if (docsTotal > 0) {
            const pct = Math.min(
              100,
              Math.max(0, Math.round((nIndexed / Math.max(docsTotal, 1)) * 100)),
            )
            const stillIndexing = nProcessing > 0 || nIndexed < docsTotal
            icon = (
              <Progress
                type="circle"
                percent={
                  stillIndexing
                    ? nProcessing > 0
                      ? Math.max(pct, 12)
                      : pct
                    : 100
                }
                size={32}
                showInfo={false}
                strokeWidth={10}
                status={
                  stillIndexing
                    ? nProcessing > 0
                      ? 'active'
                      : 'normal'
                    : 'success'
                }
              />
            )
          } else if (st === 'processing') {
            icon = (
              <Progress
                type="circle"
                percent={55}
                size={32}
                showInfo={false}
                strokeWidth={10}
                status="active"
              />
            )
          } else if (st === 'ready' || st === 'empty' || st === 'draft') {
            icon = (
              <Progress
                type="circle"
                percent={100}
                size={32}
                showInfo={false}
                strokeWidth={10}
                status="success"
              />
            )
          } else {
            icon = (
              <Progress
                type="circle"
                percent={100}
                size={32}
                showInfo={false}
                strokeWidth={10}
                status="success"
              />
            )
          }
        } else if (i === 3) {
          icon = (
            <Progress
              type="circle"
              percent={100}
              size={32}
              showInfo={false}
              strokeWidth={10}
              status="success"
            />
          )
        }
      }
      return {
        title: def.title,
        status: (i < si ? 'finish' : i === si ? 'process' : 'wait') as
          | 'finish'
          | 'process'
          | 'wait',
        disabled: i > si,
        icon,
      }
    })
  }, [currentStep, uploadLoading, uploadPercent, docsItems, docsTotal, kbDetail])

  useEffect(() => {
    if (kbNumericId == null || kbError) return
    if (currentStep === 1) {
      setDocPage(1)
      loadDocuments(1)
    }
  }, [currentStep, kbNumericId, kbError, loadDocuments])

  useEffect(() => {
    if (kbNumericId == null || kbError || currentStep !== 2) return
    loadDocuments(docPage)
  }, [currentStep, kbNumericId, kbError, docPage, loadDocuments])

  useEffect(() => {
    if (kbNumericId == null || kbError || currentStep !== 3) return
    loadDocuments(docPage)
  }, [currentStep, kbNumericId, kbError, docPage, loadDocuments])

  /** 第 3 步：索引进行中定时刷新文档列表，圆环与文档状态同步 */
  useEffect(() => {
    if (kbNumericId == null || kbError || currentStep !== 3) return
    const hasProcessing = docsItems.some((d) => d.status === 'processing')
    if (!hasProcessing) return
    const id = window.setInterval(() => {
      void loadDocuments(docPage)
    }, 3000)
    return () => clearInterval(id)
  }, [
    kbNumericId,
    kbError,
    currentStep,
    docsItems,
    docPage,
    loadDocuments,
  ])

  const chunkStepEnteredRef = useRef(false)
  useEffect(() => {
    if (currentStep !== 2) {
      chunkStepEnteredRef.current = false
      return
    }
    if (!chunkStepEnteredRef.current) {
      chunkStepEnteredRef.current = true
      setChunkPendingPage(1)
    }
  }, [currentStep])

  useEffect(() => {
    if (kbNumericId == null || kbError || currentStep !== 2) return
    void loadChunkPendingDocuments(chunkPendingPage)
  }, [currentStep, kbNumericId, kbError, chunkPendingPage, loadChunkPendingDocuments])

  /** 第 2 步：有文档处于 ingest 时定时刷新，避免「已保存切块但仍显示处理中」直到手动刷新 */
  useEffect(() => {
    if (kbNumericId == null || kbError || currentStep !== 2) return
    const hasProcessing =
      chunkPendingItems.some((d) => d.status === 'processing') ||
      docsItems.some((d) => d.status === 'processing')
    if (!hasProcessing) return
    const id = window.setInterval(() => {
      void loadDocuments(docPage)
      void loadChunkPendingDocuments(chunkPendingPage)
    }, 3000)
    return () => clearInterval(id)
  }, [
    kbNumericId,
    kbError,
    currentStep,
    chunkPendingItems,
    docsItems,
    docPage,
    chunkPendingPage,
    loadDocuments,
    loadChunkPendingDocuments,
  ])

  useEffect(() => {
    if (currentStep !== 2 || kbDetail == null) return
    chunkForm.setFieldsValue({
      chunk_size: kbDetail.chunk_size,
      chunk_overlap: kbDetail.chunk_overlap,
      chunk_method:
        kbDetail.chunk_method === 'md'
          ? 'markdown'
          : kbDetail.chunk_method === 'length'
            ? 'char'
            : kbDetail.chunk_method,
      chunk_separator: kbDetail.chunk_separator ?? '',
    })
  }, [currentStep, kbDetail, chunkForm])

  useEffect(() => {
    if (currentStep !== 3 || kbNumericId == null || kbError) return
    setEmbeddingModelsLoading(true)
    void fetchAllEmbeddingModels()
      .then(setEmbeddingModels)
      .catch((e: unknown) =>
        message.error(e instanceof Error ? e.message : '加载向量模型列表失败'),
      )
      .finally(() => setEmbeddingModelsLoading(false))
  }, [currentStep, kbNumericId, kbError])

  useEffect(() => {
    if (currentStep !== 3 || kbDetail == null) return
    const st = kbDetail.storage_type as NonNullable<KnowledgeUpdateBody['storage_type']>
    indexConfigForm.setFieldsValue({
      index_mode: st,
      embedding_model_config_id: kbDetail.embedding_model_config_id,
      retrieval_top_k: parseKnowledgeRetrievalTopK(kbDetail.config_json ?? null),
      retrieval_rrf_k: parseKnowledgeRetrievalRrf(kbDetail.config_json ?? null),
    })
  }, [currentStep, kbDetail, indexConfigForm])

  useEffect(() => {
    if (kbNumericId == null) return
    try {
      const raw = sessionStorage.getItem(`knowledgeSearchHistory:${kbNumericId}`)
      if (!raw) return
      const j = JSON.parse(raw) as unknown
      if (Array.isArray(j)) {
        setSearchQueryHistory(j.filter((x) => typeof x === 'string').slice(0, 10) as string[])
      }
    } catch {
      /* ignore */
    }
  }, [kbNumericId])

  useEffect(() => {
    if (kbDetail?.config_json && typeof kbDetail.config_json === 'object') {
      setKbSearchRrfK(parseKnowledgeRetrievalRrf(kbDetail.config_json as Record<string, unknown>))
    }
  }, [kbDetail?.config_json])

  useEffect(() => {
    if (m1PreviewDoc == null || currentStep !== 1) {
      if (currentStep !== 1) setM1PreviewChunks(null)
      return
    }
    if (m1PreviewDoc.status === 'indexed' && (m1PreviewDoc.chunk_count ?? 0) > 0) {
      void loadM1DocPreviewChunks(m1PreviewDoc)
    } else {
      setM1PreviewChunks(null)
    }
  }, [m1PreviewDoc, currentStep, loadM1DocPreviewChunks])

  const runKbSearch = useCallback(
    async (queryOverride?: string) => {
      if (kbNumericId == null) return
      const t = (typeof queryOverride === 'string' ? queryOverride : kbSearchDraft).trim()
      if (!t) {
        message.warning('请输入检索关键词')
        return
      }
      const topK = kbSearchLimit >= 1 && kbSearchLimit <= 100 ? kbSearchLimit : 20
      if (kbSearchRecallLimit != null && kbSearchRecallLimit < topK) {
        message.warning('召回上限须大于等于 Top K')
        return
      }
      setKbSearchLoading(true)
      try {
        const data = await searchKnowledgeInKb(kbNumericId, {
          q: t,
          limit: topK,
          retrieval_override:
            kbSearchRetrievalMode === 'inherit' ? undefined : kbSearchRetrievalMode,
          recall_limit: kbSearchRecallLimit ?? undefined,
          rrf_k: kbSearchRrfK ?? undefined,
        })
        setKbSearchHits(data.items)
        setKbSearchMeta({
          configured_retrieval: data.configured_retrieval,
          applied_retrieval: data.applied_retrieval,
          note: data.note,
        })
        setSearchQueryHistory((prev) => {
          const next = [t, ...prev.filter((x) => x !== t)].slice(0, 10)
          try {
            sessionStorage.setItem(
              `knowledgeSearchHistory:${kbNumericId}`,
              JSON.stringify(next),
            )
          } catch {
            /* ignore */
          }
          return next
        })
      } catch (e: unknown) {
        message.error(e instanceof Error ? e.message : '检索失败')
        setKbSearchHits([])
        setKbSearchMeta(null)
      } finally {
        setKbSearchLoading(false)
      }
    },
    [
      kbNumericId,
      kbSearchDraft,
      kbSearchLimit,
      kbSearchRecallLimit,
      kbSearchRetrievalMode,
      kbSearchRrfK,
    ],
  )

  const saveIndexConfigAndBuildVectors = useCallback(async () => {
    if (kbNumericId == null || kbDetail == null) return
    try {
      const v = await indexConfigForm.validateFields()
      setSaveAndBuildIndexLoading(true)
      const mode = v.index_mode
      const prev =
        kbDetail.config_json && typeof kbDetail.config_json === 'object'
          ? { ...kbDetail.config_json }
          : {}
      const prevR =
        prev.retrieval && typeof prev.retrieval === 'object'
          ? { ...(prev.retrieval as Record<string, unknown>) }
          : {}
      const out = await updateKnowledgeBase(kbNumericId, {
        storage_type: mode,
        retrieval_type: mode,
        embedding_model_config_id: v.embedding_model_config_id,
        config_json: {
          ...prev,
          retrieval: {
            ...prevR,
            top_k: v.retrieval_top_k,
            rrf_k: v.retrieval_rrf_k,
          },
        },
      })
      setKbDetail(out)

      if (mode !== 'vector' && mode !== 'hybrid') {
        message.success('索引配置已保存')
        return
      }
      if (v.embedding_model_config_id == null) {
        message.warning(
          '索引配置已保存。向量/混合检索需选择 Embedding 模型；请选择后再次点击「保存并构建」。',
        )
        return
      }
      const data = await postBuildKnowledgeVectors(kbNumericId)
      const head = data.errors.slice(0, 5)
      const errTail = data.errors.length > 5 ? ` 等共 ${data.errors.length} 条` : ''
      const errMsg =
        head.length > 0 ? ` 失败说明：${head.join('；')}${errTail}` : ''
      const summary = `已向 Milvus 写入 ${data.built} 个文档的向量；跳过 ${data.skipped} 个。${errMsg}`.trim()
      if (data.errors.length > 0) {
        message.warning(summary)
      } else {
        message.success(summary)
      }
      if (data.built > 0) {
        setMaxStep((s) => Math.max(s, 4))
        setSearchParams({ step: '4' })
      }
    } catch (e: unknown) {
      if (
        e &&
        typeof e === 'object' &&
        'errorFields' in e &&
        Array.isArray((e as { errorFields: unknown }).errorFields)
      ) {
        return
      }
      message.error(e instanceof Error ? e.message : '保存或构建失败')
    } finally {
      setSaveAndBuildIndexLoading(false)
    }
  }, [kbNumericId, kbDetail, indexConfigForm, setSearchParams])

  const chunkMethodOptions = useMemo(() => {
    const base = [...KNOWLEDGE_CHUNK_METHOD_BASE_OPTIONS]
    const raw = kbDetail?.chunk_method
    if (kbDetail == null || !raw) return base
    const normalized = raw === 'md' ? 'markdown' : raw === 'length' ? 'char' : raw
    if (base.some((o) => o.value === normalized)) return base
    return [...base, { label: `当前库：${raw}`, value: normalized }]
  }, [kbDetail])

  const chunkDocPickColumns = useMemo<ColumnsType<KnowledgeDocumentOut>>(
    () => [
      { title: '文件名', dataIndex: 'filename', key: 'filename', ellipsis: true },
      {
        title: (
          <span>
            索引进度
            <Tooltip title="后台解析原文并写入 Mongo 分片；保存左侧切块覆盖不会单独结束此状态，需等 ingest 完成。">
              <InfoCircleOutlined style={{ marginLeft: 6, opacity: 0.65 }} aria-label="说明" />
            </Tooltip>
          </span>
        ),
        key: 'status',
        width: 168,
        render: (_, r) => {
          const d = knowledgeDocStatusDisplay(r.status)
          const hasChunkOverride = Boolean((r.chunk_method ?? '').trim())
          const ingestLabel =
            r.status === 'processing' && hasChunkOverride ? '索引进行中' : d.label
          const ingestTooltip =
            r.status === 'processing' && hasChunkOverride
              ? '切块覆盖已写入服务端；后台仍在解析原文并写入分片，与保存按钮是否成功无关。完成后此处变为「已完成」。'
              : r.status === 'processing'
                ? '全文解析与分片写入进行中；完成后将变为「已完成」。若长时间不变，请到第 1 步对该文档「重建索引」。'
                : undefined
          return (
            <div className="knowledge-doc-status-cell knowledge-doc-status-cell--chunk-pick">
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                {hasChunkOverride && r.status !== 'indexed' ? (
                  <Tooltip title="文档级切块参数（chunk_method 等）已持久化；下方为全文 ingest 状态。">
                    <Tag color="success">切块已保存</Tag>
                  </Tooltip>
                ) : null}
                <Tooltip title={ingestTooltip}>
                  <Tag icon={d.icon} color={d.color}>
                    {ingestLabel}
                  </Tag>
                </Tooltip>
              </Space>
            </div>
          )
        },
      },
    ],
    [],
  )

  const revertDocChunkToKbDefault = useCallback(
    async (docId: number) => {
      if (kbNumericId == null) return
      setDocChunkRevertId(docId)
      try {
        await patchKnowledgeDocument(kbNumericId, docId, { chunk_method: null })
        message.success('已恢复为知识库默认切块策略')
        await reloadDocumentsKeepPagination()
      } catch (e: unknown) {
        message.error(e instanceof Error ? e.message : '恢复失败')
      } finally {
        setDocChunkRevertId(null)
      }
    },
    [kbNumericId, reloadDocumentsKeepPagination],
  )

  const docChunkStrategyColumns = useMemo<ColumnsType<KnowledgeDocumentOut>>(
    () => [
      { title: '文件名', dataIndex: 'filename', key: 'filename', ellipsis: true },
      {
        title: '当前策略',
        key: 'strategy',
        render: (_, r) => {
          if (kbDetail == null) return '—'
          if (r.chunk_method) {
            const m =
              r.chunk_method === 'md'
                ? 'markdown'
                : r.chunk_method === 'length'
                  ? 'char'
                  : r.chunk_method
            const sep = (r.chunk_separator ?? '').trim()
            return (
              <span>
                {m} · {r.chunk_size ?? kbDetail.chunk_size} / {r.chunk_overlap ?? kbDetail.chunk_overlap}
                {sep ? ` · 分隔「${sep}」` : ''}
              </span>
            )
          }
          return (
            <Text type="secondary">
              知识库默认（
              {kbDetail.chunk_method === 'md'
                ? 'markdown'
                : kbDetail.chunk_method === 'length'
                  ? 'char'
                  : kbDetail.chunk_method}{' '}
              ·{' '}
              {kbDetail.chunk_size} / {kbDetail.chunk_overlap}）
            </Text>
          )
        },
      },
      {
        title: '操作',
        key: 'actions',
        width: 108,
        render: (_, r) =>
          r.chunk_method ? (
            <Popconfirm
              title="清除文档级覆盖？"
              description="将恢复为当前知识库的默认切块策略。"
              okText="恢复"
              cancelText="取消"
              onConfirm={() => void revertDocChunkToKbDefault(r.id)}
            >
              <Button type="link" size="small" loading={docChunkRevertId === r.id}>
                恢复默认
              </Button>
            </Popconfirm>
          ) : (
            <Text type="secondary">—</Text>
          ),
      },
    ],
    [kbDetail, docChunkRevertId, revertDocChunkToKbDefault],
  )

  const docsItemsChunkStrategyTable = useMemo(
    () => docsItems.filter(knowledgeDocEligibleForChunkStrategyTable),
    [docsItems],
  )

  const stepIndex = Math.max(0, Math.min(currentStep - 1, 3))

  const docManageStats = useMemo(() => {
    const bytes = docsItems.reduce((s, d) => s + (d.size_bytes ?? 0), 0)
    const estChunks = docsItems.reduce((s, d) => s + (d.chunk_count ?? 0), 0)
    return { bytes, estChunks }
  }, [docsItems])

  const knowledgeSearchHitColumnsDynamic = useMemo(
    () => buildKnowledgeSearchHitColumns(kbSearchDraft),
    [kbSearchDraft],
  )

  return (
    <div className="agent-list-page knowledge-page">
      <div className="agent-list-table-shell knowledge-detail-shell">
        <Spin spinning={kbLoading} wrapperClassName="knowledge-detail-spin-wrap">
          <div className="knowledge-detail-head">
            <Steps
              className="knowledge-detail-steps knowledge-detail-steps--ring-icons"
              type="default"
              current={stepIndex}
              items={knowledgeStepsItems}
              onChange={(next) => {
                const targetStep = next + 1
                if (targetStep <= maxStep) {
                  setSearchParams({ step: String(targetStep) })
                }
              }}
            />
          </div>
          <div className="knowledge-detail-body knowledge-detail-body--with-footer">
            {kbError ? (
              <Alert type="error" message={kbError} showIcon style={{ marginBottom: 16 }} />
            ) : null}
            <div className="knowledge-detail-main">
          {currentStep === 1 && kbNumericId != null && !kbError ? (
            <>
              <div className="knowledge-m1-layout">
                <div className="knowledge-m1-upload">
                  <Upload.Dragger
                    multiple
                    maxCount={20}
                    accept=".md,.txt,.pdf,.docx,.json,.html,.htm"
                    showUploadList={false}
                    disabled={uploadLoading}
                    className="knowledge-detail-dragger knowledge-detail-dragger--compact"
                    fileList={step1UploadFileList}
                    beforeUpload={() => false}
                    onChange={onStep1UploadChange}
                  >
                    <div className="knowledge-detail-dragger-compact-inner">
                      <Upload
                        multiple
                        maxCount={20}
                        accept=".md,.txt,.pdf,.docx,.json,.html,.htm"
                        showUploadList={false}
                        disabled={uploadLoading}
                        fileList={step1UploadFileList}
                        beforeUpload={() => false}
                        onChange={onStep1UploadChange}
                      >
                        <Button
                          type="primary"
                          icon={<UploadOutlined />}
                          loading={uploadLoading}
                          className="knowledge-detail-dragger-btn"
                          block
                        >
                          选择文件
                        </Button>
                      </Upload>
                      <p className="knowledge-detail-dragger-sub">
                        拖拽或点击上传 · 多选 · 不支持格式将提示并跳过；上传进度见顶部步骤「文档管理」圆形进度。
                      </p>
                    </div>
                  </Upload.Dragger>
                </div>
                <div className="knowledge-m1-body">
                  <div className="knowledge-detail-list-block knowledge-m1-list">
                    {docsError ? (
                      <Alert type="error" message={docsError} showIcon style={{ marginBottom: 12 }} />
                    ) : null}
                    {uploadFailureHint ? (
                      <Alert
                        type="warning"
                        showIcon
                        style={{ marginBottom: 12 }}
                        message="部分文件未处理"
                        description={<pre className="knowledge-m1-fail-pre">{uploadFailureHint}</pre>}
                      />
                    ) : null}
                    <Space wrap style={{ marginBottom: 8 }}>
                      <Button
                        danger
                        disabled={docManageSelectedKeys.length === 0}
                        onClick={() => void batchDeleteSelectedDocs()}
                      >
                        批量删除
                      </Button>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        勾选后可批量删除；点击列表行在右侧抽屉预览
                      </Text>
                    </Space>
                    <div className="knowledge-detail-table-wrap">
                      <Table<KnowledgeDocumentOut>
                        rowKey="id"
                        loading={docsLoading}
                        columns={knowledgeDocColumns}
                        dataSource={docsItems}
                        scroll={{ x: 'max-content' }}
                        pagination={false}
                        rowSelection={{
                          selectedRowKeys: docManageSelectedKeys,
                          onChange: (keys) => setDocManageSelectedKeys(keys),
                        }}
                        onRow={(record) => ({
                          onClick: () => setM1PreviewDoc(record),
                        })}
                      />
                    </div>
                    <div className="knowledge-detail-list-toolbar">
                      <Text type="secondary" className="knowledge-detail-list-toolbar-status">
                        共 {docsTotal} 个文件 · 总大小 {formatDocBytes(docManageStats.bytes)} · 已入库分片
                        约 {docManageStats.estChunks} 条
                      </Text>
                      <Pagination
                        className="knowledge-detail-list-pagination"
                        size="small"
                        current={docPage}
                        pageSize={docPageSize}
                        total={docsTotal}
                        showSizeChanger={false}
                        onChange={(p) => {
                          setDocPage(p)
                          loadDocuments(p)
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
              <Drawer
                className="knowledge-m1-preview-drawer"
                title="文件预览"
                placement="right"
                width={520}
                destroyOnClose
                open={m1PreviewDoc != null}
                onClose={() => {
                  setM1PreviewDoc(null)
                  setM1PreviewChunks(null)
                }}
                extra={
                  m1PreviewDoc ? (
                    <Button
                      type="link"
                      size="small"
                      onClick={() => {
                        openDocPreview(m1PreviewDoc)
                      }}
                    >
                      元数据
                    </Button>
                  ) : null
                }
                styles={{ body: { paddingTop: 12 } }}
              >
                {m1PreviewDoc ? (
                  <>
                    <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
                      {m1PreviewDoc.filename}
                    </Text>
                    {m1PreviewLoading ? (
                      <Spin />
                    ) : m1PreviewDoc.status !== 'indexed' || (m1PreviewDoc.chunk_count ?? 0) === 0 ? (
                      <Space direction="vertical" size="small">
                        <Tag>{knowledgeDocStatusDisplay(m1PreviewDoc.status).label}</Tag>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          索引完成后可在此查看分片文本；PDF/Word 等以分片文本为准。
                        </Text>
                      </Space>
                    ) : m1PreviewChunks?.items?.length ? (
                      <div className="knowledge-m1-preview-body">
                        {(m1PreviewDoc.filename.toLowerCase().endsWith('.md') ||
                          m1PreviewDoc.filename.toLowerCase().endsWith('.markdown')) &&
                        m1PreviewChunks.items[0]?.text ? (
                          <div className="knowledge-m1-md">
                            <Markdown
                              remarkPlugins={[remarkGfm]}
                              components={workspaceMarkdownComponents}
                            >
                              {m1PreviewChunks.items[0].text.slice(0, 12000)}
                            </Markdown>
                          </div>
                        ) : (
                          <pre className="knowledge-m1-code">
                            {(m1PreviewChunks.items[0]?.text ?? '').slice(0, 12000)}
                          </pre>
                        )}
                        {m1PreviewChunks.total > 1 ? (
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            仅展示首块；共 {m1PreviewChunks.total} 块。
                          </Text>
                        ) : null}
                      </div>
                    ) : (
                      <Text type="secondary">暂无分片数据</Text>
                    )}
                  </>
                ) : null}
              </Drawer>
            </>
          ) : currentStep === 2 && kbNumericId != null && !kbError ? (
            kbDetail == null ? (
              <div className="knowledge-detail-step-scroll">
                <Spin />
              </div>
            ) : (
              <div className="knowledge-detail-step-scroll knowledge-detail-step2-chunk-wrap">
                <div className="knowledge-detail-step2-chunk-grid">
                  <div className="knowledge-detail-step2-chunk-col">
                    <div className="knowledge-detail-step2-col-head">
                      <Title level={5} className="knowledge-detail-step2-section-title">
                        待配置文档
                      </Title>
                      <Button
                        type="primary"
                        variant="solid"
                        size="small"
                        loading={chunkSaveLoading}
                        onClick={() => void saveChunkStrategyAndMaybePreview()}
                      >
                        保存并查看
                      </Button>
                    </div>
                    <Table<KnowledgeDocumentOut>
                      rowKey="id"
                      size="small"
                      loading={chunkPendingLoading}
                      columns={chunkDocPickColumns}
                      dataSource={chunkPendingItems}
                      locale={{
                        emptyText: '暂无待配置文档（均已索引且已保存文档级覆盖，或列表为空）',
                      }}
                      rowSelection={{
                        selectedRowKeys: chunkSelectedRowKeys,
                        onChange: (keys) => setChunkSelectedRowKeys(keys),
                      }}
                      pagination={{
                        size: 'small',
                        current: chunkPendingPage,
                        pageSize: docPageSize,
                        total: chunkPendingTotal,
                        showSizeChanger: false,
                        onChange: (p) => {
                          setChunkPendingPage(p)
                          void loadChunkPendingDocuments(p)
                        },
                      }}
                      scroll={{ x: 'max-content' }}
                    />
                    <Space wrap align="center" style={{ marginTop: 10, marginBottom: 4 }}>
                      <Text type="secondary">场景预设</Text>
                      <Segmented
                        options={CHUNK_SCENARIO_PRESETS.map((s) => ({ label: s.label, value: s.key }))}
                        value={chunkScenarioKey ?? undefined}
                        onChange={(v) => {
                          const key = String(v)
                          setChunkScenarioKey(key)
                          const p = CHUNK_SCENARIO_PRESETS.find((x) => x.key === key)
                          if (p) {
                            chunkForm.setFieldsValue({
                              chunk_method: p.chunk_method,
                              chunk_size: p.chunk_size,
                              chunk_overlap: p.chunk_overlap,
                            })
                          }
                        }}
                      />
                    </Space>
                    <Form
                      form={chunkForm}
                      layout="vertical"
                      size="small"
                      className="knowledge-detail-step2-chunk-form"
                    >
                      <Form.Item
                        label="切块方式"
                        name="chunk_method"
                        rules={[{ required: true, message: '请选择切块方式' }]}
                        extra={
                          <span className="knowledge-detail-step2-chunk-method-extra">
                            {knowledgeChunkMethodHint(effectiveChunkMethod)}
                          </span>
                        }
                      >
                        <Select options={chunkMethodOptions} />
                      </Form.Item>
                      <Row gutter={[12, 0]} className="knowledge-detail-step2-chunk-params-row">
                        <Col xs={24} sm={12}>
                          <Form.Item
                            label={knowledgeChunkSizeLabel(effectiveChunkMethod)}
                            name="chunk_size"
                            rules={[
                              { required: true, message: '请输入切块大小' },
                              {
                                type: 'number',
                                min: 32,
                                max: 32768,
                                message: '须在 32～32768 之间',
                              },
                            ]}
                          >
                            <InputNumber min={32} max={32768} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={24} sm={12}>
                          <Form.Item
                            label={knowledgeChunkOverlapLabel(effectiveChunkMethod)}
                            name="chunk_overlap"
                            dependencies={['chunk_size', 'chunk_method']}
                            rules={[
                              { required: true, message: '请输入重叠长度' },
                              {
                                type: 'number',
                                min: 0,
                                max: 8192,
                                message: '须在 0～8192 之间',
                              },
                              ({ getFieldValue }) => ({
                                validator(_, value) {
                                  const size = getFieldValue('chunk_size') as number | undefined
                                  if (
                                    size != null &&
                                    value != null &&
                                    Number(value) >= Number(size)
                                  ) {
                                    return Promise.reject(new Error('重叠长度须小于目标块大小'))
                                  }
                                  return Promise.resolve()
                                },
                              }),
                            ]}
                            extra="相邻块重叠，利于衔接"
                          >
                            <InputNumber min={0} max={8192} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                      </Row>
                      {knowledgeChunkMethodShowsSeparator(effectiveChunkMethod) ? (
                        <Form.Item
                          label="自定义分隔串（可选）"
                          name="chunk_separator"
                          extra="仅对字符窗 / 句边界等模式生效；留空则使用后端默认。"
                        >
                          <Input placeholder="例如连续换行" allowClear maxLength={64} />
                        </Form.Item>
                      ) : null}
                    </Form>
                  </div>
                  <div className="knowledge-detail-step2-chunk-col knowledge-detail-step2-chunk-col--right">
                    <div className="knowledge-detail-step2-col-head">
                      <Title level={5} className="knowledge-detail-step2-section-title">
                        预览
                      </Title>
                      <Button
                        type="primary"
                        variant="outlined"
                        size="small"
                        onClick={() => {
                          const first = chunkPendingItems[0] ?? docsItems[0]
                          if (!first || kbNumericId == null) {
                            message.warning('请先上传文档并等待列表加载')
                            return
                          }
                          setChunksPreviewDocId(first.id)
                          setChunksPreviewOpen(true)
                          void loadChunkPreviewPage(first.id, 1)
                        }}
                      >
                        结果预览
                      </Button>
                    </div>
                    <Table<KnowledgeDocumentOut>
                      rowKey="id"
                      size="small"
                      loading={docsLoading}
                      columns={docChunkStrategyColumns}
                      dataSource={docsItemsChunkStrategyTable}
                      pagination={false}
                      scroll={{ x: 'max-content' }}
                      locale={{
                        emptyText: '暂无文档级切块覆盖；在左侧勾选并保存后将显示在此处。',
                      }}
                    />
                  </div>
                </div>
              </div>
            )
          ) : currentStep === 3 && kbNumericId != null && !kbError ? (
            kbDetail == null ? (
              <div className="knowledge-detail-step-scroll">
                <Spin />
              </div>
            ) : (
              <div className="knowledge-detail-step-scroll knowledge-m3-wrap">
                {kbDetail.storage_type !== kbDetail.retrieval_type ? (
                  <Alert
                    type="warning"
                    showIcon
                    style={{ marginBottom: 8 }}
                    message={`存储类型（${kbDetail.storage_type}）与检索类型（${kbDetail.retrieval_type}）不一致；保存后将统一为所选模式。`}
                  />
                ) : null}
                <Form
                  form={indexConfigForm}
                  layout="vertical"
                  size="small"
                  className="knowledge-detail-index-config-form knowledge-m3-index-form"
                >
                  <Title level={5} className="knowledge-detail-step2-section-title">
                    存储与检索模式
                  </Title>
                  <Form.Item
                    label="模式"
                    name="index_mode"
                    rules={[{ required: true, message: '请选择模式' }]}
                  >
                    <Select
                      options={KNOWLEDGE_INDEX_MODE_OPTIONS.map((o) => ({
                        value: o.value,
                        label: o.label,
                        title: o.description,
                      }))}
                    />
                  </Form.Item>
                  <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>
                    切换选项后可将下方说明作为参考；保存后写入知识库。
                  </Text>
                  <Divider style={{ margin: '16px 0' }} />
                  <Title level={5} className="knowledge-detail-step2-section-title">
                    向量模型
                  </Title>
                  <Form.Item label="Embedding 模型" name="embedding_model_config_id">
                    <Select
                      allowClear
                      placeholder="不选则无向量模型"
                      loading={embeddingModelsLoading}
                      options={embeddingModels.map((m) => ({
                        value: Number(m.id),
                        label: `${m.model_name}（${m.provider_name} · #${m.id}）`,
                      }))}
                    />
                  </Form.Item>
                  <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>
                    向量 / 混合检索需配置；在「模型提供商」维护模型。
                  </Text>
                  <Divider style={{ margin: '16px 0' }} />
                  <Title level={5} className="knowledge-detail-step2-section-title">
                    检索参数（默认）
                  </Title>
                  <Form.Item
                    label={
                      <span>
                        Top K
                        <Tooltip title="一次检索最多返回的分片条数，与试检索 / search 接口共用。">
                          <InfoCircleOutlined style={{ marginLeft: 6 }} />
                        </Tooltip>
                      </span>
                    }
                    name="retrieval_top_k"
                    rules={[
                      { required: true, message: '请输入 Top K' },
                      {
                        type: 'number',
                        min: 1,
                        max: 100,
                        message: '须在 1～100',
                      },
                    ]}
                  >
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                  <Slider
                    min={1}
                    max={100}
                    style={{ marginBottom: 16 }}
                    value={
                      typeof watchedRetrievalTopK === 'number' ? watchedRetrievalTopK : 20
                    }
                    onChange={(n) => indexConfigForm.setFieldsValue({ retrieval_top_k: n })}
                  />
                  <Form.Item
                    label={
                      <span>
                        RRF k（混合默认）
                        <Tooltip title="混合检索融合时常数，越大尾部排名影响越小；默认 60 适合多数场景。">
                          <InfoCircleOutlined style={{ marginLeft: 6 }} />
                        </Tooltip>
                      </span>
                    }
                    name="retrieval_rrf_k"
                    rules={[
                      { required: true, message: '请设置 RRF k' },
                      { type: 'number', min: 1, max: 200, message: '1～200' },
                    ]}
                  >
                    <InputNumber min={1} max={200} style={{ width: '100%' }} />
                  </Form.Item>
                  <Slider
                    min={1}
                    max={200}
                    value={typeof watchedRetrievalRrfK === 'number' ? watchedRetrievalRrfK : 60}
                    onChange={(n) => indexConfigForm.setFieldsValue({ retrieval_rrf_k: n })}
                  />
                  <Form.Item className="knowledge-detail-index-config-submit">
                    <Tooltip
                      title={
                        watchedIndexMode === 'vector' || watchedIndexMode === 'hybrid'
                          ? '先保存索引配置；随后从 Mongo 已有分片写入 Milvus（需服务端正確配置 MILVUS_URI）。若有文档构建失败，将在提示中列出原因。'
                          : '保存存储/检索模式与检索参数。'
                      }
                    >
                      <Button
                        type="primary"
                        className="knowledge-detail-index-save-btn"
                        loading={saveAndBuildIndexLoading}
                        onClick={() => void saveIndexConfigAndBuildVectors()}
                      >
                        保存并构建
                      </Button>
                    </Tooltip>
                  </Form.Item>
                </Form>
              </div>
            )
          ) : currentStep === 4 && kbNumericId != null && !kbError ? (
            <div className="knowledge-detail-step-scroll">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <div className="knowledge-detail-step4-query-row">
                  <div className="knowledge-detail-step4-query-params">
                    <Space wrap align="center" size={[12, 12]}>
                      <Space align="center" size={6}>
                        <Text type="secondary">检索策略</Text>
                        <Select
                          style={{ width: 200 }}
                          value={kbSearchRetrievalMode}
                          onChange={(v) => {
                            setKbSearchRetrievalMode(v)
                            setKbSearchHits([])
                            setKbSearchMeta(null)
                          }}
                          options={[
                            { value: 'inherit', label: '与知识库配置一致' },
                            { value: 'keyword', label: '关键词' },
                            { value: 'vector', label: '向量' },
                            { value: 'hybrid', label: '混合' },
                          ]}
                        />
                      </Space>
                      <Space align="center" size={6}>
                        <Text type="secondary">Top K</Text>
                        <InputNumber
                          min={1}
                          max={100}
                          value={kbSearchLimit}
                          onChange={(v) => {
                            const n = typeof v === 'number' ? v : kbSearchLimit
                            setKbSearchLimit(Number.isFinite(n) ? n : 20)
                            setKbSearchHits([])
                            setKbSearchMeta(null)
                          }}
                        />
                      </Space>
                      <Space align="center" size={6}>
                        <Text type="secondary">召回上限</Text>
                        <InputNumber
                          min={1}
                          max={100}
                          placeholder="自动"
                          value={kbSearchRecallLimit ?? undefined}
                          onChange={(v) => {
                            setKbSearchRecallLimit(v == null ? null : Number(v))
                            setKbSearchHits([])
                            setKbSearchMeta(null)
                          }}
                        />
                        <Tooltip title="各子路预取条数，须 ≥ Top K；混合/向量扩大召回。不填则服务端按默认公式。">
                          <InfoCircleOutlined aria-label="召回上限说明" />
                        </Tooltip>
                      </Space>
                      <Space align="center" size={6}>
                        <Text type="secondary">RRF k</Text>
                        <InputNumber
                          min={1}
                          max={200}
                          placeholder="默认 60"
                          value={kbSearchRrfK ?? undefined}
                          onChange={(v) => {
                            setKbSearchRrfK(v == null ? null : Number(v))
                            setKbSearchHits([])
                            setKbSearchMeta(null)
                          }}
                        />
                        <Tooltip title="混合检索倒数排名融合常数：贡献 1/(k+rank)。k 越大尾部排名影响越小；不填为服务端默认 60。">
                          <InfoCircleOutlined aria-label="RRF k 说明" />
                        </Tooltip>
                      </Space>
                    </Space>
                  </div>
                  <div className="agent-list-search-wrap knowledge-detail-step4-query-search-wrap tools-page-search">
                    <AutoComplete
                      value={kbSearchDraft}
                      options={searchQueryHistory.map((h) => ({ value: h, label: h }))}
                      onChange={(v) => setKbSearchDraft(v)}
                      onSelect={(v) => {
                        setKbSearchDraft(v)
                        void runKbSearch(v)
                      }}
                    >
                      <Input
                        className="agent-list-search-input"
                        prefix={<SearchOutlined className="agent-list-search-icon" />}
                        suffix={
                          kbSearchLoading ? (
                            <LoadingOutlined spin aria-label="检索中" />
                          ) : undefined
                        }
                        placeholder="输入查询词，回车检索"
                        allowClear
                        onPressEnter={() => void runKbSearch()}
                      />
                    </AutoComplete>
                  </div>
                </div>
                {kbSearchMeta ? (
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      配置检索：{knowledgeRetrievalTypeLabel(kbSearchMeta.configured_retrieval)}
                      {' · '}
                      本次执行：{knowledgeRetrievalTypeLabel(kbSearchMeta.applied_retrieval)}
                      {kbSearchMeta.applied_retrieval === 'hybrid' ? ' · RRF 融合' : null}
                    </Text>
                    {kbSearchMeta.note ? (
                      <Alert type="info" showIcon message={kbSearchMeta.note} />
                    ) : null}
                  </Space>
                ) : null}
                <Table<KnowledgeSearchHit>
                  rowKey={(_, i) => `kb-search-${i}`}
                  size="small"
                  pagination={false}
                  columns={knowledgeSearchHitColumnsDynamic}
                  dataSource={kbSearchHits}
                  locale={{ emptyText: '输入关键词后按回车检索' }}
                  scroll={{ x: 'max-content' }}
                />
              </Space>
            </div>
          ) : (
            <div className="knowledge-detail-step-scroll">
              <Text type="secondary">该步骤用于展示运维动作，支持 loading 与步骤解锁策略。</Text>
              <div className="progress-wrap">
                <Progress percent={0} />
              </div>
            </div>
          )}
            </div>
          </div>
        </Spin>
        {!kbError && kbNumericId != null ? (
          <div className="knowledge-detail-footer">
            <Space wrap className="knowledge-detail-footer-actions">
              {currentStep > 1 ? (
                <Button onClick={() => setSearchParams({ step: String(currentStep - 1) })}>
                  上一步
                </Button>
              ) : null}
              {currentStep === 1 ? (
                <Button
                  type="primary"
                  disabled={docsTotal === 0}
                  onClick={() => {
                    setMaxStep((s) => Math.max(s, 2))
                    setSearchParams({ step: '2' })
                  }}
                >
                  下一步
                </Button>
              ) : null}
              {currentStep === 2 ? (
                <Button
                  onClick={() => {
                    setMaxStep((s) => Math.max(s, 3))
                    setSearchParams({ step: '3' })
                  }}
                >
                  下一步：索引配置
                </Button>
              ) : null}
              {currentStep === 3 ? (
                <Button
                  type="primary"
                  onClick={() => {
                    setMaxStep((s) => Math.max(s, 4))
                    setSearchParams({ step: '4' })
                  }}
                >
                  下一步：索引验证
                </Button>
              ) : null}
              {currentStep === 4 ? (
                <Button type="primary" onClick={() => navigate('/resources/knowledge')}>
                  完成
                </Button>
              ) : null}
            </Space>
          </div>
        ) : null}
      </div>
      <Modal
        title="切块结果预览"
        open={chunksPreviewOpen}
        width={920}
        destroyOnClose
        onCancel={() => {
          setChunksPreviewOpen(false)
          setChunksPreviewData(null)
          setChunksPreviewDocId(null)
        }}
        footer={null}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <div className="knowledge-chunk-preview-toolbar">
            <Text type="secondary" className="knowledge-chunk-preview-toolbar-label">
              文档
            </Text>
            <Select
              className="knowledge-chunk-preview-toolbar-doc"
              placeholder="选择文档"
              value={chunksPreviewDocId ?? undefined}
              options={docsItems.map((d) => ({
                label: `${d.filename}（#${d.id} · ${knowledgeDocStatusDisplay(d.status).label}）`,
                value: d.id,
              }))}
              onChange={(id) => {
                void loadChunkPreviewPage(id, 1)
              }}
            />
          </div>
          {!chunksPreviewLoading &&
          chunksPreviewData != null &&
          (chunksPreviewData.total ?? 0) === 0 ? (
            <Alert
              type="warning"
              showIcon
              message={
                (chunksPreviewData.empty_hint && chunksPreviewData.empty_hint.trim()) ||
                (chunksPreviewData.document_status !== 'indexed'
                  ? `当前文档状态为「${chunksPreviewData.document_status}」，分片尚未写入 Mongo。请等待 ingest/索引完成后再预览，或在上方选择已 indexed 的文档。`
                  : '当前暂无分片记录。若文档刚上传，请稍等；若仅修改了切块策略，已入库内容需后台重建任务后才会按新策略重切（当前未接自动重切）。也可切换上方其他文档查看。')
              }
            />
          ) : null}
          <Table<KnowledgeChunkOut>
            rowKey={(r) => String(r.chunk_index)}
            size="small"
            loading={chunksPreviewLoading}
            columns={knowledgeChunkPreviewColumns}
            dataSource={chunksPreviewData?.items ?? []}
            locale={{
              emptyText:
                chunksPreviewLoading || chunksPreviewData == null
                  ? '加载中…'
                  : '暂无分片文本',
            }}
            pagination={false}
          />
          <div className="knowledge-chunk-preview-pagination-bar">
            <Pagination
              size="small"
              disabled={chunksPreviewDocId == null}
              current={chunksPreviewPage}
              pageSize={chunksPreviewPageSize}
              total={chunksPreviewData?.total ?? 0}
              showSizeChanger={false}
              onChange={(p) => {
                if (chunksPreviewDocId != null) {
                  void loadChunkPreviewPage(chunksPreviewDocId, p)
                }
              }}
            />
          </div>
        </Space>
      </Modal>
      <Modal
        title="文档信息"
        open={docPreviewOpen}
        width={520}
        destroyOnClose
        onCancel={() => {
          setDocPreviewOpen(false)
          setPreviewDoc(null)
        }}
        footer={
          <Button
            type="primary"
            onClick={() => {
              setDocPreviewOpen(false)
              setPreviewDoc(null)
            }}
          >
            关闭
          </Button>
        }
      >
        {previewDoc ? (
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <Paragraph style={{ marginBottom: 0 }}>
              <strong>文件名</strong>：{previewDoc.filename}
            </Paragraph>
            <Paragraph copyable style={{ marginBottom: 0 }}>
              <strong>存储键</strong>：{previewDoc.object_key ?? '—'}
            </Paragraph>
            <Text type="secondary">完整内容预览需从对象存储拉取，此处仅展示元数据。</Text>
          </Space>
        ) : null}
      </Modal>
    </div>
  )
}

type HttpToolFormValues = {
  name: string
  description: string
  url: string
  method: string
  headersText: string
  body: string
  timeout_seconds: number
  enabled: boolean
  input_schema_text: string
  output_schema_text: string
}

type CreateExternalToolModalValues = {
  name: string
  kind: 'http' | 'mcp'
}

type McpToolFormValues = {
  name: string
  description: string
  transport_type: 'http' | 'sse'
  server_url: string
  /** 远端 tools/call 的 name；直连模式必填；调度模式可留空 */
  mcp_tool_name: string
  /** 与后端 dispatch_mode 一致：先 list_tools 再 call_tool */
  dispatch_mode: boolean
  /** 写入 connection_config.protocol_version */
  protocol_version: string
  connection_config_text: string
  /** 除 mcp_tool_name / dispatch_mode 外的 tools_config 扩展键（JSON 对象，可选） */
  tools_config_extras_text: string
  enabled: boolean
}

type ExternalToolDrawerState =
  | null
  | { kind: 'http'; mode: 'create' | 'edit' }
  | { kind: 'mcp'; mode: 'create' | 'edit' }

const HTTP_METHOD_OPTIONS = [
  { label: 'GET', value: 'GET' },
  { label: 'HEAD', value: 'HEAD' },
  { label: 'POST', value: 'POST' },
  { label: 'PUT', value: 'PUT' },
  { label: 'PATCH', value: 'PATCH' },
  { label: 'DELETE', value: 'DELETE' },
]

const MCP_TRANSPORT_OPTIONS: { label: string; value: McpToolFormValues['transport_type'] }[] = [
  { label: 'HTTP (Streamable)', value: 'http' },
  { label: 'SSE', value: 'sse' },
]

export function ToolsPage() {
  const [externalDrawer, setExternalDrawer] = useState<ExternalToolDrawerState>(null)
  const [editingRecord, setEditingRecord] = useState<AgentRegisteredToolOut | null>(null)
  /** 编辑提交时携带，用于乐观锁；创建时为 undefined */
  const httpEditVersionRef = useRef<number | undefined>(undefined)
  const [httpSaving, setHttpSaving] = useState(false)
  const [httpForm] = Form.useForm<HttpToolFormValues>()
  const [mcpForm] = Form.useForm<McpToolFormValues>()
  const [registeredTools, setRegisteredTools] = useState<AgentRegisteredToolOut[]>([])
  const [persistedHttpByName, setPersistedHttpByName] = useState<
    Map<string, AgentHttpToolPersistedOut>
  >(() => new Map())
  const [toolsLoading, setToolsLoading] = useState(true)
  const [toolsError, setToolsError] = useState('')
  const [originFilter, setOriginFilter] = useState<'all' | 'builtin' | 'http' | 'mcp'>('all')
  const [toolsSearchDraft, setToolsSearchDraft] = useState('')
  const [debouncedToolsSearch, setDebouncedToolsSearch] = useState('')
  const [mcpProbeLoading, setMcpProbeLoading] = useState(false)
  const [mcpProbeOpen, setMcpProbeOpen] = useState(false)
  const [mcpProbeText, setMcpProbeText] = useState('')
  const [createToolModalOpen, setCreateToolModalOpen] = useState(false)
  const [createToolForm] = Form.useForm<CreateExternalToolModalValues>()

  useEffect(() => {
    const t = window.setTimeout(() => {
      setDebouncedToolsSearch(toolsSearchDraft.trim())
    }, 350)
    return () => window.clearTimeout(t)
  }, [toolsSearchDraft])

  const loadTools = useCallback(() => {
    setToolsLoading(true)
    setToolsError('')
    return Promise.all([fetchAgentRegisteredTools(), fetchPersistedHttpTools()])
      .then(([reg, http]) => {
        setRegisteredTools(reg.items)
        setPersistedHttpByName(new Map(http.items.map((i) => [i.name, i])))
      })
      .catch((e: unknown) => setToolsError(e instanceof Error ? e.message : '加载失败'))
      .finally(() => setToolsLoading(false))
  }, [])

  useEffect(() => {
    void loadTools()
  }, [loadTools])

  const filteredTools = useMemo(() => {
    let list =
      originFilter === 'all'
        ? registeredTools
        : registeredTools.filter((t) => t.origin === originFilter)
    const q = debouncedToolsSearch.toLowerCase()
    if (!q) return list
    return list.filter((t) => {
      const mcpUrl =
        typeof t.mcp_config?.server_url === 'string' ? t.mcp_config.server_url : ''
      const parts = [
        t.name,
        t.description,
        t.parameters_summary,
        t.http_request?.url ?? '',
        t.http_request?.method ?? '',
        mcpUrl,
      ]
      return parts.some((p) => p.toLowerCase().includes(q))
    })
  }, [registeredTools, originFilter, debouncedToolsSearch])

  const openCreateHttp = useCallback(
    (initialName?: string) => {
      setExternalDrawer({ kind: 'http', mode: 'create' })
      setEditingRecord(null)
      httpEditVersionRef.current = undefined
      httpForm.resetFields()
      const trimmed = initialName?.trim()
      httpForm.setFieldsValue({
        method: 'GET',
        timeout_seconds: 15,
        headersText: '',
        body: '',
        enabled: true,
        input_schema_text: '',
        output_schema_text: '',
        ...(trimmed ? { name: trimmed } : {}),
      })
    },
    [httpForm],
  )

  const openCreateMcp = useCallback(
    (initialName?: string) => {
      setExternalDrawer({ kind: 'mcp', mode: 'create' })
      setEditingRecord(null)
      httpEditVersionRef.current = undefined
      mcpForm.resetFields()
      const trimmed = initialName?.trim()
      mcpForm.setFieldsValue({
        transport_type: 'http',
        server_url: '',
        mcp_tool_name: '',
        dispatch_mode: false,
        protocol_version: '',
        connection_config_text: '',
        tools_config_extras_text: '',
        enabled: true,
        ...(trimmed ? { name: trimmed } : {}),
      })
    },
    [mcpForm],
  )

  const openCreateToolModal = useCallback(() => {
    createToolForm.resetFields()
    createToolForm.setFieldsValue({ kind: 'http' })
    setCreateToolModalOpen(true)
  }, [createToolForm])

  const submitCreateToolModal = useCallback(async () => {
    const values = await createToolForm.validateFields()
    const name = values.name.trim()
    setCreateToolModalOpen(false)
    createToolForm.resetFields()
    if (values.kind === 'http') {
      openCreateHttp(name)
    } else {
      openCreateMcp(name)
    }
  }, [createToolForm, openCreateHttp, openCreateMcp])

  const openEditHttp = useCallback(
    async (r: AgentRegisteredToolOut) => {
      const hr = r.http_request
      setExternalDrawer({ kind: 'http', mode: 'edit' })
      setEditingRecord(r)
      let persisted: AgentHttpToolPersistedOut | undefined
      try {
        const http = await fetchPersistedHttpTools()
        persisted = http.items.find((i) => i.name === r.name)
      } catch {
        message.warning('未能加载库表详情，启用状态与乐观锁版本可能不准确')
      }
      httpEditVersionRef.current = persisted?.version
      httpForm.setFieldsValue({
        name: r.name,
        description: r.description,
        url: hr?.url ?? '',
        method: hr?.method ?? 'GET',
        headersText: hr?.headers ? JSON.stringify(hr.headers, null, 2) : '',
        body: hr?.body ?? '',
        timeout_seconds: hr?.timeout_seconds ?? 15,
        enabled: persisted?.enabled ?? true,
        input_schema_text: persisted?.input_schema
          ? JSON.stringify(persisted.input_schema, null, 2)
          : '',
        output_schema_text: persisted?.output_schema
          ? JSON.stringify(persisted.output_schema, null, 2)
          : '',
      })
    },
    [httpForm],
  )

  const openEditMcp = useCallback(
    async (r: AgentRegisteredToolOut) => {
      setExternalDrawer({ kind: 'mcp', mode: 'edit' })
      setEditingRecord(r)
      let persisted: AgentHttpToolPersistedOut | undefined
      try {
        const http = await fetchPersistedHttpTools()
        persisted = http.items.find((i) => i.name === r.name)
      } catch {
        message.warning('未能加载库表详情，启用状态与乐观锁版本可能不准确')
      }
      httpEditVersionRef.current = persisted?.version
      const tt = persisted?.transport_type as McpToolFormValues['transport_type'] | undefined
      const suFromReg =
        r.mcp_config && typeof r.mcp_config.server_url === 'string'
          ? r.mcp_config.server_url
          : ''
      const pTc = persisted?.tools_config
      let mcpToolName = ''
      let dispatchMode = false
      let toolsConfigExtrasText = ''
      if (pTc && typeof pTc === 'object' && !Array.isArray(pTc)) {
        const o = { ...(pTc as Record<string, unknown>) }
        if (typeof o.mcp_tool_name === 'string') mcpToolName = o.mcp_tool_name
        if (o.dispatch_mode === true) dispatchMode = true
        delete o.mcp_tool_name
        delete o.dispatch_mode
        if (Object.keys(o).length > 0) {
          toolsConfigExtrasText = JSON.stringify(o, null, 2)
        }
      }
      const pCc = persisted?.connection_config
      let protocolVersion = ''
      if (pCc && typeof pCc === 'object' && !Array.isArray(pCc)) {
        const o = pCc as Record<string, unknown>
        if (typeof o.protocol_version === 'string') protocolVersion = o.protocol_version
      }
      mcpForm.setFieldsValue({
        name: r.name,
        description: r.description,
        transport_type: tt === 'sse' ? 'sse' : 'http',
        server_url: (typeof persisted?.server_url === 'string' && persisted.server_url
          ? persisted.server_url
          : suFromReg) || '',
        mcp_tool_name: mcpToolName,
        dispatch_mode: dispatchMode,
        protocol_version: protocolVersion,
        connection_config_text: persisted?.connection_config
          ? JSON.stringify(persisted.connection_config, null, 2)
          : '',
        tools_config_extras_text: toolsConfigExtrasText,
        enabled: persisted?.enabled ?? true,
      })
    },
    [mcpForm],
  )

  const openEditExternal = useCallback(
    (r: AgentRegisteredToolOut) => {
      if (r.origin === 'http') void openEditHttp(r)
      else if (r.origin === 'mcp') void openEditMcp(r)
    },
    [openEditHttp, openEditMcp],
  )

  const toolsTableColumns: ColumnsType<AgentRegisteredToolOut> = useMemo(
    () => [
      {
        title: '注册名',
        dataIndex: 'name',
        width: 128,
        ellipsis: true,
        render: (v: string) => (
          <Tooltip title={v}>
            <span className="agent-tool-name-chip">{v}</span>
          </Tooltip>
        ),
      },
      {
        title: '说明',
        dataIndex: 'description',
        ellipsis: true,
        render: (text: string) => (
          <Tooltip title={text || '—'}>
            <span style={{ color: '#4e5969' }}>{text || '—'}</span>
          </Tooltip>
        ),
      },
      {
        title: '请求 / MCP',
        key: 'http',
        width: 220,
        ellipsis: true,
        render: (_: unknown, r: AgentRegisteredToolOut) => {
          if (r.http_request) {
            return (
              <Tooltip title={`${r.http_request.method} ${r.http_request.url}`}>
                <span style={{ color: '#86909c' }}>
                  {r.http_request.method} {r.http_request.url}
                </span>
              </Tooltip>
            )
          }
          const su =
            typeof r.mcp_config?.server_url === 'string' ? r.mcp_config.server_url : ''
          if (su) {
            return (
              <Tooltip title={su}>
                <span style={{ color: '#86909c' }}>MCP {su}</span>
              </Tooltip>
            )
          }
          return <span style={{ color: '#86909c' }}>—</span>
        },
      },
      {
        title: '参数',
        dataIndex: 'parameters_summary',
        width: 160,
        ellipsis: true,
        render: (text: string) => (
          <Tooltip title={text}>
            <span style={{ color: '#4e5969' }}>{text}</span>
          </Tooltip>
        ),
      },
      {
        title: (
          <TableColumnFilterHeader
            label="来源"
            tooltip="按来源筛选"
            active={originFilter !== 'all'}
          >
            <Radio.Group
              value={originFilter}
              onChange={(e) => {
                setOriginFilter(e.target.value as 'all' | 'builtin' | 'http' | 'mcp')
              }}
            >
              <Space direction="vertical" size={6}>
                <Radio value="all">全部</Radio>
                <Radio value="builtin">内置</Radio>
                <Radio value="http">HTTP</Radio>
                <Radio value="mcp">MCP</Radio>
              </Space>
            </Radio.Group>
          </TableColumnFilterHeader>
        ),
        dataIndex: 'origin',
        width: 96,
        render: (v: AgentRegisteredToolOut['origin']) => (
          <Tag
            color={
              v === 'builtin' ? 'blue' : v === 'http' ? 'green' : v === 'mcp' ? 'cyan' : 'default'
            }
          >
            {v === 'builtin' ? '内置' : v === 'http' ? 'HTTP' : v === 'mcp' ? 'MCP' : '其他'}
          </Tag>
        ),
      },
      {
        title: '启用',
        key: 'enabled',
        width: 72,
        align: 'center',
        render: (_: unknown, r: AgentRegisteredToolOut) => {
          if (r.origin === 'http') {
            const row = persistedHttpByName.get(r.name)
            const on = row?.enabled ?? true
            return (
              <Tag color={on ? 'success' : 'default'}>{on ? '是' : '否'}</Tag>
            )
          }
          if (r.origin === 'mcp') {
            const row = persistedHttpByName.get(r.name)
            const on = row?.enabled ?? true
            return (
              <Tag color={on ? 'success' : 'default'}>{on ? '是' : '否'}</Tag>
            )
          }
          return <span style={{ color: '#c9cdd4' }}>—</span>
        },
      },
      {
        title: '操作',
        key: 'actions',
        width: 112,
        align: 'center',
        render: (_: unknown, r: AgentRegisteredToolOut) =>
          r.origin === 'http' || r.origin === 'mcp' ? (
            <Space size={4} className="agent-list-actions-cell">
              <Tooltip title="编辑">
                <Button
                  type="text"
                  className="agent-list-row-action"
                  icon={<EditOutlined />}
                  aria-label="编辑"
                  onClick={() => openEditExternal(r)}
                />
              </Tooltip>
              <Popconfirm
                title="删除后需重新添加才能使用，确定删除？"
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
                onConfirm={async () => {
                  try {
                    await deletePersistedHttpTool(r.name)
                    message.success('已删除')
                    await loadTools()
                  } catch (e: unknown) {
                    message.error(e instanceof Error ? e.message : '删除失败')
                  }
                }}
              >
                <Tooltip title="删除">
                  <span>
                    <Button
                      type="text"
                      danger
                      className="agent-list-row-action"
                      icon={<DeleteOutlined />}
                      aria-label="删除"
                    />
                  </span>
                </Tooltip>
              </Popconfirm>
            </Space>
          ) : (
            <span style={{ color: '#c9cdd4' }}>—</span>
          ),
      },
    ],
    [loadTools, openEditExternal, persistedHttpByName, originFilter],
  )

  const parseOptionalJsonObject = (
    raw: string | undefined,
    label: string,
  ): Record<string, unknown> | undefined => {
    const s = raw?.trim()
    if (!s) return undefined
    try {
      const parsed = JSON.parse(s) as unknown
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        throw new Error('invalid')
      }
      return parsed as Record<string, unknown>
    } catch {
      message.error(`${label}须为合法 JSON 对象`)
      return undefined
    }
  }

  const runMcpProbeListTools = async () => {
    if (externalDrawer?.kind !== 'mcp') return
    const values = mcpForm.getFieldsValue() as McpToolFormValues
    const su = values.server_url?.trim()
    if (!su) {
      message.warning('请先填写 MCP server_url')
      return
    }
    const connection_config_parsed = parseOptionalJsonObject(
      values.connection_config_text,
      'connection_config（连接扩展）',
    )
    if (connection_config_parsed === undefined && values.connection_config_text?.trim()) return
    const connection_config: Record<string, unknown> = {
      ...(connection_config_parsed &&
      typeof connection_config_parsed === 'object' &&
      !Array.isArray(connection_config_parsed)
        ? { ...(connection_config_parsed as Record<string, unknown>) }
        : {}),
    }
    if (values.protocol_version?.trim()) {
      connection_config.protocol_version = values.protocol_version.trim()
    }
    const hasConnectionConfig = Object.keys(connection_config).length > 0
    const transport_type: McpToolFormValues['transport_type'] =
      values.transport_type === 'sse' ? 'sse' : 'http'
    setMcpProbeLoading(true)
    try {
      const data = await probeMcpListTools({
        server_url: su,
        transport_type,
        ...(hasConnectionConfig ? { connection_config } : {}),
      })
      setMcpProbeText(JSON.stringify(data, null, 2))
      setMcpProbeOpen(true)
      message.success('tools/list 探测成功')
    } catch (e: unknown) {
      message.error(e instanceof Error ? e.message : '探测失败')
    } finally {
      setMcpProbeLoading(false)
    }
  }

  const onHttpSubmit = async (values: HttpToolFormValues) => {
    let headers: Record<string, string> | undefined
    const ht = values.headersText?.trim()
    if (ht) {
      try {
        const parsed = JSON.parse(ht) as unknown
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
          throw new Error('invalid')
        }
        headers = parsed as Record<string, string>
      } catch {
        message.error('请求头须为合法 JSON 对象，例如 {"Authorization":"Bearer …"}')
        return
      }
    }
    const input_schema = parseOptionalJsonObject(values.input_schema_text, '入参 JSON Schema')
    if (input_schema === undefined && values.input_schema_text?.trim()) return
    const output_schema = parseOptionalJsonObject(values.output_schema_text, '出参 JSON Schema')
    if (output_schema === undefined && values.output_schema_text?.trim()) return

    setHttpSaving(true)
    try {
      if (externalDrawer?.kind === 'http' && externalDrawer.mode === 'create') {
        await createPersistedHttpTool({
          name: values.name.trim(),
          description: values.description.trim(),
          url: values.url.trim(),
          method: values.method,
          headers,
          body: values.body?.trim() || undefined,
          timeout_seconds: values.timeout_seconds,
          enabled: values.enabled,
          ...(input_schema !== undefined ? { input_schema } : {}),
          ...(output_schema !== undefined ? { output_schema } : {}),
        })
        message.success('已保存到库表并注册到当前进程')
      } else if (externalDrawer?.kind === 'http' && externalDrawer.mode === 'edit' && editingRecord) {
        const ver = httpEditVersionRef.current
        await updatePersistedHttpTool(editingRecord.name, {
          description: values.description.trim(),
          url: values.url.trim(),
          method: values.method,
          headers,
          body: values.body?.trim() || undefined,
          timeout_seconds: values.timeout_seconds,
          enabled: values.enabled,
          ...(input_schema !== undefined ? { input_schema } : {}),
          ...(output_schema !== undefined ? { output_schema } : {}),
          ...(ver !== undefined ? { version: ver } : {}),
        })
        message.success('已更新')
      }
      setExternalDrawer(null)
      await loadTools()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '保存失败'
      if (msg.includes('版本冲突') || msg.includes('409')) {
        message.error(`${msg}，请关闭抽屉后重新打开再编辑`)
      } else {
        message.error(msg)
      }
    } finally {
      setHttpSaving(false)
    }
  }

  const onMcpSubmit = async (values: McpToolFormValues) => {
    if (!values.dispatch_mode && !values.mcp_tool_name?.trim()) {
      message.error('未开启调度模式时，须填写远端工具名（与远端 tools/list 中 name 一致）')
      return
    }
    const connection_config_parsed = parseOptionalJsonObject(
      values.connection_config_text,
      'connection_config（连接扩展）',
    )
    if (connection_config_parsed === undefined && values.connection_config_text?.trim()) return
    const tools_config_extras_parsed = parseOptionalJsonObject(
      values.tools_config_extras_text,
      '其它 tools_config',
    )
    if (tools_config_extras_parsed === undefined && values.tools_config_extras_text?.trim()) return

    const tools_config: Record<string, unknown> = {
      ...(tools_config_extras_parsed &&
      typeof tools_config_extras_parsed === 'object' &&
      !Array.isArray(tools_config_extras_parsed)
        ? { ...(tools_config_extras_parsed as Record<string, unknown>) }
        : {}),
    }
    if (values.dispatch_mode) {
      tools_config.dispatch_mode = true
      delete tools_config.mcp_tool_name
    } else {
      delete tools_config.dispatch_mode
      tools_config.mcp_tool_name = values.mcp_tool_name.trim()
    }

    const connection_config: Record<string, unknown> = {
      ...(connection_config_parsed &&
      typeof connection_config_parsed === 'object' &&
      !Array.isArray(connection_config_parsed)
        ? { ...(connection_config_parsed as Record<string, unknown>) }
        : {}),
    }
    if (values.protocol_version?.trim()) {
      connection_config.protocol_version = values.protocol_version.trim()
    }

    const hasToolsConfig = Object.keys(tools_config).length > 0
    const hasConnectionConfig = Object.keys(connection_config).length > 0

    setHttpSaving(true)
    try {
      if (externalDrawer?.kind === 'mcp' && externalDrawer.mode === 'create') {
        await createPersistedMcpTool({
          name: values.name.trim(),
          description: values.description.trim(),
          transport_type: values.transport_type,
          server_url: values.server_url.trim(),
          enabled: values.enabled,
          ...(hasConnectionConfig ? { connection_config } : {}),
          ...(hasToolsConfig ? { tools_config } : {}),
        })
        message.success('已保存到库表并注册到当前进程')
      } else if (externalDrawer?.kind === 'mcp' && externalDrawer.mode === 'edit' && editingRecord) {
        const ver = httpEditVersionRef.current
        await updatePersistedMcpTool(editingRecord.name, {
          description: values.description.trim(),
          transport_type: values.transport_type,
          server_url: values.server_url.trim(),
          enabled: values.enabled,
          ...(hasConnectionConfig ? { connection_config } : {}),
          ...(hasToolsConfig ? { tools_config } : {}),
          ...(ver !== undefined ? { version: ver } : {}),
        })
        message.success('已更新')
      }
      setExternalDrawer(null)
      await loadTools()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '保存失败'
      if (msg.includes('版本冲突') || msg.includes('409')) {
        message.error(`${msg}，请关闭抽屉后重新打开再编辑`)
      } else {
        message.error(msg)
      }
    } finally {
      setHttpSaving(false)
    }
  }

  const externalDrawerTitle =
    externalDrawer?.kind === 'http'
      ? externalDrawer.mode === 'edit'
        ? '编辑 HTTP 工具'
        : '添加 HTTP 工具'
      : externalDrawer?.kind === 'mcp'
        ? externalDrawer.mode === 'edit'
          ? '编辑 MCP 工具'
          : '添加 MCP 工具'
        : ''

  return (
    <div className="agent-list-page">
      <div className="agent-list-table-shell">
        <div className="agent-list-toolbar">
          <div className="agent-list-toolbar-left">
            <Button
              type="primary"
              className="agent-list-action-primary"
              icon={<PlusOutlined />}
              onClick={openCreateToolModal}
            >
              添加工具
            </Button>
          </div>
          <div className="tools-page-toolbar-right">
            <div className="agent-list-search-wrap tools-page-search">
              <Input
                className="agent-list-search-input"
                prefix={<SearchOutlined className="agent-list-search-icon" />}
                placeholder="搜索注册名、说明、URL…"
                allowClear
                value={toolsSearchDraft}
                onChange={(e) => setToolsSearchDraft(e.target.value)}
              />
            </div>
          </div>
        </div>

        {toolsError ? (
          <div className="agent-list-table-wrap" style={{ padding: 16 }}>
            <Alert type="error" message={toolsError} showIcon />
          </div>
        ) : (
          <div className="agent-list-table-wrap">
            <div className="agent-list-table-inner">
              <Table<AgentRegisteredToolOut>
                className="agent-list-table"
                size="small"
                rowKey="name"
                loading={toolsLoading}
                pagination={false}
                columns={toolsTableColumns}
                dataSource={filteredTools}
                locale={{
                  emptyText: (
                    <div style={{ padding: '40px 0', textAlign: 'center' }}>
                      <Text type="secondary">暂无工具数据</Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={openCreateToolModal}
                        >
                          添加工具
                        </Button>
                      </div>
                    </div>
                  ),
                }}
              />
            </div>
          </div>
        )}
      </div>

      <Drawer
        title={externalDrawerTitle}
        placement="right"
        width={640}
        open={externalDrawer !== null}
        onClose={() => setExternalDrawer(null)}
        destroyOnClose
        maskClosable
        footer={
          <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
            <Button onClick={() => setExternalDrawer(null)} disabled={httpSaving}>
              取消
            </Button>
            <Button
              type="primary"
              className="agent-list-btn-primary"
              loading={httpSaving}
              onClick={() => {
                if (externalDrawer?.kind === 'http') httpForm.submit()
                else if (externalDrawer?.kind === 'mcp') mcpForm.submit()
              }}
            >
              保存
            </Button>
          </Space>
        }
      >
        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
          保存后写入库表并注册到当前进程；重启后端会重放。HTTP 生产环境须鉴权并防范 SSRF。
          {externalDrawer?.mode === 'edit' ? (
            <>
              {' '}
              编辑保存时会做并发校验（非配置历史版本）；若提示冲突，请关闭抽屉后重新打开再保存。
            </>
          ) : null}
        </Paragraph>
        {externalDrawer?.kind === 'http' && (
          <Form<HttpToolFormValues>
            form={httpForm}
            layout="vertical"
            onFinish={(v) => void onHttpSubmit(v)}
          >
            <Form.Item
              label="注册名（逻辑名）"
              name="name"
              rules={
                externalDrawer.mode === 'create'
                  ? [
                      { required: true, message: '请输入注册名' },
                      {
                        pattern: /^[a-zA-Z][a-zA-Z0-9_]*$/,
                        message: '以字母开头，仅字母数字下划线',
                      },
                    ]
                  : []
              }
            >
              <Input
                placeholder="例如 my_status_api"
                disabled={externalDrawer.mode === 'edit'}
                autoComplete="off"
              />
            </Form.Item>
            <Form.Item
              label="说明（给模型看的描述）"
              name="description"
              rules={[{ required: true, message: '请输入说明' }]}
            >
              <Input.TextArea rows={2} placeholder="何时应调用此工具" />
            </Form.Item>
            <Form.Item label="HTTP 方法" name="method" rules={[{ required: true }]}>
              <Select options={HTTP_METHOD_OPTIONS} />
            </Form.Item>
            <Form.Item label="URL" name="url" rules={[{ required: true, message: '请输入 URL' }]}>
              <Input placeholder="https://..." />
            </Form.Item>
            <Form.Item label="超时（秒）" name="timeout_seconds" rules={[{ required: true }]}>
              <InputNumber min={1} max={120} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="启用" name="enabled" valuePropName="checked">
              <Switch checkedChildren="开" unCheckedChildren="关" />
            </Form.Item>
            <Collapse
              bordered={false}
              style={{ background: 'transparent' }}
              items={[
                {
                  key: 'http-advanced',
                  label: '高级（可选）：请求头、请求体、入参/出参 Schema',
                  children: (
                    <>
                      <Form.Item label="请求头（JSON 对象）" name="headersText">
                        <Input.TextArea
                          rows={3}
                          placeholder='{"Authorization":"Bearer …"}'
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                      <Form.Item label="请求体（GET/HEAD 忽略）" name="body">
                        <Input.TextArea
                          rows={4}
                          placeholder="原始正文，如 JSON"
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                      <Form.Item
                        label="入参 JSON Schema"
                        name="input_schema_text"
                        tooltip="新建留空不写库；编辑留空表示不改已存值"
                      >
                        <Input.TextArea
                          rows={4}
                          placeholder='{"type":"object",...}'
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                      <Form.Item label="出参 JSON Schema" name="output_schema_text">
                        <Input.TextArea
                          rows={4}
                          placeholder='{"type":"object",...}'
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                    </>
                  ),
                },
              ]}
            />
          </Form>
        )}
        {externalDrawer?.kind === 'mcp' && (
          <Form<McpToolFormValues>
            form={mcpForm}
            layout="vertical"
            onFinish={(v) => void onMcpSubmit(v)}
          >
            <Form.Item
              label="注册名（逻辑名）"
              name="name"
              rules={
                externalDrawer.mode === 'create'
                  ? [
                      { required: true, message: '请输入注册名' },
                      {
                        pattern: /^[a-zA-Z][a-zA-Z0-9_]*$/,
                        message: '以字母开头，仅字母数字下划线',
                      },
                    ]
                  : []
              }
            >
              <Input
                placeholder="例如 my_mcp_tool"
                disabled={externalDrawer.mode === 'edit'}
                autoComplete="off"
              />
            </Form.Item>
            <Form.Item
              label="说明（给模型看的描述）"
              name="description"
              rules={[{ required: true, message: '请输入说明' }]}
            >
              <Input.TextArea rows={2} placeholder="何时应调用此工具" />
            </Form.Item>
            <Form.Item
              label="MCP 服务端 URL"
              name="server_url"
              rules={[{ required: true, message: '请输入 MCP 服务端 URL' }]}
            >
              <Input placeholder="Streamable：…/mcp 或 …/message；SSE：…/sse（以文档为准）" />
            </Form.Item>
            <Form.Item
              label="传输类型"
              name="transport_type"
              rules={[{ required: true }]}
              tooltip="须与 URL 匹配：Streamable HTTP 与 SSE 为不同入口；探测会按此项选择握手方式。"
            >
              <Select options={MCP_TRANSPORT_OPTIONS} />
            </Form.Item>
            <Form.Item
              label="调度模式（先 list 再 call，适合多远端工具）"
              name="dispatch_mode"
              valuePropName="checked"
              tooltip="开启后由模型按 step=list_tools / call_tool 使用同一注册名；不必填远端工具名。"
            >
              <Switch checkedChildren="开" unCheckedChildren="关" />
            </Form.Item>
            <Form.Item noStyle dependencies={['dispatch_mode']}>
              {({ getFieldValue }) => {
                const dispatch = Boolean(getFieldValue('dispatch_mode'))
                return (
                  <Form.Item
                    label="远端工具名（直连 tools/call 的 name）"
                    name="mcp_tool_name"
                    hidden={dispatch}
                    rules={
                      dispatch
                        ? []
                        : [{ required: true, message: '请填写远端 tools/list 中的 name' }]
                    }
                    tooltip="与远端 tools/call 的 name 一致，例如 search_docs_by_lang_chain"
                  >
                    <Input placeholder="例如 search_docs_by_lang_chain" autoComplete="off" />
                  </Form.Item>
                )
              }}
            </Form.Item>
            <Form.Item label="探测 tools/list">
              <Space wrap>
                <Button
                  type="default"
                  loading={mcpProbeLoading}
                  onClick={() => void runMcpProbeListTools()}
                >
                  测试 tools/list
                </Button>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  不落库；会带上高级里的协议版本与连接配置
                </Typography.Text>
              </Space>
            </Form.Item>
            <Form.Item label="启用" name="enabled" valuePropName="checked">
              <Switch checkedChildren="开" unCheckedChildren="关" />
            </Form.Item>
            <Collapse
              bordered={false}
              style={{ background: 'transparent' }}
              items={[
                {
                  key: 'mcp-advanced',
                  label: '高级（可选）：协议版本、鉴权头、其它 tools_config',
                  children: (
                    <>
                      <Form.Item
                        label="MCP 协议版本 protocol_version"
                        name="protocol_version"
                        tooltip="如 LangChain 文档站常用 2025-06-18；留空走后端默认"
                      >
                        <Input placeholder="2025-06-18" autoComplete="off" />
                      </Form.Item>
                      <Form.Item
                        label="连接扩展 connection_config（JSON）"
                        name="connection_config_text"
                        tooltip="如 headers、verify_ssl、timeout_seconds；可与协议版本并存"
                      >
                        <Input.TextArea
                          rows={3}
                          placeholder='{"headers":{"Authorization":"Bearer …"}}'
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                      <Form.Item
                        label="其它 tools_config（JSON）"
                        name="tools_config_extras_text"
                        tooltip="除调度开关与直连远端名外的扩展键；一般留空"
                      >
                        <Input.TextArea
                          rows={3}
                          placeholder="{}"
                          style={{ fontFamily: 'monospace', fontSize: 12 }}
                        />
                      </Form.Item>
                    </>
                  ),
                },
              ]}
            />
          </Form>
        )}
      </Drawer>
      <Modal
        title="添加工具"
        open={createToolModalOpen}
        okText="下一步"
        cancelText="取消"
        destroyOnClose
        onCancel={() => {
          setCreateToolModalOpen(false)
          createToolForm.resetFields()
        }}
        onOk={() => void submitCreateToolModal()}
      >
        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
          填写注册名并选择类型后，将在侧栏中继续填写 URL、说明等详情。
        </Paragraph>
        <Form form={createToolForm} layout="vertical">
          <Form.Item
            label="注册名（逻辑名）"
            name="name"
            rules={[
              { required: true, message: '请输入注册名' },
              {
                pattern: /^[a-zA-Z][a-zA-Z0-9_]*$/,
                message: '以字母开头，仅字母数字下划线',
              },
            ]}
          >
            <Input placeholder="例如 my_http_tool" autoComplete="off" />
          </Form.Item>
          <Form.Item label="工具类型" name="kind" rules={[{ required: true, message: '请选择类型' }]}>
            <Radio.Group>
              <Radio value="http">HTTP 工具</Radio>
              <Radio value="mcp">MCP 工具</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
      </Modal>
      <Modal
        title="tools/list 探测结果"
        open={mcpProbeOpen}
        onCancel={() => setMcpProbeOpen(false)}
        footer={null}
        width={720}
        destroyOnHidden
      >
        <Input.TextArea
          readOnly
          rows={18}
          value={mcpProbeText}
          style={{ fontFamily: 'monospace', fontSize: 12 }}
        />
      </Modal>
    </div>
  )
}

export function SystemMonitorPage() {
  const [tokenDays, setTokenDays] = useState(14)
  const [data, setData] = useState<MonitorDashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const tokenChartAreaRef = useRef<HTMLDivElement>(null)
  const [tokenChartPxHeight, setTokenChartPxHeight] = useState(320)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const d = await fetchMonitorDashboard(tokenDays)
      setData(d)
    } catch (e) {
      setData(null)
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [tokenDays])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    const handleRefresh = () => void load()
    window.addEventListener('app:refresh', handleRefresh)
    return () => {
      window.removeEventListener('app:refresh', handleRefresh)
    }
  }, [load])

  const tokenChartData = useMemo(
    () =>
      (data?.token_usage ?? []).map((p) => ({
        dateFull: p.date,
        user_tokens: p.user_tokens,
        assistant_tokens: p.assistant_tokens,
        total_tokens: p.total_tokens,
      })),
    [data?.token_usage],
  )

  useLayoutEffect(() => {
    const el = tokenChartAreaRef.current
    if (!el || tokenChartData.length === 0) return

    const measure = () => {
      const h = Math.round(el.getBoundingClientRect().height)
      if (h > 0) {
        setTokenChartPxHeight(Math.max(160, h))
      }
    }

    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    let raf2 = 0
    const raf1 = window.requestAnimationFrame(() => {
      raf2 = window.requestAnimationFrame(measure)
    })
    return () => {
      window.cancelAnimationFrame(raf1)
      window.cancelAnimationFrame(raf2)
      ro.disconnect()
    }
  }, [tokenChartData.length, tokenDays, loading, data])

  const infraTag = (c: InfraComponentStatus) => {
    if (c.status === 'ok') return <Tag color="success">正常</Tag>
    if (c.status === 'error') return <Tag color="error">异常</Tag>
    return <Tag color="default">未启用</Tag>
  }

  return (
    <div className="page system-monitor-page">
      {error ? (
        <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />
      ) : null}

      <div className="system-monitor-body">
        <div className="system-monitor-spin-fill">
          <Spin spinning={loading && !data}>
            <Row className="system-monitor-layout-row" gutter={[16, 16]}>
              <Col span={24}>
                <Card title="基础设施">
                  {!data?.infrastructure.length ? (
                    <Text type="secondary">暂无数据</Text>
                  ) : (
                    <Row gutter={[12, 12]}>
                      {data.infrastructure.map((c) => (
                        <Col xs={24} sm={12} lg={8} key={c.key}>
                          <div className="system-monitor-infra-cell">
                            <div className="system-monitor-infra-head">
                              <Text strong>{c.label}</Text>
                              {infraTag(c)}
                            </div>
                            {c.address?.trim() ? (
                              <Paragraph
                                type="secondary"
                                copyable={{ text: c.address.trim() }}
                                style={{
                                  marginTop: 8,
                                  marginBottom: 0,
                                  fontSize: 12,
                                  fontFamily:
                                    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                                  wordBreak: 'break-all',
                                  lineHeight: 1.45,
                                }}
                              >
                                {c.address.trim()}
                              </Paragraph>
                            ) : null}
                            {c.detail ? (
                              <Text
                                type="secondary"
                                style={{
                                  fontSize: 12,
                                  display: 'block',
                                  marginTop: c.address?.trim() ? 6 : 8,
                                }}
                                ellipsis
                              >
                                {c.detail}
                              </Text>
                            ) : null}
                          </div>
                        </Col>
                      ))}
                    </Row>
                  )}
                </Card>
              </Col>
              <Col span={24}>
                <Card
                  className="system-monitor-token-card"
                  title="Token 消耗统计"
                  extra={
                    <Segmented
                      size="small"
                      options={[
                        { label: '近 7 天', value: 7 },
                        { label: '近 14 天', value: 14 },
                        { label: '近 30 天', value: 30 },
                      ]}
                      value={tokenDays}
                      onChange={(v) => setTokenDays(v as number)}
                    />
                  }
                >
                  <div className="system-monitor-token-card-inner">
                    <Statistic
                      title={`最近 ${tokenDays} 天合计（用户 + 助手 tokens）`}
                      value={data?.token_usage_grand_total ?? 0}
                      style={{ marginBottom: 16, flexShrink: 0 }}
                    />
                    {tokenChartData.length === 0 ? (
                      <div className="system-monitor-token-chart-area system-monitor-token-chart-area--empty">
                        <Text type="secondary">暂无数据</Text>
                      </div>
                    ) : (
                      <div ref={tokenChartAreaRef} className="system-monitor-token-chart-area">
                        <ResponsiveContainer width="100%" height={tokenChartPxHeight}>
                            <BarChart
                              data={tokenChartData}
                              margin={{ top: 12, right: 8, left: 4, bottom: 8 }}
                              barCategoryGap="14%"
                            >
                              <CartesianGrid
                                strokeDasharray="4 4"
                                stroke="rgba(0, 0, 0, 0.06)"
                                vertical={false}
                              />
                              <XAxis
                                dataKey="dateFull"
                                tickFormatter={(v) => String(v).slice(5)}
                                tick={{ fontSize: 12, fill: '#64748b' }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={{ stroke: '#e5e7eb' }}
                              />
                              <YAxis
                                tick={{ fontSize: 12, fill: '#64748b' }}
                                axisLine={false}
                                tickLine={false}
                                allowDecimals={false}
                                width={48}
                              />
                              <RechartsTooltip
                                formatter={(value, name) => {
                                  const n = typeof value === 'number' ? value : Number(value)
                                  const safe = Number.isFinite(n) ? n : 0
                                  return [safe.toLocaleString(), String(name)]
                                }}
                                contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
                              />
                              <Legend
                                wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                                formatter={(value) => <span style={{ color: '#64748b' }}>{value}</span>}
                              />
                              <Bar
                                dataKey="user_tokens"
                                name="用户"
                                stackId="role"
                                fill="var(--color-token-chart-user)"
                                maxBarSize={56}
                              />
                              <Bar
                                dataKey="assistant_tokens"
                                name="助手"
                                stackId="role"
                                fill="var(--color-token-chart-assistant)"
                                radius={[6, 6, 0, 0]}
                                maxBarSize={56}
                              />
                            </BarChart>
                          </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                </Card>
              </Col>
            </Row>
          </Spin>
        </div>
      </div>
    </div>
  )
}

const TRACE_LIST_DEFAULT_PAGE_SIZE = 10

export function TraceMonitorPage() {
  const navigate = useNavigate()
  const [traceId, setTraceId] = useState('')
  const traceIdRef = useRef(traceId)
  traceIdRef.current = traceId
  const listMetaRef = useRef({ page: 1, pageSize: TRACE_LIST_DEFAULT_PAGE_SIZE })
  const [traceList, setTraceList] = useState<TraceListData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadTraceList = async (
    targetTraceId: string,
    nextPage?: number,
    nextPageSize?: number,
  ) => {
    const normalized = targetTraceId.trim()
    const page = nextPage ?? listMetaRef.current.page
    const pageSize = nextPageSize ?? listMetaRef.current.pageSize
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams()
      if (normalized) {
        params.set('trace_id', normalized)
      }
      params.set('page', String(page))
      params.set('page_size', String(pageSize))
      const response = await fetch(`/api/traces?${params.toString()}`, {
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) {
        throw new Error(`trace list request failed: ${response.status}`)
      }
      const payload = await readJsonBody<{ data?: TraceListData }>(response)
      if (!payload?.data) {
        throw new Error('trace list payload missing data')
      }
      setTraceList(payload.data)
      listMetaRef.current = {
        page: payload.data.meta.page,
        pageSize: payload.data.meta.page_size,
      }
    } catch {
      setTraceList(null)
      setError('链路查询失败，请检查后端服务或 trace_id。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadTraceList('', 1, TRACE_LIST_DEFAULT_PAGE_SIZE)

    const handleRefresh = () => {
      void loadTraceList(
        traceIdRef.current,
        listMetaRef.current.page,
        listMetaRef.current.pageSize,
      )
    }
    window.addEventListener('app:refresh', handleRefresh)
    return () => {
      window.removeEventListener('app:refresh', handleRefresh)
    }
  }, [])

  const columns: ColumnsType<TraceListItem> = useMemo(
    () => [
      { title: 'Trace Id', dataIndex: 'trace_id', width: 200, ellipsis: true },
      { title: '会话 ID', dataIndex: 'session_id', width: 140, ellipsis: true },
      { title: '任务 ID', dataIndex: 'agent_task_id', width: 140, ellipsis: true },
      { title: 'Trace Name', dataIndex: 'trace_name', width: 120, ellipsis: true },
      {
        title: '耗时',
        dataIndex: 'duration_ms',
        width: 96,
        render: (v: number) => <Text strong>{(v / 1000).toFixed(2)}s</Text>,
      },
      {
        title: '状态',
        dataIndex: 'status',
        width: 100,
        render: (value: TraceListItem['status']) => {
          const color =
            value === 'success' ? 'success' : value === 'failed' ? 'error' : 'processing'
          return <Tag color={color}>{value.toUpperCase()}</Tag>
        },
      },
      { title: '执行时间', dataIndex: 'executed_at', width: 168, ellipsis: true },
      {
        title: '操作',
        key: 'action',
        width: 64,
        fixed: 'right',
        render: (_, row) => (
          <Tooltip title="查看详情">
            <Button
              type="text"
              className="agent-list-row-action"
              icon={<EyeOutlined />}
              onClick={() => navigate(`/monitor/tracing/${row.trace_id}`)}
            />
          </Tooltip>
        ),
      },
    ],
    [navigate],
  )

  return (
    <div className="agent-list-page">
      <div className="agent-list-table-shell">
        <div className="agent-list-toolbar">
          <div className="agent-list-toolbar-left">
            <Text style={{ fontSize: 13, color: '#86909c' }}>运行列表</Text>
          </div>
          <div className="agent-list-search-wrap">
            <Input
              className="agent-list-search-input"
              prefix={<SearchOutlined className="agent-list-search-icon" />}
              placeholder="搜索 trace_id（回车）"
              allowClear
              value={traceId}
              onChange={(e) => setTraceId(e.target.value)}
              onPressEnter={() => void loadTraceList(traceId, 1, listMetaRef.current.pageSize)}
            />
          </div>
        </div>
        {error ? (
          <div style={{ padding: '0 16px 8px' }}>
            <Alert type="error" message={error} showIcon />
          </div>
        ) : null}
        {loading && !traceList ? (
          <div className="agent-list-table-wrap" style={{ padding: 24 }}>
            <Skeleton active paragraph={{ rows: 8 }} />
          </div>
        ) : (
          <div className="agent-list-table-wrap">
            <div className="agent-list-table-inner">
              <Table<TraceListItem>
                className="agent-list-table"
                rowKey="trace_id"
                size="small"
                loading={loading}
                pagination={false}
                scroll={{ x: 1120 }}
                columns={columns}
                dataSource={traceList?.items ?? []}
              />
            </div>
            {traceList?.meta ? (
              <div className="agent-list-pagination-bar">
                <Pagination
                  size="small"
                  current={traceList.meta.page}
                  pageSize={traceList.meta.page_size}
                  total={traceList.meta.total}
                  showSizeChanger
                  pageSizeOptions={[10, 20, 50, 100]}
                  showTotal={(t) => `共 ${t} 条`}
                  onChange={(p, ps) => void loadTraceList(traceId, p, ps)}
                />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}

export { TraceMonitorDetailPage } from './traceMonitorDetail'

