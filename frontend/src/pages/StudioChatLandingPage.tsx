import {
  AppstoreOutlined,
  CheckOutlined,
  DownOutlined,
  SendOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { Button, Dropdown, Empty, Input, Spin, Typography, message, notification } from 'antd'
import type { MenuProps } from 'antd'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { Link } from 'react-router-dom'

import { workspaceMarkdownComponents } from '../components/chat/WorkspaceMarkdownMermaid'
import {
  formatAgentStreamProgressLabel,
  streamProgressPhaseBelongsToShellAgent,
} from '../utils/formatAgentStreamProgress'
import { useWorkbench } from '../contexts/useWorkbench'
import {
  invokeAgent,
  invokeAgentStream,
  type AgentKind,
  type AgentProcessTraceStep,
  type AgentToolHistoryEntry,
  type KnowledgeInvokeCitation,
  type PromptEngineeringBody,
  type ResponseFormat,
  type ToolChoiceMode,
} from '../api/agentApi'
import { fetchAgentById, type AgentDetailOut } from '../api/agentsApi'

import './pages.css'
import './StudioChatLandingPage.css'

const { Text } = Typography

type ChatRole = 'user' | 'assistant'

type ChatLine = {
  role: ChatRole
  content: string
  at: string
  thinkingText?: string
  structured?: Record<string, unknown> | null
  processTrace?: AgentProcessTraceStep[]
  knowledgeCitations?: KnowledgeInvokeCitation[]
  toolHistory?: AgentToolHistoryEntry[]
}

type StudioInvokeSnapshot = {
  agentId: number
  agentKind: AgentKind
  workspaceNamespace: string
  systemModelId: string
  promptSystem: string
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
  toolChoiceMode: ToolChoiceMode
  forcedToolName: string
  stripThinking: boolean
  streamOutput: boolean
  includeToolMessagesInRaw: boolean
}

function chatLineTimestamp(): string {
  return new Date().toISOString()
}

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

function mergeInvokeFieldsFromConfigJson(
  raw: AgentDetailOut['config_json'],
  base: StudioInvokeSnapshot,
): StudioInvokeSnapshot {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) return base
  const o = raw as Record<string, unknown>
  const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
  const str = (v: unknown, d: string) => (typeof v === 'string' ? v : d)
  const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d)
  const next = { ...base }
  if ('temperature' in o) next.temperature = num(o.temperature, base.temperature)
  if ('max_tokens' in o) next.maxTokens = Math.round(num(o.max_tokens, base.maxTokens))
  if ('top_p' in o) next.topP = num(o.top_p, base.topP)
  if ('stop_words' in o) next.stopWords = str(o.stop_words, base.stopWords)
  if ('seed' in o) {
    const v = o.seed
    if (v === null) next.seed = null
    else if (typeof v === 'number' && Number.isFinite(v)) next.seed = Math.round(v)
  }
  if ('frequency_penalty' in o) {
    const v = o.frequency_penalty
    if (v === null || v === undefined) next.frequencyPenalty = 0
    else if (typeof v === 'number' && Number.isFinite(v)) next.frequencyPenalty = v
  }
  if ('presence_penalty' in o) {
    const v = o.presence_penalty
    if (v === null || v === undefined) next.presencePenalty = 0
    else if (typeof v === 'number' && Number.isFinite(v)) next.presencePenalty = v
  }
  if ('response_format' in o) {
    const rf = o.response_format
    if (rf === 'text' || rf === 'json_object' || rf === 'json_schema') {
      next.responseFormat = rf
    }
  }
  if ('tool_names' in o && Array.isArray(o.tool_names) && o.tool_names.every((x) => typeof x === 'string')) {
    next.toolNames = o.tool_names as string[]
  }
  if ('use_tool_choice' in o) next.useToolChoice = bool(o.use_tool_choice, next.useToolChoice)
  if ('tool_choice_mode' in o) {
    const m = o.tool_choice_mode
    if (m === 'auto' || m === 'none' || m === 'required' || m === 'specific') {
      next.toolChoiceMode = m
    }
  }
  if ('forced_tool_name' in o) next.forcedToolName = str(o.forced_tool_name, next.forcedToolName)
  if ('strip_thinking' in o) next.stripThinking = bool(o.strip_thinking, next.stripThinking)
  if ('stream_output' in o) next.streamOutput = bool(o.stream_output, next.streamOutput)
  if ('include_tool_messages_in_raw' in o) {
    next.includeToolMessagesInRaw = bool(o.include_tool_messages_in_raw, next.includeToolMessagesInRaw)
  }
  if ('prompt_system_prompt' in o) {
    next.promptSystem = str(o.prompt_system_prompt, next.promptSystem)
  } else if ('prompt_system' in o || 'prompt_developer' in o || 'prompt_auxiliary' in o) {
    let sys = str(o.prompt_system, '')
    const dev = str(o.prompt_developer, '')
    const aux = str(o.prompt_auxiliary, '')
    if (dev) sys = sys ? `${sys}\n\n## 能力（补充）\n${dev}` : `## 能力（补充）\n${dev}`
    if (aux) sys = sys ? `${sys}\n\n## 约束与输出（补充）\n${aux}` : `## 约束与输出（补充）\n${aux}`
    next.promptSystem = sys
  }
  return next
}

function buildStudioInvokeFromDetail(detail: AgentDetailOut): StudioInvokeSnapshot {
  const base: StudioInvokeSnapshot = {
    agentId: detail.id,
    agentKind: detail.agent_kind,
    workspaceNamespace: detail.workspace_namespace?.trim() || 'default',
    systemModelId: detail.sys_model_id != null ? String(detail.sys_model_id) : '',
    promptSystem: '',
    temperature: 0.7,
    maxTokens: 4096,
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
    streamOutput: true,
    includeToolMessagesInRaw: false,
  }
  let merged = mergeInvokeFieldsFromConfigJson(detail.config_json, base)
  const col = detail.system_prompt?.trim()
  if (col) merged = { ...merged, promptSystem: col }
  merged = { ...merged, streamOutput: true }
  return merged
}

function splitAssistantThinkingAndBody(raw: string): { thinking: string; body: string } {
  const blockRe =
    /<think>[\s\S]*?<\/redacted_thinking>|<thinking>[\s\S]*?<\/thinking>/gi
  const innerRe =
    /<think>([\s\S]*?)<\/redacted_thinking>|<thinking>([\s\S]*?)<\/thinking>/gi
  const parts: string[] = []
  let m: RegExpExecArray | null
  while ((m = innerRe.exec(raw)) !== null) {
    const inner = (m[1] ?? m[2] ?? '').trim()
    if (inner) parts.push(inner)
  }
  const body = raw.replace(blockRe, '').replace(/\n{3,}/g, '\n\n').trim()
  return { thinking: parts.join('\n\n'), body }
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

function AssistantBubbleMarkdown({
  text,
  thinkingText: thinkingFromApi,
}: {
  text: string
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

function namespaceLabel(ns: string): string {
  const t = ns.trim()
  return t || 'default'
}

export function StudioChatLandingPage() {
  const { workbenchAgents, workbenchAgentId, loading: listLoading, selectWorkbenchAgent } = useWorkbench()
  const [selectedNs, setSelectedNs] = useState<string | null>(null)
  const [invokeSnap, setInvokeSnap] = useState<StudioInvokeSnapshot | null>(null)
  const [configLoading, setConfigLoading] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)

  const [hasThread, setHasThread] = useState(false)
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ChatLine[]>([])
  const [conversationSessionId, setConversationSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [streamExploringText, setStreamExploringText] = useState('')
  const [streamExploringPhase, setStreamExploringPhase] = useState('')

  const invokeAbortRef = useRef<AbortController | null>(null)
  const chatScrollRef = useRef<HTMLDivElement>(null)
  const exploringScrollRef = useRef<HTMLDivElement>(null)
  const chatStickBottomRef = useRef(true)
  const streamFlushRafRef = useRef<number | null>(null)
  const streamPendingStreamRef = useRef<{ content: string; exploring: string } | null>(null)

  const sortedNamespaces = useMemo(() => {
    const rows = workbenchAgents.map((a) => ({
      id: a.id,
      ns: (a.workspace_namespace ?? 'default').trim() || 'default',
    }))
    rows.sort((a, b) => a.ns.localeCompare(b.ns, 'zh-Hans-CN'))
    return rows
  }, [workbenchAgents])

  useEffect(() => {
    if (listLoading || sortedNamespaces.length === 0 || selectedNs != null) return
    const preferred =
      workbenchAgentId != null ? sortedNamespaces.find((r) => r.id === workbenchAgentId) : undefined
    const pick = preferred ?? sortedNamespaces[0]
    setSelectedNs(pick.ns)
  }, [listLoading, sortedNamespaces, workbenchAgentId, selectedNs])

  useEffect(() => {
    if (selectedNs == null) return
    const row = sortedNamespaces.find((r) => r.ns === selectedNs)
    if (!row) {
      setInvokeSnap(null)
      setConfigError('未找到该命名空间对应的工作区')
      return
    }
    let cancelled = false
    setConfigLoading(true)
    setConfigError(null)
    void fetchAgentById(row.id)
      .then((detail) => {
        if (cancelled) return
        setInvokeSnap(buildStudioInvokeFromDetail(detail))
        selectWorkbenchAgent(row.id)
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setInvokeSnap(null)
        setConfigError(e instanceof Error ? e.message : '加载工作区配置失败')
      })
      .finally(() => {
        if (!cancelled) setConfigLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedNs, sortedNamespaces, selectWorkbenchAgent])

  const handleChatScroll = useCallback(() => {
    const el = chatScrollRef.current
    if (!el) return
    const { scrollTop, scrollHeight, clientHeight } = el
    chatStickBottomRef.current = scrollHeight - scrollTop - clientHeight <= 80
  }, [])

  useLayoutEffect(() => {
    const el = chatScrollRef.current
    if (!el || !chatStickBottomRef.current) return
    el.scrollTop = el.scrollHeight
  }, [messages, loading, hasThread])

  const sendMessage = useCallback(async () => {
    const text = draft.trim()
    if (!text || loading || !invokeSnap) return
    if (!invokeSnap.systemModelId) {
      void message.warning('该工作区未绑定系统模型，请先到控制台配置')
      return
    }
    if (invokeSnap.agentKind !== 'workbench' && !invokeSnap.promptSystem.trim()) {
      void message.warning('系统提示词为空，请先到控制台配置')
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
    setHasThread(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text, at: userAt },
      ...(invokeSnap.streamOutput ? [{ role: 'assistant' as const, content: '', at: assistantAt }] : []),
    ])
    setLoading(true)

    try {
      const promptsPayload = buildPromptsPayload(invokeSnap.promptSystem)
      const stopList = parseStopWordsForApi(invokeSnap.stopWords)
      const responseBlock: { response_format: ResponseFormat } = { response_format: invokeSnap.responseFormat }

      const invokeBody = {
        agent_kind: invokeSnap.agentKind,
        user_message: text,
        hyperparameters: {
          temperature: invokeSnap.temperature,
          max_tokens: invokeSnap.maxTokens,
          top_p: invokeSnap.topP,
          ...(stopList ? { stop: stopList } : {}),
          ...(invokeSnap.seed != null ? { seed: invokeSnap.seed } : {}),
          frequency_penalty: invokeSnap.frequencyPenalty,
          presence_penalty: invokeSnap.presencePenalty,
        },
        response: responseBlock,
        tool_choice: invokeSnap.useToolChoice
          ? {
              mode: invokeSnap.toolChoiceMode,
              forced_tool_name:
                invokeSnap.toolChoiceMode === 'specific' ? invokeSnap.forcedToolName.trim() || null : null,
            }
          : null,
        output: {
          strip_thinking_blocks: invokeSnap.stripThinking,
          include_tool_messages_in_raw: invokeSnap.includeToolMessagesInRaw,
        },
        config_id: invokeSnap.systemModelId,
        tool_names: invokeSnap.toolNames.length ? invokeSnap.toolNames : undefined,
        ...(promptsPayload ? { prompts: promptsPayload } : {}),
        agent_id: invokeSnap.agentId,
        ...(conversationSessionId ? { conversation_session_id: conversationSessionId } : {}),
        workspace_namespace: invokeSnap.workspaceNamespace.trim() || 'default',
      }

      const data = invokeSnap.streamOutput
        ? await invokeAgentStream(invokeBody, {
            signal: ac.signal,
            onStart: (ev) => {
              if (ev.conversation_session_id) {
                setConversationSessionId(ev.conversation_session_id)
              }
            },
            onProgress: (ev) => {
              const label = formatAgentStreamProgressLabel(ev).trim()
              if (label && streamProgressPhaseBelongsToShellAgent(ev, invokeSnap.agentId)) {
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
                    m.role === 'assistant' && m.at === assistantAt ? { ...m, content: pair.content } : m,
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
      const kc = data.knowledge_citations?.length ? data.knowledge_citations : undefined

      setMessages((prev) => {
        if (invokeSnap.streamOutput) {
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
          if (invokeSnap.streamOutput) {
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
        key: 'studio-chat-invoke-error',
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
  }, [conversationSessionId, draft, invokeSnap, loading])

  const onPickNs = (ns: string) => {
    if (loading) return
    setSelectedNs(ns)
    setConversationSessionId(null)
    setMessages([])
    setHasThread(false)
    setDraft('')
    setStreamExploringPhase('')
    setStreamExploringText('')
  }

  const workspaceMenuItems: MenuProps['items'] = useMemo(
    () =>
      sortedNamespaces.map(({ ns }) => {
        const label = namespaceLabel(ns)
        const active = selectedNs === ns
        return {
          key: ns,
          label: (
            <span className="studio-chat-landing__workspace-menu-item">
              <span className="studio-chat-landing__workspace-menu-item-text">{label}</span>
              {active ? <CheckOutlined className="studio-chat-landing__workspace-menu-check" /> : null}
            </span>
          ),
        }
      }),
    [sortedNamespaces, selectedNs],
  )

  const composerDisabled = configLoading || !invokeSnap || Boolean(configError)
  const showExploring = Boolean(loading && invokeSnap?.streamOutput)

  /** 编排进度区有 max-height + overflow，流式更新时始终滚到最新一行 */
  useLayoutEffect(() => {
    if (!showExploring) return
    const el = exploringScrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [streamExploringText, streamExploringPhase, showExploring])

  const renderComposer = (variant: 'hero' | 'dock') => (
    <div className="studio-chat-landing__composer-shell">
      <Input.TextArea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="输入消息，开始与编排助手对话…"
        autoSize={{ minRows: variant === 'hero' ? 4 : 2, maxRows: variant === 'hero' ? 12 : 8 }}
        disabled={composerDisabled}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            void sendMessage()
          }
        }}
      />
      <div className="studio-chat-landing__composer-foot">
        <div className="studio-chat-landing__composer-foot-left">
          <Dropdown
            menu={{
              selectable: true,
              selectedKeys: selectedNs ? [selectedNs] : [],
              items: workspaceMenuItems,
              onClick: ({ key }) => onPickNs(String(key)),
              style: { maxHeight: 320, overflow: 'auto' },
            }}
            trigger={['click']}
            disabled={loading || composerDisabled}
            placement="topLeft"
          >
            <Button
              type="default"
              className="studio-chat-landing__workspace-dropdown-trigger"
              loading={configLoading}
              disabled={loading || composerDisabled}
              aria-label="选择命名空间"
              aria-haspopup="menu"
            >
              <span className="studio-chat-landing__workspace-dropdown-inner">
                <span className="studio-chat-landing__workspace-dropdown-value">
                  {namespaceLabel(selectedNs ?? 'default')}
                </span>
                <DownOutlined className="studio-chat-landing__workspace-dropdown-caret" />
              </span>
            </Button>
          </Dropdown>
        </div>
        <Button
          type="primary"
          shape="circle"
          icon={<SendOutlined />}
          className="studio-chat-landing__send-btn"
          loading={loading}
          disabled={composerDisabled || !draft.trim()}
          onClick={() => void sendMessage()}
          aria-label="发送"
        />
      </div>
    </div>
  )

  return (
    <div className={`studio-chat-landing${hasThread ? ' studio-chat-landing--has-thread' : ''}`}>
      <div className="studio-chat-landing__float">
        <Link to="/pixel-office">
          <Button type="primary" icon={<AppstoreOutlined />}>
            像素工作室
          </Button>
        </Link>
        <Link to="/workbench">
          <Button type="default" icon={<SettingOutlined />}>
            控制台
          </Button>
        </Link>
      </div>

      <div className="studio-chat-landing__main">
        {listLoading ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Spin size="large" />
          </div>
        ) : sortedNamespaces.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Empty description="暂无可用命名空间，请先在控制台创建工作区">
              <Link to="/workbench">
                <Button type="primary">前往控制台</Button>
              </Link>
            </Empty>
          </div>
        ) : (
          <>
            <div className="studio-chat-landing__hero">
              <div className="studio-chat-landing__brand">
                <div className="studio-chat-landing__brand-mark" aria-hidden>
                  ✦
                </div>
                <div className="studio-chat-landing__brand-title">工作区对话</div>
                <div className="studio-chat-landing__brand-sub">在左下角选择命名空间并输入消息</div>
              </div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', position: 'relative' }}>
                {configLoading ? (
                  <div className="studio-chat-landing__config-overlay">
                    <Spin />
                  </div>
                ) : null}
                {renderComposer('hero')}
              </div>
              {configError ? (
                <Text type="danger" style={{ textAlign: 'center' }}>
                  {configError}
                </Text>
              ) : null}
            </div>

            <div className="studio-chat-landing__thread">
              <div
                ref={chatScrollRef}
                className="studio-chat-landing__scroll"
                onScroll={handleChatScroll}
              >
                {showExploring ? (
                  <div
                    ref={exploringScrollRef}
                    className="studio-chat-landing__exploring-float"
                    aria-live="polite"
                  >
                    {streamExploringPhase.trim() ? (
                      <div className="studio-chat-landing__exploring-float-phase" aria-label="流式流程状态">
                        {streamExploringPhase.trim()}
                      </div>
                    ) : (
                      <div
                        className="studio-chat-landing__exploring-float-phase studio-chat-landing__exploring-float-phase--muted"
                        aria-label="流式流程状态"
                      >
                        等待流式阶段…
                      </div>
                    )}
                    {streamExploringText.trim() ? (
                      <div className="studio-chat-landing__exploring-float-body">
                        {streamExploringText}
                      </div>
                    ) : null}
                  </div>
                ) : null}
                {messages.map((m, i) => (
                  <div
                    key={`${m.at}-${m.role}-${i}`}
                    className={`studio-chat-landing__msg-row studio-chat-landing__msg-row--${m.role}`}
                  >
                    <div
                      className={`studio-chat-landing__bubble studio-chat-landing__bubble--${m.role === 'user' ? 'user' : 'assistant'}`}
                    >
                      {m.role === 'assistant' &&
                      loading &&
                      invokeSnap?.streamOutput &&
                      !m.content.trim() ? (
                        <div className="studio-chat-landing__wait" aria-busy="true">
                          <Spin size="small" />
                          <span>生成中…</span>
                        </div>
                      ) : m.role === 'assistant' ? (
                        <AssistantBubbleMarkdown text={m.content} thinkingText={m.thinkingText} />
                      ) : (
                        <ChatMarkdownBody text={m.content} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="studio-chat-landing__dock-wrap">{renderComposer('dock')}</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
