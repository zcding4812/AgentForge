import { ClearOutlined, HistoryOutlined, SwapOutlined } from '@ant-design/icons'
import {
  Button,
  Card,
  Drawer,
  Empty,
  Input,
  List,
  Modal,
  Pagination,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { workspaceMarkdownComponents } from '../components/chat/WorkspaceMarkdownMermaid'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import {
  invokeAgent,
  invokeAgentStream,
  type AgentInvokeData,
  type AgentStreamProgressEvent,
} from '../api/agentApi'
import {
  createConversationSession,
  getConversationSessionDetail,
  listConversationSessions,
  type ConversationMessageOut,
  type ConversationSessionOut,
} from '../api/conversationApi'
import { fetchAgentById, fetchAgentsList, type AgentDetailOut, type AgentOut } from '../api/agentsApi'
import { fetchWorkspaceNamespaces } from '../api/workspaceNamespacesApi'
import {
  buildAgentInvokeBodyFromDetail,
  parseInvokeWorkspaceFromAgentDetail,
} from '../utils/buildAgentInvokeBodyFromDetail'
import {
  formatAgentStreamProgressLabel,
  streamProgressPhaseBelongsToShellAgent,
} from '../utils/formatAgentStreamProgress'
import { applyPixelStreamProgress } from '../utils/pixelStreamBridge'
import { getRealtimeWebSocketUrl } from '../utils/realtimeWebSocketUrl'

import './pages.css'
import './pixelAgentsShell.css'

const { Text } = Typography

const PIXEL_HISTORY_PAGE_SIZE = 15

/** 本轮完成后不写库再拉全量：在本地追加 user/assistant，避免 loading 闪烁与错误 messages_agent_id 过滤导致「对话没了」 */
function appendSyntheticRoundToMessages(
  prev: ConversationMessageOut[],
  params: {
    sessionId: string
    agentEntityId: number
    userText: string
    assistantText: string
    tokens?: number | null
    thinkingText?: string | null
    processTrace?: AgentInvokeData['process_trace']
    toolHistory?: AgentInvokeData['tool_history']
  },
): ConversationMessageOut[] {
  const maxId = prev.length === 0 ? 0 : Math.max(...prev.map((m) => m.id))
  let nextId = maxId + 1
  const userId = nextId++
  const asstId = nextId++
  const iso = new Date().toISOString()
  const meta: Record<string, unknown> = {}
  if (params.thinkingText) meta.thinking_text = params.thinkingText
  if (params.processTrace?.length) meta.process_trace = params.processTrace
  if (params.toolHistory?.length) meta.tool_history = params.toolHistory

  const userRow: ConversationMessageOut = {
    id: userId,
    session_id: params.sessionId,
    agent_id: params.agentEntityId,
    role: 'user',
    content: params.userText,
    content_type: 'text',
    created_at: iso,
    metadata: null,
    turn_index: 0,
    tokens: null,
  }
  const asstRow: ConversationMessageOut = {
    id: asstId,
    session_id: params.sessionId,
    agent_id: params.agentEntityId,
    role: 'assistant',
    content: params.assistantText,
    content_type: 'text',
    created_at: iso,
    metadata: Object.keys(meta).length ? meta : null,
    turn_index: 0,
    reply_message_id: userId,
    tokens: params.tokens ?? null,
  }
  return [...prev, userRow, asstRow]
}

function postToPixel(win: Window | null | undefined, data: unknown) {
  if (!win) return
  win.postMessage(data, '*')
}

/** 主包 `settingsLoaded`：默认始终显示小人头顶名称（否则仅选中时显示） */
function postPixelAlwaysShowLabels(win: Window | null | undefined) {
  postToPixel(win, { type: 'settingsLoaded', alwaysShowLabels: true })
}

/** 自 iframe 上行的、可转发到 /api/realtime 画布主题的消息类型 */
const CANVAS_TO_REALTIME_MSG_TYPES = new Set(['ai-agents-canvas', 'pixelAgentAction'])

/**
 * 全屏嵌入上游 pixel-agents 构建的 webview（/pixel-native），
 * 通过 postMessage 驱动工具条动画并与后端 Agent 对话。
 */
export function PixelAgentsShell() {
  const [searchParams, setSearchParams] = useSearchParams()
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [namespace, setNamespace] = useState('default')
  const [agents, setAgents] = useState<AgentOut[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<AgentDetailOut | null>(null)
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const chatLogRef = useRef<HTMLDivElement>(null)
  const didAutoSelectWorkbench = useRef(false)
  /** 流式：与 Console 页一致，侧栏实时显示累计正文（结束后再用接口历史对齐） */
  const [streamPreview, setStreamPreview] = useState<{ user: string; assistant: string } | null>(null)
  /** 工作台等：`delta.lane=exploring` 的编排过程，固定显示在消息区顶部 */
  const [streamExploringText, setStreamExploringText] = useState('')
  const [streamExploringPhase, setStreamExploringPhase] = useState('')
  const streamPendingStreamRef = useRef<{ content: string; exploring: string } | null>(null)
  const streamExploringScrollRef = useRef<HTMLPreElement>(null)
  const streamFlushRafRef = useRef<number | null>(null)
  const pixelRealtimeWsRef = useRef<WebSocket | null>(null)
  /** 子 Agent 执行中：结束时对该 id 再发一次 agentToolsClear，避免工具条残留 */
  const subAgentActiveIdRef = useRef<number | null>(null)
  /** 消息加载世代：切换 Agent 或竞态时丢弃过期的 getConversationSessionDetail 结果 */
  const messageLoadGenRef = useRef(0)
  /** 当前侧栏展示的会话元信息（首屏取该 Agent/工作台 owner 下最近更新的一条；可从历史抽屉切换或新建） */
  const [viewSession, setViewSession] = useState<Pick<
    ConversationSessionOut,
    'session_id' | 'title' | 'updated_at'
  > | null>(null)
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historySessions, setHistorySessions] = useState<ConversationSessionOut[]>([])
  const [historyPage, setHistoryPage] = useState(1)
  const [historyTotal, setHistoryTotal] = useState(0)
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)

  const resolvedNs = namespace.trim() || 'default'

  const namespaceFromQuery = searchParams.get('namespace')
  useEffect(() => {
    const q = namespaceFromQuery?.trim()
    if (q) setNamespace(q)
  }, [namespaceFromQuery])

  /** 与后端 /api/realtime/ws 建立连接，并订阅当前命名空间画布主题（可接收他端 publish；本页上行见 message 监听） */
  useEffect(() => {
    const topic = `pixel:namespace:${resolvedNs}`
    const url = getRealtimeWebSocketUrl()
    const ws = new WebSocket(url)
    pixelRealtimeWsRef.current = ws
    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({ type: 'subscribe', topics: [topic] }))
      } catch {
        /* ignore */
      }
    }
    return () => {
      pixelRealtimeWsRef.current = null
      try {
        ws.close()
      } catch {
        /* ignore */
      }
    }
  }, [resolvedNs])

  /** 画布（iframe）经 postMessage 上行，转发到 WebSocket 主题，以便其它订阅者或后续服务端逻辑消费 */
  useEffect(() => {
    const onMessage = (ev: MessageEvent) => {
      if (ev.source !== iframeRef.current?.contentWindow) return
      if (!ev.data || typeof ev.data !== 'object') return
      const t = (ev.data as { type?: unknown }).type
      if (typeof t !== 'string' || !CANVAS_TO_REALTIME_MSG_TYPES.has(t)) return
      const sock = pixelRealtimeWsRef.current
      if (!sock || sock.readyState !== WebSocket.OPEN) return
      try {
        sock.send(
          JSON.stringify({
            type: 'publish',
            topic: `pixel:namespace:${resolvedNs}`,
            data: {
              client: 'pixel-shell',
              namespace: resolvedNs,
              envelope: ev.data,
            },
          }),
        )
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [resolvedNs])

  const forwardCanvas = useCallback(
    (win: Window | null | undefined, data: object) => {
      postToPixel(win, data)
      const s = pixelRealtimeWsRef.current
      if (!s || s.readyState !== WebSocket.OPEN) return
      try {
        s.send(
          JSON.stringify({
            type: 'publish',
            topic: `pixel:namespace:${resolvedNs}`,
            data: {
              client: 'pixel-shell',
              stream: 'invoke',
              canvas: data,
            },
          }),
        )
      } catch {
        /* ignore */
      }
    },
    [resolvedNs],
  )

  const iframeSrc = useMemo(() => {
    const q = new URLSearchParams()
    q.set('namespace', resolvedNs)
    /** 与 pixel-native 静态资源（含 view-mode.css）更新对齐，减轻 iframe 强缓存 */
    q.set('v', '10')
    return `/pixel-native/index.html?${q.toString()}`
  }, [resolvedNs])

  /** 下拉：与 Agent 中心一致，走 GET /api/workspace-namespaces（库内 WorkspaceNamespace 行） */
  const [namespaceOptions, setNamespaceOptions] = useState<string[]>(['default'])
  const [namespacesLoading, setNamespacesLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setNamespacesLoading(true)
    void fetchWorkspaceNamespaces()
      .then((data) => {
        if (cancelled) return
        const slugs = data.items
          .map((x) => x.slug.trim())
          .filter((s) => s.length > 0)
        setNamespaceOptions(
          slugs.length > 0 ? [...slugs].sort((a, b) => a.localeCompare(b)) : ['default'],
        )
      })
      .catch(() => {
        if (!cancelled) setNamespaceOptions(['default'])
      })
      .finally(() => {
        if (!cancelled) setNamespacesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const namespaceSelectOptions = useMemo(() => {
    const s = new Set(namespaceOptions)
    s.add(resolvedNs)
    return [...s]
      .sort((a, b) => a.localeCompare(b))
      .map((ns) => ({ value: ns, label: ns }))
  }, [namespaceOptions, resolvedNs])

  const onNamespaceChange = useCallback(
    (ns: string) => {
      setNamespace(ns)
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev)
          p.set('namespace', ns)
          return p
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const loadAgents = useCallback(async () => {
    try {
      const d = await fetchAgentsList({
        page: 1,
        page_size: 100,
        workspace_namespace: resolvedNs,
      })
      setAgents(d.items)
      if (selectedId != null && !d.items.some((a) => a.id === selectedId)) {
        setSelectedId(null)
        setDetail(null)
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载 Agent 列表失败')
    }
  }, [resolvedNs, selectedId])

  const [chatMessages, setChatMessages] = useState<ConversationMessageOut[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [sessionBootstrapLoading, setSessionBootstrapLoading] = useState(false)

  useEffect(() => {
    void loadAgents()
  }, [loadAgents])

  /** 命名空间切换时立刻丢弃旧 Agent 列表与侧栏会话，避免用上一命名空间的 workbench id 去 list/load 会话（会表现为历史空白或错会话）。 */
  useEffect(() => {
    messageLoadGenRef.current += 1
    didAutoSelectWorkbench.current = false
    setAgents([])
    setSelectedId(null)
    setDetail(null)
    setChatMessages([])
    setSessionId(null)
    setViewSession(null)
    setStreamPreview(null)
    setStreamExploringPhase('')
    setStreamExploringText('')
    setHistoryDrawerOpen(false)
    setHistoryLoading(false)
    setHistorySessions([])
    setHistoryPage(1)
    setHistoryTotal(0)
    setHistoryDetailLoading(false)
  }, [resolvedNs])

  const workbenchAgent = useMemo(() => agents.find((a) => a.agent_kind === 'workbench') ?? null, [agents])

  useEffect(() => {
    if (didAutoSelectWorkbench.current || selectedId != null) return
    if (!workbenchAgent) return
    setSelectedId(workbenchAgent.id)
    didAutoSelectWorkbench.current = true
  }, [workbenchAgent, selectedId])

  const historyListAgentId = useMemo(() => {
    if (workbenchAgent != null) return workbenchAgent.id
    return selectedId
  }, [workbenchAgent, selectedId])

  const loadMessages = useCallback(async (sessionId: string, gen: number) => {
    if (gen !== messageLoadGenRef.current) return
    setMessagesLoading(true)
    try {
      const d = await getConversationSessionDetail(sessionId, {
        page: 1,
        page_size: 80,
        ...(historyListAgentId != null && workbenchAgent == null
          ? { messages_agent_id: historyListAgentId }
          : {}),
      })
      if (gen !== messageLoadGenRef.current) return
      setChatMessages(d.messages)
      setViewSession({
        session_id: d.session.session_id,
        title: d.session.title,
        updated_at: d.session.updated_at,
      })
      setSessionId(d.session.session_id)
    } catch (e) {
      if (gen !== messageLoadGenRef.current) return
      message.error(e instanceof Error ? e.message : '加载对话消息失败')
      setChatMessages([])
      setViewSession(null)
    } finally {
      // 与「世代」丢弃结果解耦，避免过期请求结束时永远不关 loading，侧栏一直转圈、历史像被清空
      setMessagesLoading(false)
    }
  }, [historyListAgentId, workbenchAgent])

  const loadMessagesRef = useRef(loadMessages)
  loadMessagesRef.current = loadMessages

  const loadHistoryPage = useCallback(
    async (page: number) => {
      if (historyListAgentId == null) return
      setHistoryLoading(true)
      try {
        const data = await listConversationSessions({
          page,
          page_size: PIXEL_HISTORY_PAGE_SIZE,
          agent_id: historyListAgentId,
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
    [historyListAgentId],
  )

  const openHistoryDrawer = useCallback(() => {
    if (historyListAgentId == null) {
      void message.warning('请先选择要对话的 Agent')
      return
    }
    setHistoryDrawerOpen(true)
    void loadHistoryPage(1)
  }, [historyListAgentId, loadHistoryPage])

  const onSelectHistorySession = useCallback(async (session: ConversationSessionOut) => {
    const g = ++messageLoadGenRef.current
    setHistoryDetailLoading(true)
    try {
      setSessionId(session.session_id)
      setHistoryDrawerOpen(false)
      setStreamPreview(null)
      setStreamExploringPhase('')
      setStreamExploringText('')
      await loadMessagesRef.current(session.session_id, g)
    } finally {
      if (g === messageLoadGenRef.current) {
        setHistoryDetailLoading(false)
      }
    }
  }, [])

  const startNewConversation = useCallback(() => {
    if (loading) {
      void message.warning('本轮仍在进行，请稍后再新建对话')
      return
    }
    if (historyListAgentId == null) {
      void message.warning('请先选择要对话的 Agent')
      return
    }
    Modal.confirm({
      title: '新建对话',
      content: '将创建一个空会话；当前侧栏消息会切换为新会话（历史仍可在抽屉中找回）。是否继续？',
      okText: '新建',
      cancelText: '取消',
      onOk: async () => {
        const g = ++messageLoadGenRef.current
        setSessionBootstrapLoading(true)
        setStreamPreview(null)
        setStreamExploringPhase('')
        setStreamExploringText('')
        setChatMessages([])
        setSessionId(null)
        setViewSession(null)
        try {
          const created = await createConversationSession({
            agent_id: historyListAgentId,
            title: `像素办公室 · ${new Date().toLocaleString('zh-CN', {
              year: 'numeric',
              month: '2-digit',
              day: '2-digit',
              hour: '2-digit',
              minute: '2-digit',
            })}`,
          })
          if (g !== messageLoadGenRef.current) return
          setSessionId(created.session_id)
          await loadMessagesRef.current(created.session_id, g)
          void message.success('已创建新对话')
        } catch (e: unknown) {
          if (g !== messageLoadGenRef.current) return
          void message.error(e instanceof Error ? e.message : '创建会话失败')
          setSessionId(null)
          setChatMessages([])
          setViewSession(null)
        } finally {
          if (g === messageLoadGenRef.current) {
            setSessionBootstrapLoading(false)
          }
        }
      },
    })
  }, [historyListAgentId, loading])

  useEffect(() => {
    if (selectedId == null) {
      setDetail(null)
      if (!workbenchAgent) {
        setChatMessages([])
        setSessionId(null)
        setViewSession(null)
      }
      return
    }
    let c = false
    void fetchAgentById(selectedId)
      .then((d) => {
        if (!c) setDetail(d)
      })
      .catch(() => {
        if (!c) setDetail(null)
      })
    return () => {
      c = true
    }
  }, [selectedId, workbenchAgent])

  /** 命名空间内有工作台时：会话始终挂在该工作台 agent 上，切换下拉 Agent 不换 session */
  useEffect(() => {
    if (workbenchAgent == null) return
    const ownerId = workbenchAgent.id
    const g = ++messageLoadGenRef.current
    let cancelled = false
    setSessionBootstrapLoading(true)
    setSessionId(null)
    setChatMessages([])
    setViewSession(null)
    void listConversationSessions({ agent_id: ownerId, page: 1, page_size: 50 })
      .then(async (d) => {
        if (cancelled || g !== messageLoadGenRef.current) return
        const sorted = [...d.items].sort(
          (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
        )
        let sidToLoad: string | null = sorted[0]?.session_id ?? null
        if (!sidToLoad) {
          try {
            const created = await createConversationSession({
              agent_id: ownerId,
              title: '像素办公室',
            })
            if (cancelled || g !== messageLoadGenRef.current) return
            sidToLoad = created.session_id
          } catch (e) {
            if (!cancelled && g === messageLoadGenRef.current) {
              message.error(e instanceof Error ? e.message : '创建会话失败')
              setSessionId(null)
              setChatMessages([])
              setViewSession(null)
            }
            return
          }
        }
        setSessionId(sidToLoad)
        void loadMessagesRef.current(sidToLoad, g)
      })
      .catch((e) => {
        if (!cancelled && g === messageLoadGenRef.current) {
          message.error(e instanceof Error ? e.message : '加载会话失败')
          setSessionId(null)
          setChatMessages([])
          setViewSession(null)
        }
      })
      .finally(() => {
        if (!cancelled && g === messageLoadGenRef.current) {
          setSessionBootstrapLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [resolvedNs, workbenchAgent?.id])

  /** 无工作台时：仍按所选 Agent 各自最近会话 */
  useEffect(() => {
    if (workbenchAgent != null) return
    if (selectedId == null) return
    const g = ++messageLoadGenRef.current
    let cancelled = false
    setSessionBootstrapLoading(true)
    setSessionId(null)
    setChatMessages([])
    setViewSession(null)
    void listConversationSessions({ agent_id: selectedId, page: 1, page_size: 50 })
      .then((d) => {
        if (cancelled || g !== messageLoadGenRef.current) return
        const sorted = [...d.items].sort(
          (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
        )
        const latest = sorted[0]
        if (latest) {
          setSessionId(latest.session_id)
          void loadMessagesRef.current(latest.session_id, g)
        } else {
          setSessionId(null)
          setChatMessages([])
          setViewSession(null)
        }
      })
      .catch((e) => {
        if (!cancelled && g === messageLoadGenRef.current) {
          message.error(e instanceof Error ? e.message : '加载会话失败')
          setSessionId(null)
          setChatMessages([])
          setViewSession(null)
        }
      })
      .finally(() => {
        if (!cancelled && g === messageLoadGenRef.current) {
          setSessionBootstrapLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [selectedId, workbenchAgent])

  const sendToAgent = async () => {
    const text = draft.trim()
    if (!text || loading || !detail || selectedId == null) {
      if (!detail) message.warning('请选择 Agent 并等待详情加载')
      return
    }
    const getWin = () => iframeRef.current?.contentWindow
    const aid = selectedId
    let toolSeq = 0
    const mkToolId = () => `ai-${Date.now()}-${toolSeq++}`

    setLoading(true)
    subAgentActiveIdRef.current = null
    forwardCanvas(getWin() ?? undefined, { type: 'agentToolsClear', id: aid })
    // 主包 message 仅见 `id`+`status` 用于 setAgentActive / 气泡，勿发主包不消费的 actionHint、workPhase
    forwardCanvas(getWin() ?? undefined, { type: 'agentStatus', id: aid, status: 'active' })

    try {
      const ws = parseInvokeWorkspaceFromAgentDetail(detail)
      const body = buildAgentInvokeBodyFromDetail(
        detail,
        text,
        sessionId,
        ws,
        workbenchAgent != null && sessionId
          ? { conversationOwnerAgentId: workbenchAgent.id }
          : undefined,
      )
      const useStream = ws.streamOutput

      if (useStream) {
        streamPendingStreamRef.current = null
        setStreamExploringPhase('')
        setStreamExploringText('')
        setStreamPreview({ user: text, assistant: '' })
      }

      let data: AgentInvokeData
      if (useStream) {
        data = await invokeAgentStream(body, {
          onStart: (ev) => {
            if (ev.conversation_session_id) {
              setSessionId(ev.conversation_session_id)
            }
          },
          onProgress: (ev: AgentStreamProgressEvent) => {
            applyPixelStreamProgress(ev, (msg) => forwardCanvas(getWin() ?? undefined, msg), {
              fallbackAgentId: aid,
              canvasPixelToolAgentIdAllowlist: aid,
            })
            if (ev.stage === 'sub_agent' && ev.sub_phase && ev.active_agent_id != null) {
              subAgentActiveIdRef.current = ev.sub_phase === 'start' ? ev.active_agent_id : null
            }
            const label = formatAgentStreamProgressLabel(ev).trim()
            if (label && streamProgressPhaseBelongsToShellAgent(ev, aid)) {
              setStreamExploringPhase(label)
            }
          },
          onDelta: (contentAcc, exploringAcc) => {
            streamPendingStreamRef.current = { content: contentAcc, exploring: exploringAcc }
            if (streamFlushRafRef.current != null) return
            streamFlushRafRef.current = window.requestAnimationFrame(() => {
              streamFlushRafRef.current = null
              const pair = streamPendingStreamRef.current
              if (pair == null) return
              setStreamExploringText(pair.exploring)
              setStreamPreview((p) => (p ? { ...p, assistant: pair.content } : null))
            })
          },
        })
      } else {
        data = await invokeAgent(body)
        const tid = mkToolId()
        forwardCanvas(getWin() ?? undefined, {
          type: 'agentToolStart',
          id: aid,
          toolId: tid,
          toolName: 'Task',
          status: 'Generating…',
          runInBackground: true,
        })
        forwardCanvas(getWin() ?? undefined, { type: 'agentToolDone', id: aid, toolId: tid })
      }

      const lastSubId = subAgentActiveIdRef.current
      subAgentActiveIdRef.current = null
      forwardCanvas(getWin() ?? undefined, { type: 'agentToolsClear', id: aid })
      if (lastSubId != null) forwardCanvas(getWin() ?? undefined, { type: 'agentToolsClear', id: lastSubId })
      forwardCanvas(getWin() ?? undefined, { type: 'agentStatus', id: aid, status: 'waiting' })

      const sid = data.conversation_session_id ?? sessionId
      if (sid) {
        setSessionId(sid)
      }
      setStreamPreview(null)
      setStreamExploringPhase('')
      setStreamExploringText('')
      if (sid && selectedId != null) {
        const agentEntityId = selectedId
        setChatMessages((prev) =>
          appendSyntheticRoundToMessages(prev, {
            sessionId: sid,
            agentEntityId,
            userText: text,
            assistantText: data.assistant_text ?? '',
            tokens: data.tokens,
            thinkingText: data.thinking_text,
            processTrace: data.process_trace,
            toolHistory: data.tool_history,
          }),
        )
        const nowIso = new Date().toISOString()
        setViewSession((vs) =>
          vs?.session_id === sid
            ? { ...vs, updated_at: nowIso }
            : { session_id: sid, title: vs?.title?.trim() || '对话', updated_at: nowIso },
        )
      }

      message.success('本轮完成')
      setDraft('')
    } catch (e) {
      message.error(e instanceof Error ? e.message : '调用失败')
      const lastSubId = subAgentActiveIdRef.current
      subAgentActiveIdRef.current = null
      forwardCanvas(getWin() ?? undefined, { type: 'agentToolsClear', id: aid })
      if (lastSubId != null) forwardCanvas(getWin() ?? undefined, { type: 'agentToolsClear', id: lastSubId })
      forwardCanvas(getWin() ?? undefined, { type: 'agentStatus', id: aid, status: 'waiting' })
    } finally {
      if (streamFlushRafRef.current != null) {
        window.cancelAnimationFrame(streamFlushRafRef.current)
        streamFlushRafRef.current = null
      }
      setStreamPreview(null)
      setStreamExploringPhase('')
      setStreamExploringText('')
      setLoading(false)
    }
  }

  const agentOptions = useMemo(
    () =>
      agents.map((a) => ({
        value: a.id,
        label: `${a.name} (#${a.id})`,
      })),
    [agents],
  )

  /** 有工作台时：库内仍为同一会话，侧栏按当前所选 Agent 的 message.agent_id 过滤，便于切换下拉只看该角色对话 */
  const visibleChatMessages = useMemo(() => {
    if (workbenchAgent == null) return chatMessages
    if (selectedId == null) return []
    const ownerId = workbenchAgent.id
    return chatMessages.filter((m) => {
      const aid = m.agent_id ?? ownerId
      return aid === selectedId
    })
  }, [chatMessages, workbenchAgent, selectedId])

  useLayoutEffect(() => {
    const el = chatLogRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [
    visibleChatMessages,
    messagesLoading,
    sessionBootstrapLoading,
    streamPreview?.assistant,
    streamExploringText,
    loading,
  ])

  useLayoutEffect(() => {
    const el = streamExploringScrollRef.current
    if (!el || streamPreview == null) return
    el.scrollTop = el.scrollHeight
  }, [streamExploringText, streamPreview])

  const isChatEmpty = visibleChatMessages.length === 0 && !streamPreview
  const showEmptyScrollHint =
    isChatEmpty && !sessionBootstrapLoading && !messagesLoading

  const chatTopBarTitle = useMemo(() => {
    if (!viewSession) return '对话'
    const t = viewSession.title?.trim()
    return t || '未命名对话'
  }, [viewSession])

  const chatTopBarTooltip = useMemo(() => {
    if (!viewSession) {
      return workbenchAgent
        ? '本命名空间存在工作台：库内为同一会话；侧栏列表按当前下拉的 Agent 过滤展示。切换 Agent 可查看各角色在此会话中的消息。'
        : '选择 Agent 后会加载该成员下最近更新的一条会话；可用顶栏「历史」切换其它会话或「新建」开新会话。'
    }
    return viewSession.session_id
  }, [viewSession, workbenchAgent])

  return (
    <div className="pixel-agents-shell">
      <header className="pixel-agents-shell__bar">
        <div className="pixel-agents-shell__bar-row">
          <Space wrap align="center">
            <Select
              showSearch
              optionFilterProp="label"
              placeholder="工作空间"
              style={{ width: 220 }}
              loading={namespacesLoading}
              value={resolvedNs}
              options={namespaceSelectOptions}
              onChange={(v) => onNamespaceChange(v)}
            />
          </Space>
          <Link to="/studio">
            <Button type="primary" icon={<SwapOutlined />}>
              切换
            </Button>
          </Link>
        </div>
      </header>

      <div className="pixel-agents-shell__main">
        <iframe
          key={resolvedNs}
          ref={iframeRef}
          title="Pixel Agents"
          className="pixel-agents-shell__iframe"
          src={iframeSrc}
          onLoad={() => {
            const w = iframeRef.current?.contentWindow
            postPixelAlwaysShowLabels(w)
            window.setTimeout(() => postPixelAlwaysShowLabels(iframeRef.current?.contentWindow), 200)
          }}
        />

        <aside className="pixel-agents-shell__panel">
          <div className="pixel-agents-shell__panel-toolbar">
            <div className="pixel-agents-shell__member-toolbar">
              <span className="pixel-agents-shell__member-toolbar-label">对话题成员</span>
              <Select
                showSearch
                optionFilterProp="label"
                placeholder="选择要对话或查看消息的 Agent"
                className="pixel-agents-shell__agent-select"
                options={agentOptions}
                value={selectedId ?? undefined}
                onChange={(v) => setSelectedId(v ?? null)}
                allowClear
              />
            </div>
          </div>

          <div className="agent-workspace-page pixel-agents-shell__workspace">
            <Card
              className="chat-panel agent-workspace-chat-panel"
              styles={{ header: { display: 'none' }, body: { padding: 0 } }}
            >
              <div className="agent-workspace-chat-panel-inner">
                <div className="agent-workspace-chat-topbar agent-workspace-chat-topbar--with-title">
                  <div className="agent-workspace-chat-topbar__title-wrap">
                    <Tooltip title={chatTopBarTooltip} placement="bottomLeft">
                      <span className="agent-workspace-chat-topbar__title-text">{chatTopBarTitle}</span>
                    </Tooltip>
                  </div>
                  <div className="agent-workspace-chat-topbar__actions">
                    <Tooltip
                      title={
                        historyListAgentId == null
                          ? '请先选择要对话的 Agent'
                          : workbenchAgent
                            ? '查看该命名空间工作台的历史会话（库内会话挂在工作台上）'
                            : '查看该 Agent 的历史会话'
                      }
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={<HistoryOutlined />}
                        aria-label="历史对话"
                        className="agent-workspace-chat-history-btn"
                        disabled={
                          historyListAgentId == null || loading || sessionBootstrapLoading
                        }
                        onClick={openHistoryDrawer}
                      />
                    </Tooltip>
                    <Tooltip
                      title={
                        historyListAgentId == null
                          ? '请先选择要对话的 Agent'
                          : '新建空会话（二次确认）'
                      }
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={<ClearOutlined />}
                        aria-label="新建对话"
                        className="agent-workspace-chat-clear-btn"
                        disabled={
                          historyListAgentId == null || loading || sessionBootstrapLoading
                        }
                        onClick={startNewConversation}
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
                        {historyTotal > PIXEL_HISTORY_PAGE_SIZE ? (
                          <Pagination
                            className="agent-workspace-history-pagination"
                            size="small"
                            current={historyPage}
                            total={historyTotal}
                            pageSize={PIXEL_HISTORY_PAGE_SIZE}
                            showSizeChanger={false}
                            onChange={(p) => void loadHistoryPage(p)}
                          />
                        ) : null}
                      </>
                    )}
                  </Spin>
                </Drawer>

                <div
                  ref={chatLogRef}
                  className="agent-workspace-chat-messages-scroll"
                >
                  {streamPreview ? (
                    <div className="agent-workspace-stream-exploring" aria-live="polite">
                      <div className="agent-workspace-stream-exploring__head">
                        <span className="agent-workspace-stream-exploring__title">对话生成中</span>
                        {streamExploringPhase ? (
                          <Text
                            ellipsis={{ tooltip: streamExploringPhase }}
                            className="agent-workspace-stream-exploring__phase"
                          >
                            {streamExploringPhase}
                          </Text>
                        ) : (
                          <Text
                            type="secondary"
                            className="agent-workspace-stream-exploring__phase-placeholder"
                          >
                            编排与工具过程将显示于此…
                          </Text>
                        )}
                      </div>
                      <pre
                        ref={streamExploringScrollRef}
                        className="agent-workspace-stream-exploring__viewport"
                      >
                        {streamExploringText.trim() ? streamExploringText : '…'}
                      </pre>
                    </div>
                  ) : null}
                  <div
                    className={`agent-workspace-messages-inner${
                      showEmptyScrollHint ? ' agent-workspace-messages-inner--empty' : ''
                    }`}
                  >
                    {sessionBootstrapLoading || messagesLoading ? (
                      <div className="pixel-agents-shell__chat-loading">
                        <Spin size="small" />
                        <Text type="secondary">加载消息…</Text>
                      </div>
                    ) : null}
                    <div className="messages agent-workspace-messages">
                      {visibleChatMessages.map((m) => {
                        if (m.role === 'user' || m.role === 'assistant') {
                          return (
                            <div
                              key={m.id}
                              className={
                                m.role === 'user'
                                  ? 'agent-workspace-msg-row agent-workspace-msg-row--user'
                                  : 'agent-workspace-msg-row agent-workspace-msg-row--assistant'
                              }
                            >
                              <div
                                className={
                                  m.role === 'user'
                                    ? 'agent-workspace-msg-bubble agent-workspace-msg-bubble--user'
                                    : 'agent-workspace-msg-bubble agent-workspace-msg-bubble--assistant'
                                }
                              >
                                <div className="agent-workspace-msg__body agent-workspace-msg__body--md">
                                  {m.role === 'assistant' ? (
                                    <div className="agent-workspace-msg__md">
                                      <Markdown
                                        remarkPlugins={[remarkGfm]}
                                        components={workspaceMarkdownComponents}
                                      >
                                        {m.content}
                                      </Markdown>
                                    </div>
                                  ) : (
                                    m.content
                                  )}
                                </div>
                              </div>
                            </div>
                          )
                        }
                        return (
                          <div key={m.id} className="pixel-agents-shell__msg-system">
                            <Tag>系统</Tag>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {m.content}
                            </Text>
                            {m.created_at ? (
                              <Text type="secondary" style={{ fontSize: 11, marginLeft: 8 }}>
                                {new Date(m.created_at).toLocaleString()}
                              </Text>
                            ) : null}
                          </div>
                        )
                      })}
                      {streamPreview ? (
                        <>
                          <div className="agent-workspace-msg-row agent-workspace-msg-row--user">
                            <div className="agent-workspace-msg-bubble agent-workspace-msg-bubble--user">
                              <div className="agent-workspace-msg__body agent-workspace-msg__body--md">
                                {streamPreview.user}
                              </div>
                            </div>
                          </div>
                          <div className="agent-workspace-msg-row agent-workspace-msg-row--assistant">
                            <div className="agent-workspace-msg-bubble agent-workspace-msg-bubble--assistant">
                              <div className="agent-workspace-msg__body agent-workspace-msg__body--md">
                                {streamPreview.assistant ? (
                                  <div className="agent-workspace-msg__md">
                                    <Markdown
                                      remarkPlugins={[remarkGfm]}
                                      components={workspaceMarkdownComponents}
                                    >
                                      {streamPreview.assistant}
                                    </Markdown>
                                  </div>
                                ) : (
                                  <Text type="secondary">…</Text>
                                )}
                              </div>
                            </div>
                          </div>
                        </>
                      ) : null}
                      {loading && !streamPreview ? (
                        <div className="agent-workspace-msg-row agent-workspace-msg-row--assistant">
                          <div className="agent-workspace-msg-bubble agent-workspace-msg-bubble--assistant agent-workspace-msg-bubble--loading">
                            <div className="agent-workspace-msg__first-wait" aria-busy="true">
                              <Spin size="small" />
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                    {showEmptyScrollHint ? (
                      <p className="agent-workspace-chat-empty-hint">
                        {workbenchAgent != null &&
                        selectedId != null &&
                        chatMessages.length > 0
                          ? '当前所选 Agent 在此会话中暂无消息记录，可切换下拉查看其它成员或发送一条开始对话。'
                          : '输入消息开始对话...'}
                      </p>
                    ) : null}
                  </div>
                </div>

                <div className="agent-workspace-chat-footer">
                  <div className="agent-workspace-chat-composer">
                    <Input.TextArea
                      className="agent-workspace-chat-composer__textarea"
                      placeholder="输入消息，Enter 发送 · Shift+Enter 换行"
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onPressEnter={(e) => {
                        if (!e.shiftKey) {
                          e.preventDefault()
                          void sendToAgent()
                        }
                      }}
                      disabled={loading || selectedId == null}
                    />
                    <div className="agent-workspace-chat-footer__actions">
                      <Button
                        type="primary"
                        size="small"
                        className="agent-workspace-chat-btn-send"
                        loading={loading}
                        disabled={selectedId == null}
                        onClick={() => void sendToAgent()}
                      >
                        发送
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </aside>
      </div>
    </div>
  )
}
