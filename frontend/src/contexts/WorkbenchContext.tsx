import { createContext, useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'

import { fetchAgentsList, type AgentOut } from '../api/agentsApi'
import {
  readStoredWorkbenchAgentId,
  writeStoredWorkbenchAgentId,
} from './workbenchStorage'

/** 与后端 ``page_size`` 上限（≤100）一致 */
const WORKBENCH_LIST_PAGE_SIZE = 100

export type WorkbenchContextValue = {
  /** 全部系统工作台 Agent（每个工作区一条 agent_kind=workbench） */
  workbenchAgents: AgentOut[]
  /** 解析后的默认/当前工作台主键（用于首页跳转与侧栏高亮辅助） */
  workbenchAgentId: number | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  /** 切换当前工作台并记住，便于下次进入 */
  selectWorkbenchAgent: (agentId: number) => void
}

export const WorkbenchContext = createContext<WorkbenchContextValue | null>(null)

function resolveDefaultWorkbenchId(agents: AgentOut[]): number | null {
  if (agents.length === 0) return null
  if (agents.length === 1) return agents[0].id
  const stored = readStoredWorkbenchAgentId()
  if (stored != null && agents.some((a) => a.id === stored)) {
    return stored
  }
  return null
}

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const [workbenchAgents, setWorkbenchAgents] = useState<AgentOut[]>([])
  const [workbenchAgentId, setWorkbenchAgentId] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const all: AgentOut[] = []
      let page = 1
      let total = Infinity
      while (all.length < total) {
        const data = await fetchAgentsList({
          page,
          page_size: WORKBENCH_LIST_PAGE_SIZE,
          agent_kind: ['workbench'],
        })
        const batch = data.items ?? []
        total = data.total ?? batch.length
        all.push(...batch)
        if (batch.length < WORKBENCH_LIST_PAGE_SIZE || all.length >= total) {
          break
        }
        page += 1
      }
      setWorkbenchAgents(all)
      setWorkbenchAgentId(resolveDefaultWorkbenchId(all))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '加载失败')
      setWorkbenchAgents([])
      setWorkbenchAgentId(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const selectWorkbenchAgent = useCallback((agentId: number) => {
    writeStoredWorkbenchAgentId(agentId)
    setWorkbenchAgentId(agentId)
  }, [])

  const value = useMemo(
    () => ({
      workbenchAgents,
      workbenchAgentId,
      loading,
      error,
      refresh: load,
      selectWorkbenchAgent,
    }),
    [workbenchAgents, workbenchAgentId, loading, error, load, selectWorkbenchAgent],
  )

  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>
}
