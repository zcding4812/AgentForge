/**
 * GET /api/workspace-namespaces — 工作区命名空间列表（slug 与 Agent workspace_namespace 一致）
 */

import { parseApiEnvelope } from './parseEnvelope'

export type WorkspaceNamespaceOut = {
  id: number
  slug: string
  workbench_agent_id: number | null
  created_at: string
  updated_at: string
}

export type WorkspaceNamespaceListData = {
  items: WorkspaceNamespaceOut[]
}

const base = '/api/workspace-namespaces'

export async function fetchWorkspaceNamespaces(): Promise<WorkspaceNamespaceListData> {
  const res = await fetch(base, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<WorkspaceNamespaceListData>(res)
}
