/** localStorage：上次进入的工作台 Agent（多工作区时用于默认选中） */
export const WORKBENCH_SELECTED_AGENT_STORAGE_KEY = 'agent-forge.selectedWorkbenchAgentId'

export function readStoredWorkbenchAgentId(): number | null {
  try {
    const raw = localStorage.getItem(WORKBENCH_SELECTED_AGENT_STORAGE_KEY)
    if (raw == null || raw === '') return null
    const n = Number.parseInt(raw, 10)
    return Number.isFinite(n) && n >= 1 ? n : null
  } catch {
    return null
  }
}

export function writeStoredWorkbenchAgentId(agentId: number): void {
  try {
    localStorage.setItem(WORKBENCH_SELECTED_AGENT_STORAGE_KEY, String(agentId))
  } catch {
    /* ignore quota / private mode */
  }
}
