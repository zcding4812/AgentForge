import { AgentWorkspacePageCore } from './ConsolePages'

/** 工作台：按命名空间进入 orchestrator（路由 ``/workbench/:workspaceNamespace``）。 */
export function WorkbenchWorkspacePage() {
  return <AgentWorkspacePageCore workspaceEntry="workbench-by-namespace" />
}
