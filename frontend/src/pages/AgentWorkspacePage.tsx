import { AgentWorkspacePageCore } from './ConsolePages'

/** Agent 中心：按数字 ID 进入编排工作区（路由 ``/agents/:agentId``）。 */
export function AgentWorkspacePage() {
  return <AgentWorkspacePageCore workspaceEntry="agent-by-id" />
}
