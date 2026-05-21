import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { WorkbenchProvider } from './contexts/WorkbenchContext'
import { AppLayout } from './layouts/AppLayout'
import {
  AgentListPage,
  AgentWorkspacePage,
  KnowledgeDetailPage,
  KnowledgeListPage,
  PixelAgentsShell,
  StudioChatLandingPage,
  SystemMonitorPage,
  TraceMonitorDetailPage,
  TraceMonitorPage,
  ToolsPage,
  WorkbenchHomePage,
  WorkbenchWorkspacePage,
} from './pages'
import { ModelListPanel, ModelProvidersPanel } from './pages/ProviderSettings'

function App() {
  return (
    <BrowserRouter>
      <WorkbenchProvider>
      <Routes>
        <Route path="/pixel-office" element={<PixelAgentsShell />} />
        <Route path="/studio" element={<StudioChatLandingPage />} />
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Navigate to="/workbench" replace />} />
          <Route path="workbench" element={<WorkbenchHomePage />} />
          <Route path="workbench/:workspaceNamespace" element={<WorkbenchWorkspacePage />} />
          <Route path="agents" element={<AgentListPage />} />
          <Route path="agents/:agentId" element={<AgentWorkspacePage />} />
          <Route path="resources/knowledge" element={<KnowledgeListPage />} />
          <Route
            path="resources/knowledge/:kbId"
            element={<KnowledgeDetailPage />}
          />
          <Route path="resources/tools" element={<ToolsPage />} />
          <Route path="monitor" element={<Navigate to="/monitor/system" replace />} />
          <Route path="monitor/system" element={<SystemMonitorPage />} />
          <Route path="monitor/tracing" element={<TraceMonitorPage />} />
          <Route path="monitor/tracing/:traceId" element={<TraceMonitorDetailPage />} />
          <Route path="settings" element={<Navigate to="/settings/models" replace />} />
          <Route path="settings/models" element={<ModelListPanel />} />
          <Route path="settings/providers" element={<ModelProvidersPanel />} />
          <Route path="settings/logs" element={<Navigate to="/settings/models" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      </WorkbenchProvider>
    </BrowserRouter>
  )
}

export default App
