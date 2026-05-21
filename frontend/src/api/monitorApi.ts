import { parseApiEnvelope } from './parseEnvelope'

export type MonitorHealth = {
  status: string
  uptime_seconds: number
  database_status: string
}

export type InfraComponentStatus = {
  key: string
  label: string
  status: 'ok' | 'error' | 'disabled'
  detail: string | null
  /** 脱敏连接串或端点（后端字段） */
  address?: string | null
}

export type TokenUsageDayPoint = {
  date: string
  user_tokens: number
  assistant_tokens: number
  /** user_tokens + assistant_tokens */
  total_tokens: number
}

export type MonitorDashboardData = {
  health: MonitorHealth
  infrastructure: InfraComponentStatus[]
  token_usage: TokenUsageDayPoint[]
  token_usage_grand_total: number
}

/** GET /api/system/monitor-dashboard */
export async function fetchMonitorDashboard(tokenDays: number = 14): Promise<MonitorDashboardData> {
  const sp = new URLSearchParams()
  sp.set('token_days', String(tokenDays))
  const res = await fetch(`/api/system/monitor-dashboard?${sp.toString()}`, {
    headers: { Accept: 'application/json' },
  })
  return parseApiEnvelope<MonitorDashboardData>(res)
}
