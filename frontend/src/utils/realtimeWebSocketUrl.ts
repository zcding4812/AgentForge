/**
 * 浏览器内构造与当前站同源、路径为 /api/realtime/ws 的 WebSocket URL（开发环境经 Vite 代理到后端）。
 */
export function getRealtimeWebSocketUrl(): string {
  const u = new URL('/api/realtime/ws', window.location.href)
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
  return u.toString()
}
