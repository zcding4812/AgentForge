/**
 * 嵌入 pixel-agents 静态 webview：对接本仓库后端 Agent 列表、布局与座位持久化（localStorage）；
 * 同步/离场 Agent 时向父窗 `postMessage`（`type: ai-agents-canvas`），由 PixelAgentsShell 转发到 `/api/realtime/ws` 主题 `pixel:namespace:<ns>`。
 * 游戏内需自定义上报时可使用 `type: pixelAgentAction` 或 `ai-agents-canvas`（见前壳 `CANVAS_TO_REALTIME_MSG_TYPES`）。
 * 须在 index.html 中于主 bundle 之后引入。画布非缩放类 UI 隐藏见同目录 `view-mode.css`（缩放按钮/倍率提示不再隐藏）。
 *
 * 默认站位：agent-seat-defaults.json 中 seatByAgentKind.workbench 固定为工位 uid（全命名空间唯一）；
 * 其它类型不配 seatId 则由主包默认落点。可选 seatByAgentId 覆盖单 Agent。
 * 调试用：画布拖拽后 localStorage「ai-agents-pixel-seats-v1」可对照 seatId；重载后 agentCreated 带 seatId。
 */
;(function () {
  var STORAGE_LAYOUT = 'ai-agents-pixel-layout-v1'
  var STORAGE_SEATS = 'ai-agents-pixel-seats-v1'
  var addedToOffice = new Set()
  var seatDefaults = { seatByAgentKind: {}, seatByAgentId: {} }

  function patchConsole() {
    var orig = console.log
    console.log = function () {
      var a = arguments
      if (a.length >= 2 && a[0] === '[vscode.postMessage]' && a[1] && typeof a[1] === 'object') {
        var msg = a[1]
        if (msg.type === 'saveLayout' && msg.layout) {
          try {
            localStorage.setItem(STORAGE_LAYOUT, JSON.stringify(msg.layout))
          } catch (e) {}
        }
        if (msg.type === 'saveAgentSeats' && msg.seats) {
          try {
            localStorage.setItem(STORAGE_SEATS, JSON.stringify(msg.seats))
          } catch (e) {}
        }
      }
      return orig.apply(console, a)
    }
  }

  patchConsole()

  function loadSeatDefaults() {
    return fetch('./agent-seat-defaults.json', { cache: 'no-store' })
      .then(function (r) {
        return r.ok ? r.json() : {}
      })
      .then(function (j) {
        if (j && typeof j === 'object') {
          seatDefaults = {
            seatByAgentKind: j.seatByAgentKind && typeof j.seatByAgentKind === 'object' ? j.seatByAgentKind : {},
            seatByAgentId: j.seatByAgentId && typeof j.seatByAgentId === 'object' ? j.seatByAgentId : {},
          }
        }
      })
      .catch(function () {})
  }

  function resolveSeatId(agent) {
    var byId = seatDefaults.seatByAgentId[String(agent.id)]
    if (typeof byId === 'number' && !isNaN(byId)) return byId
    if (typeof byId === 'string' && byId.length > 0) return byId
    var kind = agent.agent_kind
    if (kind) {
      var byKind = seatDefaults.seatByAgentKind[kind]
      if (typeof byKind === 'number' && !isNaN(byKind)) return byKind
      if (typeof byKind === 'string' && byKind.length > 0) return byKind
    }
    return undefined
  }

  function dispatchSavedLayout() {
    try {
      var raw = localStorage.getItem(STORAGE_LAYOUT)
      if (!raw) return
      var layout = JSON.parse(raw)
      window.postMessage({ type: 'layoutLoaded', layout: layout, wasReset: false }, '*')
    } catch (e) {}
  }

  /** 在 browserMock 默认布局之后覆盖为上次保存（与上游 Pixel Agents 行为一致） */
  setTimeout(dispatchSavedLayout, 1550)

  function apiItems(body) {
    if (!body || typeof body !== 'object') return []
    var d = body.data
    if (d && Array.isArray(d.items)) return d.items
    return []
  }

  function agentsQuery() {
    var params = new URLSearchParams(window.location.search)
    /** 与主壳 PixelAgentsShell 一致：缺省为 default，保证与 GET /api/agents 筛选一致 */
    var ns = (params.get('namespace') || 'default').trim() || 'default'
    return '?page_size=100&workspace_namespace=' + encodeURIComponent(ns)
  }

  function syncAgents() {
    fetch('/api/agents' + agentsQuery(), { headers: { Accept: 'application/json' } })
      .then(function (r) {
        return r.json()
      })
      .then(function (body) {
        var items = apiItems(body)
        var ids = new Set(
          items.map(function (a) {
            return a.id
          }),
        )
        Array.from(addedToOffice).forEach(function (id) {
          if (!ids.has(id)) {
            var closed = { type: 'agentClosed', id: id }
            window.postMessage(closed, '*')
            if (window.parent && window.parent !== window) {
              try {
                window.parent.postMessage(
                  {
                    type: 'ai-agents-canvas',
                    name: 'agentClosed',
                    body: closed,
                  },
                  '*',
                )
              } catch (e) {}
            }
            addedToOffice.delete(id)
          }
        })
        items.forEach(function (a, i) {
          if (addedToOffice.has(a.id)) return
          var delay = 1150 + i * 55
          setTimeout(function () {
            var payload = {
              type: 'agentCreated',
              id: a.id,
              folderName: a.name || 'Agent ' + a.id,
            }
            var sid = resolveSeatId(a)
            if (sid !== undefined) payload.seatId = sid
            window.postMessage(payload, '*')
            if (window.parent && window.parent !== window) {
              try {
                window.parent.postMessage(
                  {
                    type: 'ai-agents-canvas',
                    name: 'agentCreated',
                    body: payload,
                    agent_kind: a.agent_kind,
                  },
                  '*',
                )
              } catch (e) {}
            }
            addedToOffice.add(a.id)
          }, delay)
        })
      })
      .catch(function () {})
  }

  window.addEventListener('load', function () {
    void loadSeatDefaults().then(function () {
      setTimeout(syncAgents, 400)
    })
  })

  window.addEventListener('message', function (ev) {
    if (ev.data && ev.data.type === 'ai-agents-bridge-refresh') {
      void loadSeatDefaults().then(function () {
        syncAgents()
      })
    }
  })
})()
