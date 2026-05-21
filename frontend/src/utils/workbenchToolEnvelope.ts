/**
 * 工作台内置工具等业务返回的信封：与后端 `wb_ok` / `wb_err` 对齐。
 * 用于前端展示 assistant 的 `structured` 或从文本中解析出的 JSON。
 */

export type WorkbenchToolEnvelope = {
  code: string
  message: string
  data: unknown
}

export function isWorkbenchToolEnvelope(v: unknown): v is WorkbenchToolEnvelope {
  if (v == null || typeof v !== 'object' || Array.isArray(v)) return false
  const o = v as Record<string, unknown>
  return (
    typeof o.code === 'string' &&
    typeof o.message === 'string' &&
    'data' in o
  )
}
