/**
 * 统一解析 FastAPI `message` + `data` 信封，并兼容空 body（避免 `response.json()` 在空正文上抛错）。
 */

export type ApiEnvelope<T> = { message?: string; data: T }

function formatDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return JSON.stringify(detail)
  return JSON.stringify(detail ?? fallback)
}

/**
 * 读取 Response 正文并 `JSON.parse`；空正文返回 `null`。
 * 用于 DELETE 错误分支或非标准空响应，避免 `Unexpected end of JSON input`。
 */
export async function readJsonBody<T = unknown>(res: Response): Promise<T | null> {
  const text = await res.text()
  const trimmed = text.trim()
  if (!trimmed) return null
  try {
    return JSON.parse(trimmed) as T
  } catch {
    throw new Error(`响应非合法 JSON（HTTP ${res.status}）`)
  }
}

/** 解析 `message` + `data` 信封；兼容空 body、错误响应无 JSON body */
export async function parseApiEnvelope<T>(res: Response): Promise<T> {
  const body = (await readJsonBody<ApiEnvelope<T> | { detail?: unknown }>(res)) as
    | ApiEnvelope<T>
    | { detail?: unknown }
    | null
  if (!res.ok) {
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body ? body.detail : res.statusText
    throw new Error(formatDetail(detail, res.statusText))
  }
  if (!body || typeof body !== 'object' || !('data' in body)) {
    throw new Error(body == null ? '响应为空' : '响应格式错误：缺少 data')
  }
  return (body as ApiEnvelope<T>).data
}
