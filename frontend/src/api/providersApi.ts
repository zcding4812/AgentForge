/**
 * LLM 提供商（Provider）与挂载模型 API（信封 message + data，与 FastAPI 约定一致）
 */

import { MAX_PAGINATION_PAGES } from '../constants'
import { parseApiEnvelope, readJsonBody } from './parseEnvelope'

/** 筛选用：chat/embedding/ocr；列表展示为后端返回的 model_type（如 llm） */
export type ModelType = 'chat' | 'embedding' | 'ocr' | string
export type HealthStatus = 'ok' | 'error' | 'checking' | 'unknown'

export type PageMeta = { page: number; page_size: number; total: number }
export type Paged<T> = { items: T[]; meta: PageMeta }

export type ModelProvider = {
  id: string
  name: string
  supported_model_types: ModelType[]
  description?: string | null
  enabled: boolean
  model_count: number
  provider_kind: string
  protocol: string
  base_url: string
  api_key_masked: string
  org_id?: string | null
  weight: number
  timeout_sec: number
  max_retries: number
  is_default: boolean
}

/** 与后端 `sys_model` + 提供商展示字段对齐 */
export type LlmModel = {
  /** sys_model.id，更新/删除/探测路径使用 */
  id: string
  model_code: string
  model_name: string
  /** provider_code */
  provider_id: string
  provider_name: string
  model_type: ModelType
  /** sys_model.endpoint */
  endpoint: string | null
  /** sys_model_provider.base_url */
  provider_base_url: string | null
  /** sys_model.timeout（秒） */
  timeout: number
  api_key_masked: string
  /** sys_model.is_enabled */
  enabled: boolean
  health_status: HealthStatus
  health_message: string | null
}

const jsonHeaders = { 'Content-Type': 'application/json', Accept: 'application/json' }

/** 与后端 `GET /api/providers` 的 `page_size` 上限一致 */
export const MODEL_PROVIDER_PAGE_SIZE_MAX = 200

const base = '/api/providers'

export async function fetchProviders(params: {
  q?: string
  status?: string
  page?: number
  page_size?: number
}): Promise<Paged<ModelProvider>> {
  const sp = new URLSearchParams()
  if (params.q) sp.set('q', params.q)
  if (params.status) sp.set('status', params.status)
  sp.set('page', String(params.page ?? 1))
  const ps = Math.min(params.page_size ?? 10, MODEL_PROVIDER_PAGE_SIZE_MAX)
  sp.set('page_size', String(ps))
  const res = await fetch(`${base}?${sp.toString()}`)
  return parseApiEnvelope<Paged<ModelProvider>>(res)
}

/** 分页拉取全部提供商（用于下拉/筛选，避免单次 page_size 超过接口上限导致 422） */
export async function fetchAllModelProviders(): Promise<ModelProvider[]> {
  const first = await fetchProviders({ page: 1, page_size: MODEL_PROVIDER_PAGE_SIZE_MAX })
  let items = [...first.items]
  let page = 2
  const maxPages = MAX_PAGINATION_PAGES
  while (items.length < first.meta.total && page <= maxPages) {
    const d = await fetchProviders({ page, page_size: MODEL_PROVIDER_PAGE_SIZE_MAX })
    items = items.concat(d.items)
    if (d.items.length === 0) break
    page += 1
  }
  return items
}

export async function createProvider(body: Record<string, unknown>): Promise<ModelProvider> {
  const res = await fetch(base, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<ModelProvider>(res)
}

export async function updateProvider(id: string, body: Record<string, unknown>): Promise<ModelProvider> {
  const res = await fetch(`${base}/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<ModelProvider>(res)
}

export async function deleteProvider(id: string): Promise<void> {
  const res = await fetch(`${base}/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (res.status === 204 || res.status === 200) return
  if (!res.ok) {
    const body = (await readJsonBody<{ detail?: unknown }>(res)) ?? {}
    throw new Error(typeof body.detail === 'string' ? body.detail : '删除失败')
  }
}

export async function probeProvider(id: string): Promise<{ ok: boolean; message?: string }> {
  const res = await fetch(`${base}/${encodeURIComponent(id)}/probe`, { method: 'POST' })
  return parseApiEnvelope<{ ok: boolean; message?: string }>(res)
}

export async function fetchLlmModels(params: {
  q?: string
  provider_id?: string
  model_type?: ModelType
  status?: string
  /** 默认 true：仅返回提供商已启用（status=1）下的模型；管理台需看全部时传 false */
  provider_enabled_only?: boolean
  page?: number
  page_size?: number
}): Promise<Paged<LlmModel>> {
  const sp = new URLSearchParams()
  if (params.q) sp.set('q', params.q)
  if (params.provider_id) sp.set('provider_id', params.provider_id)
  if (params.model_type) sp.set('model_type', params.model_type)
  if (params.status) sp.set('status', params.status)
  if (params.provider_enabled_only !== undefined) {
    sp.set('provider_enabled_only', String(params.provider_enabled_only))
  }
  sp.set('page', String(params.page ?? 1))
  sp.set('page_size', String(params.page_size ?? 10))
  const res = await fetch(`${base}/models?${sp.toString()}`)
  return parseApiEnvelope<Paged<LlmModel>>(res)
}

/** 拉取全部已启用的对话类（chat）系统模型，用于 Agent 下拉；多页拼接。 */
export async function fetchAllChatLlmModels(): Promise<LlmModel[]> {
  const first = await fetchLlmModels({
    page: 1,
    page_size: MODEL_PROVIDER_PAGE_SIZE_MAX,
    model_type: 'chat',
    status: 'enabled',
    provider_enabled_only: true,
  })
  let items = [...first.items]
  let page = 2
  const maxPages = MAX_PAGINATION_PAGES
  while (items.length < first.meta.total && page <= maxPages) {
    const d = await fetchLlmModels({
      page,
      page_size: MODEL_PROVIDER_PAGE_SIZE_MAX,
      model_type: 'chat',
      status: 'enabled',
      provider_enabled_only: true,
    })
    items = items.concat(d.items)
    if (d.items.length === 0) break
    page += 1
  }
  return items
}

/** 拉取全部已启用的 embedding 类系统模型，用于知识库「索引配置」等下拉 */
export async function fetchAllEmbeddingModels(): Promise<LlmModel[]> {
  const first = await fetchLlmModels({
    page: 1,
    page_size: MODEL_PROVIDER_PAGE_SIZE_MAX,
    model_type: 'embedding',
    status: 'enabled',
    provider_enabled_only: true,
  })
  let items = [...first.items]
  let page = 2
  const maxPages = MAX_PAGINATION_PAGES
  while (items.length < first.meta.total && page <= maxPages) {
    const d = await fetchLlmModels({
      page,
      page_size: MODEL_PROVIDER_PAGE_SIZE_MAX,
      model_type: 'embedding',
      status: 'enabled',
      provider_enabled_only: true,
    })
    items = items.concat(d.items)
    if (d.items.length === 0) break
    page += 1
  }
  return items
}

export async function createLlmModel(body: Record<string, unknown>): Promise<LlmModel> {
  const res = await fetch(`${base}/models`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<LlmModel>(res)
}

export async function updateLlmModel(id: string, body: Record<string, unknown>): Promise<LlmModel> {
  const res = await fetch(`${base}/models/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<LlmModel>(res)
}

export async function deleteLlmModel(id: string): Promise<void> {
  const res = await fetch(`${base}/models/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (res.status === 204 || res.status === 200) return
  if (!res.ok) {
    const body = (await readJsonBody<{ detail?: unknown }>(res)) ?? {}
    throw new Error(typeof body.detail === 'string' ? body.detail : '删除失败')
  }
}

export async function probeLlmModel(id: string): Promise<{ ok: boolean; message?: string }> {
  const res = await fetch(`${base}/models/${encodeURIComponent(id)}/probe`, { method: 'POST' })
  return parseApiEnvelope<{ ok: boolean; message?: string }>(res)
}
