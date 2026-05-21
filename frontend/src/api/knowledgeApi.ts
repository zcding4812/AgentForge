/**
 * 知识库 API（GET/POST /api/knowledge…），信封 message + data。
 * 路由实现见后端 ``app.api.knowledge_api``，编排见 ``app.services.knowledge_svc``。
 */

import { parseApiEnvelope } from './parseEnvelope'

export type KnowledgeBaseOut = {
  id: number
  name: string
  slug: string
  description: string | null
  storage_type: string
  retrieval_type: string
  status: string
  embedding_model_config_id: number | null
  milvus_collection: string | null
  minio_prefix: string | null
  chunk_method: string
  chunk_size: number
  chunk_overlap: number
  chunk_separator: string | null
  config_json: Record<string, unknown> | null
  /** 结构化切块策略（与 ``config_json.chunk_strategy`` 及列字段一致） */
  chunk_strategy: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type KnowledgeListData = {
  items: KnowledgeBaseOut[]
  total: number
  page: number
  page_size: number
}

export type KnowledgeDocumentOut = {
  id: number
  kb_id: number
  filename: string
  object_key: string | null
  size_bytes: number | null
  mime: string | null
  sha256: string | null
  status: string
  content_version: string
  chunk_count: number
  chunk_method: string | null
  chunk_size: number | null
  chunk_overlap: number | null
  chunk_separator: string | null
  created_at: string
  updated_at: string
}

/** POST 上传 / 重新 ingest 接受体，与 OpenAPI ``KnowledgeDocumentIngestAccepted`` 一致 */
export type KnowledgeDocumentIngestAccepted = {
  document: KnowledgeDocumentOut
  /** 幂等去重未新建任务时为 null */
  task_id: string | null
}

/** PATCH /api/knowledge/{kb_id}/documents/{doc_id}；chunk_method 显式 null 表示恢复知识库默认 */
export type KnowledgeDocumentPatchBody = {
  chunk_method?: string | null
  chunk_size?: number | null
  chunk_overlap?: number | null
  chunk_separator?: string | null
}

export type KnowledgeDocumentListData = {
  items: KnowledgeDocumentOut[]
  total: number
  page: number
  page_size: number
}

/** POST /api/knowledge/{kb_id}/search */
export type KnowledgeSearchBody = {
  q: string
  limit?: number
  /** 仅在该文档的分片中检索（与全库检索共用同一接口） */
  doc_id?: number | null
  /** 覆盖知识库 retrieval_type；不传则与库配置一致 */
  retrieval_override?: 'keyword' | 'vector' | 'hybrid' | null
  /** 各子路召回上限（≥ limit，≤100）；混合/向量扩大预取；不传则服务端默认 */
  recall_limit?: number | null
  /** 混合检索 RRF 常数 k（1～200）；不传则服务端默认 60 */
  rrf_k?: number | null
}

export type KnowledgeSearchHit = {
  doc_id: number
  chunk_index: number | null
  text_snippet: string | null
  filename: string | null
  match_type: 'chunk' | 'filename' | 'keyword' | 'vector'
  /** 混合路为 RRF 合分；关键词/向量单路多为空 */
  score?: number | null
}

export type KnowledgeSearchData = {
  items: KnowledgeSearchHit[]
  total: number
  /** 知识库索引配置中的检索类型 */
  configured_retrieval: 'keyword' | 'vector' | 'hybrid'
  /** 本次请求实际执行的检索类型 */
  applied_retrieval: 'keyword' | 'vector' | 'hybrid'
  /** 降级或说明 */
  note?: string | null
}

/** POST /api/knowledge/{kb_id}/vectors/build — 从 Mongo 已有分片构建 Milvus 向量（不重跑 ingest） */
export type KnowledgeVectorBuildData = {
  built: number
  skipped: number
  errors: string[]
}

export type KnowledgeChunkOut = {
  chunk_index: number
  text: string
  created_at: string | null
}

export type KnowledgeDocumentChunksData = {
  items: KnowledgeChunkOut[]
  total: number
  page: number
  page_size: number
  doc_id: number
  filename: string | null
  content_version: string
  document_status: string | null
  /** total=0 时后端给出的说明 */
  empty_hint?: string | null
}

export type KnowledgeCreateBody = {
  name: string
  slug?: string | null
  description?: string | null
}

/** PATCH /api/knowledge/{kb_id}；字段与 OpenAPI 对齐，均为可选 */
export type KnowledgeUpdateBody = {
  name?: string | null
  description?: string | null
  storage_type?: 'keyword' | 'vector' | 'hybrid'
  retrieval_type?: 'keyword' | 'vector' | 'hybrid'
  status?: 'empty' | 'draft' | 'ready' | 'processing' | 'failed'
  embedding_model_config_id?: number | null
  milvus_collection?: string | null
  minio_prefix?: string | null
  chunk_method?: string | null
  chunk_size?: number | null
  chunk_overlap?: number | null
  chunk_separator?: string | null
  config_json?: Record<string, unknown> | null
  /** PATCH 时与顶栏 chunk_* 同步写入 ``config_json.chunk_strategy`` */
  chunk_strategy?: Record<string, unknown> | null
}

const jsonHeaders = { 'Content-Type': 'application/json', Accept: 'application/json' }

const base = '/api/knowledge'

export async function fetchKnowledgeBasesList(params: {
  page?: number
  page_size?: number
  q?: string
  status?: string
}): Promise<KnowledgeListData> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.q?.trim()) sp.set('q', params.q.trim())
  if (params.status?.trim()) sp.set('status', params.status.trim())
  const q = sp.toString()
  const url = q ? `${base}?${q}` : base
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<KnowledgeListData>(res)
}

export async function fetchKnowledgeBase(kbId: number): Promise<KnowledgeBaseOut> {
  const res = await fetch(`${base}/${kbId}`, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<KnowledgeBaseOut>(res)
}

export async function postBuildKnowledgeVectors(kbId: number): Promise<KnowledgeVectorBuildData> {
  const res = await fetch(`${base}/${kbId}/vectors/build`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })
  return parseApiEnvelope<KnowledgeVectorBuildData>(res)
}

export async function updateKnowledgeBase(
  kbId: number,
  body: KnowledgeUpdateBody,
): Promise<KnowledgeBaseOut> {
  const res = await fetch(`${base}/${kbId}`, {
    method: 'PATCH',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<KnowledgeBaseOut>(res)
}

export async function createKnowledgeBase(body: KnowledgeCreateBody): Promise<KnowledgeBaseOut> {
  const res = await fetch(base, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<KnowledgeBaseOut>(res)
}

export async function deleteKnowledgeBase(kbId: number): Promise<void> {
  const res = await fetch(`${base}/${kbId}`, { method: 'DELETE', headers: { Accept: 'application/json' } })
  if (res.status === 204) return
  await parseApiEnvelope<unknown>(res)
}

export async function fetchKnowledgeDocuments(params: {
  kbId: number
  page?: number
  page_size?: number
  q?: string
  /**
   * 为 true 时切块第 2 步左侧「待配置」列表：排除「已 indexed 且已有文档级切块覆盖」；
   * 恢复默认后重新出现在左侧。
   */
  chunk_left_panel?: boolean
}): Promise<KnowledgeDocumentListData> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.q?.trim()) sp.set('q', params.q.trim())
  if (params.chunk_left_panel) sp.set('chunk_left_panel', 'true')
  const q = sp.toString()
  const url = q ? `${base}/${params.kbId}/documents?${q}` : `${base}/${params.kbId}/documents`
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<KnowledgeDocumentListData>(res)
}

export async function uploadKnowledgeDocument(
  kbId: number,
  file: File,
): Promise<KnowledgeDocumentIngestAccepted> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${base}/${kbId}/documents/upload`, {
    method: 'POST',
    body: fd,
    headers: { Accept: 'application/json' },
  })
  return parseApiEnvelope<KnowledgeDocumentIngestAccepted>(res)
}

export async function deleteKnowledgeDocument(kbId: number, docId: number): Promise<void> {
  const res = await fetch(`${base}/${kbId}/documents/${docId}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  })
  if (res.status === 204) return
  await parseApiEnvelope<unknown>(res)
}

export async function patchKnowledgeDocument(
  kbId: number,
  docId: number,
  body: KnowledgeDocumentPatchBody,
): Promise<KnowledgeDocumentOut> {
  const res = await fetch(`${base}/${kbId}/documents/${docId}`, {
    method: 'PATCH',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  })
  return parseApiEnvelope<KnowledgeDocumentOut>(res)
}

/** POST /api/knowledge/{kb_id}/documents/{doc_id}/ingest — 重新触发 ingest */
export async function postRequeueDocumentIngest(
  kbId: number,
  docId: number,
): Promise<KnowledgeDocumentIngestAccepted> {
  const res = await fetch(`${base}/${kbId}/documents/${docId}/ingest`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })
  return parseApiEnvelope<KnowledgeDocumentIngestAccepted>(res)
}

export async function searchKnowledgeInKb(
  kbId: number,
  body: KnowledgeSearchBody,
): Promise<KnowledgeSearchData> {
  const payload: Record<string, unknown> = {
    q: body.q.trim(),
    limit: body.limit ?? 20,
  }
  if (body.doc_id != null && Number.isFinite(Number(body.doc_id))) {
    payload.doc_id = Number(body.doc_id)
  }
  if (body.retrieval_override != null) {
    payload.retrieval_override = body.retrieval_override
  }
  if (body.recall_limit != null && Number.isFinite(Number(body.recall_limit))) {
    payload.recall_limit = Number(body.recall_limit)
  }
  if (body.rrf_k != null && Number.isFinite(Number(body.rrf_k))) {
    payload.rrf_k = Number(body.rrf_k)
  }
  const res = await fetch(`${base}/${kbId}/search`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  })
  return parseApiEnvelope<KnowledgeSearchData>(res)
}

export async function fetchDocumentChunks(params: {
  kbId: number
  docId: number
  page?: number
  page_size?: number
  content_version?: string | null
}): Promise<KnowledgeDocumentChunksData> {
  const sp = new URLSearchParams()
  if (params.page != null) sp.set('page', String(params.page))
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.content_version?.trim()) sp.set('content_version', params.content_version.trim())
  const q = sp.toString()
  const url = q
    ? `${base}/${params.kbId}/documents/${params.docId}/chunks?${q}`
    : `${base}/${params.kbId}/documents/${params.docId}/chunks`
  const res = await fetch(url, { headers: { Accept: 'application/json' } })
  return parseApiEnvelope<KnowledgeDocumentChunksData>(res)
}
