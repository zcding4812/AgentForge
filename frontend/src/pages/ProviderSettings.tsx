import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  SearchOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import {
  Button,
  Collapse,
  Drawer,
  Form,
  Input,
  InputNumber,
  Pagination,
  Popconfirm,
  Radio,
  Select,
  Skeleton,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  type LlmModel,
  type ModelProvider,
  type ModelType,
  createLlmModel,
  createProvider,
  deleteLlmModel,
  deleteProvider,
  fetchAllModelProviders,
  fetchLlmModels,
  fetchProviders,
  probeLlmModel,
  probeProvider,
  updateLlmModel,
  updateProvider,
} from '../api/providersApi'
import { TableColumnFilterHeader } from '../components/TableColumnFilterHeader'
import './agentListPage.css'

const { Text } = Typography

const TYPE_LABELS: Record<string, string> = {
  chat: '对话',
  llm: '对话',
  embedding: '嵌入',
  tts: '语音合成',
  stt: '语音识别',
  ocr: 'OCR',
  image: '图像',
}

/** 与库表 api_format 一致，写入 provider_kind → 后端存 api_format（每项 value 须唯一，勿重复 openai） */
const PROVIDER_KIND_OPTIONS = [
  { label: 'OpenAI 兼容（默认）', value: 'openai' },
  { label: 'Anthropic', value: 'anthropic' },
  { label: '百度', value: 'baidu' },
  { label: '腾讯混元', value: 'tencent' },
]

import { DEFAULT_PAGE_SIZE } from '../constants'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export function ModelProvidersPanel() {
  const [loading, setLoading] = useState(true)
  const [rows, setRows] = useState<ModelProvider[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [searchDraft, setSearchDraft] = useState('')
  const [searchApplied, setSearchApplied] = useState('')
  const [status, setStatus] = useState<string>('all')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [editing, setEditing] = useState<ModelProvider | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [form] = Form.useForm<Record<string, unknown>>()

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchProviders({
        q: searchApplied.trim() || undefined,
        status: status === 'all' ? undefined : status,
        page,
        page_size: PAGE_SIZE,
      })
      setRows(data.items)
      setTotal(data.meta.total)
    } catch (e) {
      setRows([])
      setTotal(0)
      message.error(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [searchApplied, status, page])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    const onRefresh = () => void load()
    window.addEventListener('app:refresh', onRefresh)
    return () => window.removeEventListener('app:refresh', onRefresh)
  }, [load])

  const openCreate = () => {
    setEditing(null)
    form.resetFields()
    form.setFieldsValue({
      enabled: false,
      provider_kind: 'openai',
    })
    setDrawerOpen(true)
  }

  const openEdit = (r: ModelProvider) => {
    setEditing(r)
    form.setFieldsValue({
      id: r.id,
      name: r.name,
      enabled: r.enabled,
      provider_kind: r.provider_kind,
      base_url: r.base_url,
      api_key: '',
    })
    setDrawerOpen(true)
  }

  const submit = async () => {
    try {
      const v = await form.validateFields()
      setSubmitting(true)
      /** 与 sys_model_provider 可落库字段一致（见 ModelProviderUpdate / ProviderService.update_provider） */
      const payload: Record<string, unknown> = {
        name: v.name,
        enabled: v.enabled,
        provider_kind: v.provider_kind,
        base_url: String(v.base_url).trim(),
      }
      if (!editing) {
        payload.id = v.id
        payload.api_key = String(v.api_key).trim()
        await createProvider(payload)
        message.success('已创建')
      } else {
        const ak = String(v.api_key ?? '').trim()
        if (ak.length >= 8) payload.api_key = ak
        await updateProvider(editing.id, payload)
        message.success('已保存')
      }
      setDrawerOpen(false)
      void load()
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) return
      message.error(e instanceof Error ? e.message : '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteProvider(id)
      message.success('已删除')
      void load()
    } catch (e) {
      message.error(e instanceof Error ? e.message : '删除失败')
    }
  }

  const runProbe = async (r: ModelProvider) => {
    try {
      const res = await probeProvider(r.id)
      message[res.ok ? 'success' : 'warning'](res.ok ? '连通性正常' : res.message || '探测失败')
    } catch (e) {
      message.error(e instanceof Error ? e.message : '探测失败')
    }
  }

  const columns: ColumnsType<ModelProvider> = [
    {
      title: '提供商名称',
      dataIndex: 'name',
      ellipsis: true,
      width: 152,
      render: (name: string, r) => (
        <Tooltip title={`${name} · ${r.id}`}>
          <div>
            <Text strong style={{ color: 'var(--color-text, #1d2129)' }}>
              {name}
            </Text>
            <div style={{ fontSize: 12, color: '#86909c' }}>{r.id}</div>
          </div>
        </Tooltip>
      ),
    },
    {
      title: '类型',
      dataIndex: 'provider_kind',
      width: 96,
      ellipsis: true,
      render: (k: string) =>
        PROVIDER_KIND_OPTIONS.find((o) => o.value === k)?.label ?? k,
    },
    {
      title: 'Base URL',
      dataIndex: 'base_url',
      width: 160,
      ellipsis: true,
      render: (u: string) => (
        <Tooltip title={u}>
          <Text code style={{ fontSize: 12 }}>
            {u}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: (
        <TableColumnFilterHeader
          label="状态"
          tooltip="按状态筛选"
          active={status !== 'all'}
        >
          <Radio.Group
            value={status}
            onChange={(e) => {
              setPage(1)
              setStatus(e.target.value)
            }}
          >
            <Space direction="vertical" size={6}>
              <Radio value="all">全部</Radio>
              <Radio value="enabled">已启用</Radio>
              <Radio value="disabled">已禁用</Radio>
            </Space>
          </Radio.Group>
        </TableColumnFilterHeader>
      ),
      width: 108,
      render: (_, r) =>
        r.enabled ? <Tag color="success">已启用</Tag> : <Tag>已禁用</Tag>,
    },
    {
      title: '操作',
      key: 'actions',
      width: 96,
      align: 'center',
      fixed: 'right',
      render: (_, r) => (
        <Space size={2}>
          <Tooltip title="编辑">
            <Button
              type="text"
              className="agent-list-row-action"
              icon={<EditOutlined />}
              aria-label="编辑"
              onClick={() => openEdit(r)}
            />
          </Tooltip>
          <Tooltip title="测试连通">
            <Button
              type="text"
              className="agent-list-row-action"
              icon={<ThunderboltOutlined />}
              aria-label="测试连通"
              onClick={() => void runProbe(r)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Popconfirm title="确定删除该提供商？" onConfirm={() => void handleDelete(r.id)}>
              <Button type="text" danger icon={<DeleteOutlined />} aria-label="删除" />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ]

  return (
    <div className="agent-list-page">
      <div className="agent-list-table-shell">
        <div className="agent-list-toolbar">
          <div
            className="agent-list-toolbar-left"
            style={{ flex: 1, minWidth: 0, flexWrap: 'wrap', rowGap: 8 }}
          >
            <Button
              type="primary"
              className="agent-list-action-primary"
              icon={<PlusOutlined />}
              onClick={openCreate}
            >
              添加提供商
            </Button>
            <Text type="secondary" style={{ fontSize: 13, marginLeft: 4 }}>
              一个提供商维护一套 Base URL 与密钥；多个模型请在「模型列表」中绑定同一提供商。
            </Text>
          </div>
          <div className="agent-list-search-wrap">
            <Input
              className="agent-list-search-input"
              prefix={<SearchOutlined className="agent-list-search-icon" />}
              placeholder="搜索名称或 ID（回车）"
              allowClear
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
              onPressEnter={() => {
                setPage(1)
                setSearchApplied(searchDraft.trim())
              }}
            />
          </div>
        </div>

        {loading ? (
          <div className="agent-list-table-wrap" style={{ padding: 24 }}>
            <Skeleton active paragraph={{ rows: 8 }} />
          </div>
        ) : (
          <div className="agent-list-table-wrap">
            <div className="agent-list-table-inner">
              <Table<ModelProvider>
                className="agent-list-table"
                rowKey="id"
                size="small"
                columns={columns}
                dataSource={rows}
                pagination={false}
                scroll={{ x: 820 }}
                locale={{
                  emptyText: (
                    <div style={{ padding: '48px 0', textAlign: 'center' }}>
                      <Text type="secondary">暂无提供商</Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={openCreate}
                        >
                          添加提供商
                        </Button>
                      </div>
                    </div>
                  ),
                }}
              />
            </div>
            <div className="agent-list-pagination-bar">
              <Pagination
                size="small"
                current={page}
                pageSize={PAGE_SIZE}
                total={total}
                showSizeChanger={false}
                showTotal={(t) => `共 ${t} 条`}
                onChange={(p) => setPage(p)}
              />
            </div>
          </div>
        )}
      </div>

      <Drawer
        title={editing ? '编辑提供商' : '添加提供商'}
        width={600}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        destroyOnClose
        footer={
          <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
            <Button onClick={() => setDrawerOpen(false)}>取消</Button>
            <Button type="primary" className="agent-list-btn-primary" loading={submitting} onClick={() => void submit()}>
              {editing ? '保存' : '创建'}
            </Button>
          </Space>
        }
      >
        <Form form={form} layout="vertical" className="settings-provider-form">
          <Collapse
            bordered={false}
            ghost
            className="settings-form-collapse"
            defaultActiveKey={['basic', 'conn']}
            items={[
              {
                key: 'basic',
                label: '基础信息',
                children: (
                  <>
                    <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
                      列表中的「支持类型」由下属模型行的 model_type 自动汇总，不在此单独存储。
                    </Text>
                    <Form.Item
                      name="id"
                      label="提供商编码"
                      rules={[
                        { required: true },
                        { pattern: /^[a-zA-Z0-9_-]{2,64}$/, message: '2-64 位字母数字横线' },
                      ]}
                    >
                      <Input disabled={!!editing} placeholder="例如 openai" autoComplete="off" />
                    </Form.Item>
                    <Form.Item name="name" label="提供商名称" rules={[{ required: true, max: 64 }]}>
                      <Input placeholder="显示名称" />
                    </Form.Item>
                    <Form.Item name="provider_kind" label="API 格式 (api_format)" rules={[{ required: true }]}>
                      <Select options={PROVIDER_KIND_OPTIONS} />
                    </Form.Item>
                  </>
                ),
              },
              {
                key: 'conn',
                label: '连接与鉴权',
                children: (
                  <>
                    <Form.Item name="base_url" label="Base URL" rules={[{ required: true }]}>
                      <Input placeholder="https://api.openai.com/v1" />
                    </Form.Item>
                    <Form.Item
                      name="api_key"
                      label="API Key"
                      dependencies={['enabled']}
                      rules={
                        editing
                          ? []
                          : [
                              ({ getFieldValue }) => ({
                                validator(_: unknown, value: unknown) {
                                  const on = Boolean(getFieldValue('enabled'))
                                  const s = String(value ?? '').trim()
                                  if (on && !s) {
                                    return Promise.reject(new Error('启用前请填写 API Key'))
                                  }
                                  return Promise.resolve()
                                },
                              }),
                            ]
                      }
                      extra={
                        editing
                          ? '留空则不修改密钥；启用前须已配置密钥（或无需密钥的接入方式）'
                          : '未启用时可先保存占位；开启「启用」时必须填写密钥'
                      }
                    >
                      <Input.Password
                        placeholder={editing ? '留空则不修改' : '启用时必填'}
                        autoComplete="new-password"
                      />
                    </Form.Item>
                    <Form.Item
                      name="enabled"
                      label="启用"
                      valuePropName="checked"
                      extra="默认关闭；仅在有可用密钥（或无需密钥）时可开启"
                    >
                      <Switch />
                    </Form.Item>
                  </>
                ),
              },
            ]}
          />
        </Form>
      </Drawer>
    </div>
  )
}

export function ModelListPanel() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [loading, setLoading] = useState(true)
  const [rows, setRows] = useState<LlmModel[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [searchDraft, setSearchDraft] = useState('')
  const [searchApplied, setSearchApplied] = useState('')
  const [providerId, setProviderId] = useState<string | undefined>(searchParams.get('provider_id') ?? undefined)
  const [modelType, setModelType] = useState<string>(() => searchParams.get('model_type') ?? 'all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  /** 为 true 时仅列出「提供商已启用」下的模型（与后端 provider_enabled_only 一致）；默认仅已启用 */
  const [onlyEnabledProviders, setOnlyEnabledProviders] = useState(true)
  const [providers, setProviders] = useState<ModelProvider[]>([])
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [editing, setEditing] = useState<LlmModel | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [form] = Form.useForm<Record<string, unknown>>()

  useEffect(() => {
    const pid = searchParams.get('provider_id')
    setProviderId(pid ?? undefined)
    const mt = searchParams.get('model_type')
    setModelType(mt ?? 'all')
  }, [searchParams])

  useEffect(() => {
    void fetchAllModelProviders()
      .then(setProviders)
      .catch(() => setProviders([]))
  }, [])

  /** API 拉取 + 当前页模型行推断，避免接口失败或分页导致筛选项为空 */
  const providerFilterOptions = useMemo(() => {
    const m = new Map<string, string>()
    for (const p of providers) {
      m.set(p.id, p.name)
    }
    for (const r of rows) {
      if (!m.has(r.provider_id)) {
        m.set(r.provider_id, r.provider_name)
      }
    }
    return Array.from(m.entries()).map(([id, name]) => ({ id, name }))
  }, [providers, rows])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchLlmModels({
        q: searchApplied.trim() || undefined,
        provider_id: providerId,
        model_type: modelType === 'all' ? undefined : (modelType as ModelType),
        status: statusFilter === 'all' ? 'all' : statusFilter,
        provider_enabled_only: onlyEnabledProviders,
        page,
        page_size: PAGE_SIZE,
      })
      setRows(data.items)
      setTotal(data.meta.total)
    } catch (e) {
      setRows([])
      setTotal(0)
      message.error(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [searchApplied, providerId, modelType, statusFilter, onlyEnabledProviders, page])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    const onRefresh = () => {
      void load()
      void fetchAllModelProviders().then(setProviders).catch(() => {})
    }
    window.addEventListener('app:refresh', onRefresh)
    return () => window.removeEventListener('app:refresh', onRefresh)
  }, [load])

  const setProviderQuery = (v: string | undefined) => {
    setPage(1)
    setProviderId(v)
    const next = new URLSearchParams(searchParams)
    if (v) next.set('provider_id', v)
    else next.delete('provider_id')
    setSearchParams(next)
  }

  const setModelTypeQuery = (v: string) => {
    setPage(1)
    setModelType(v)
    const next = new URLSearchParams(searchParams)
    if (v === 'all') next.delete('model_type')
    else next.set('model_type', v)
    setSearchParams(next)
  }

  const openCreate = () => {
    setEditing(null)
    form.resetFields()
    form.setFieldsValue({ enabled: false, model_type: 'llm', timeout: 30 })
    setDrawerOpen(true)
  }

  const openEdit = (r: LlmModel) => {
    setEditing(r)
    form.setFieldsValue({
      model_code: r.model_code,
      model_name: r.model_name,
      provider_id: r.provider_id,
      model_type: r.model_type,
      endpoint: r.endpoint ?? '',
      timeout: r.timeout,
      enabled: r.enabled,
    })
    setDrawerOpen(true)
  }

  const submit = async () => {
    try {
      const v = await form.validateFields()
      setSubmitting(true)
      if (editing) {
        const patch: Record<string, unknown> = {
          model_name: v.model_name,
          provider_id: v.provider_id,
          model_type: v.model_type,
          endpoint: (v.endpoint as string | undefined) ?? '',
          timeout: v.timeout as number,
          enabled: v.enabled,
        }
        await updateLlmModel(editing.id, patch)
        message.success('已保存')
      } else {
        await createLlmModel({
          model_code: v.model_code as string,
          model_name: v.model_name as string,
          provider_id: v.provider_id as string,
          model_type: v.model_type as string,
          endpoint: (v.endpoint as string | undefined) ?? '',
          timeout: (v.timeout as number | undefined) ?? 30,
          enabled: v.enabled as boolean,
        })
        message.success('已创建')
      }
      setDrawerOpen(false)
      void load()
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) return
      message.error(e instanceof Error ? e.message : '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await deleteLlmModel(id)
      message.success('已删除')
      void load()
    } catch (e) {
      message.error(e instanceof Error ? e.message : '删除失败')
    }
  }

  const columns: ColumnsType<LlmModel> = [
    {
      title: '模型名称',
      dataIndex: 'model_name',
      ellipsis: true,
      width: 160,
      render: (modelName: string, r) => (
        <Tooltip title={`${modelName} · ${r.model_code} · sys_model.id=${r.id}`}>
          <span
            style={{
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontSize: 14,
              color: 'var(--color-text, #1d2129)',
            }}
          >
            <strong>{modelName}</strong>
            <span style={{ color: '#86909c', marginLeft: 4, fontWeight: 400 }}>
              <code style={{ fontSize: 12 }}>{r.model_code}</code>
            </span>
          </span>
        </Tooltip>
      ),
    },
    {
      title: (
        <TableColumnFilterHeader
          label="所属提供商"
          tooltip="按提供商与启用范围筛选"
          active={!!providerId || !onlyEnabledProviders}
        >
          <div style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid var(--color-border-secondary, #f0f0f0)' }}>
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>
              提供商范围
            </Text>
            <Radio.Group
              value={onlyEnabledProviders ? 'enabled_only' : 'include_disabled'}
              onChange={(e) => {
                setPage(1)
                setOnlyEnabledProviders(e.target.value === 'enabled_only')
              }}
            >
              <Space direction="vertical" size={6}>
                <Radio value="enabled_only">仅已启用提供商</Radio>
                <Radio value="include_disabled">含已停用提供商</Radio>
              </Space>
            </Radio.Group>
          </div>
          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>
            指定提供商
          </Text>
          <Radio.Group
            value={providerId ?? '__all__'}
            onChange={(e) => {
              const v = e.target.value as string
              setProviderQuery(v === '__all__' ? undefined : v)
            }}
          >
            <Space direction="vertical" size={6}>
              <Radio value="__all__">全部</Radio>
              {providerFilterOptions.map((p) => (
                <Radio key={p.id} value={p.id}>
                  {p.name}
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </TableColumnFilterHeader>
      ),
      dataIndex: 'provider_name',
      ellipsis: true,
      width: 132,
    },
    {
      title: (
        <TableColumnFilterHeader
          label="类型"
          tooltip="按模型类型筛选"
          active={modelType !== 'all'}
        >
          <Radio.Group
            value={modelType}
            onChange={(e) => {
              setModelTypeQuery(e.target.value as string)
            }}
          >
            <Space direction="vertical" size={6}>
              <Radio value="all">全部</Radio>
              <Radio value="chat">对话</Radio>
              <Radio value="embedding">嵌入</Radio>
              <Radio value="ocr">OCR</Radio>
            </Space>
          </Radio.Group>
        </TableColumnFilterHeader>
      ),
      dataIndex: 'model_type',
      width: 96,
      render: (t: string) => TYPE_LABELS[t] ?? t,
    },
    {
      title: (
        <TableColumnFilterHeader
          label="状态"
          tooltip="按状态筛选"
          active={statusFilter !== 'all'}
        >
          <Radio.Group
            value={statusFilter}
            onChange={(e) => {
              setPage(1)
              setStatusFilter(e.target.value)
            }}
          >
            <Space direction="vertical" size={6}>
              <Radio value="all">全部</Radio>
              <Radio value="enabled">启用</Radio>
              <Radio value="disabled">禁用</Radio>
              <Radio value="error">异常</Radio>
            </Space>
          </Radio.Group>
        </TableColumnFilterHeader>
      ),
      width: 100,
      render: (_, r) => (r.enabled ? <Tag color="success">启用</Tag> : <Tag>禁用</Tag>),
    },
    {
      title: 'endpoint',
      key: 'endpoint',
      width: 140,
      ellipsis: true,
      render: (_, r) => {
        const ep = r.endpoint
        const hint = !ep && r.provider_base_url ? `继承：${r.provider_base_url}` : ep || ''
        return (
          <Tooltip title={hint || '未单独配置，请求地址见提供商 Base URL'}>
            <Text code style={{ fontSize: 12 }}>
              {ep || '—'}
            </Text>
          </Tooltip>
        )
      },
    },
    {
      title: '超时(s)',
      dataIndex: 'timeout',
      width: 72,
      align: 'center',
    },
    {
      title: '操作',
      key: 'actions',
      width: 96,
      align: 'center',
      fixed: 'right',
      render: (_, r) => (
        <Space size={2}>
          <Tooltip title="编辑">
            <Button
              type="text"
              className="agent-list-row-action"
              icon={<EditOutlined />}
              aria-label="编辑"
              onClick={() => openEdit(r)}
            />
          </Tooltip>
          <Tooltip title="测试探测">
            <Button
              type="text"
              className="agent-list-row-action"
              icon={<ThunderboltOutlined />}
              aria-label="测试探测"
              onClick={async () => {
                try {
                  const res = await probeLlmModel(r.id)
                  message[res.ok ? 'success' : 'error'](res.ok ? '探测成功' : res.message || '探测失败')
                  void load()
                } catch (e) {
                  message.error(e instanceof Error ? e.message : '探测失败')
                }
              }}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Popconfirm title="确定删除该模型？" onConfirm={() => void handleDelete(r.id)}>
              <Button type="text" danger icon={<DeleteOutlined />} aria-label="删除" />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ]

  return (
    <div className="agent-list-page">
      <div className="agent-list-table-shell">
        <div className="agent-list-toolbar">
          <div
            className="agent-list-toolbar-left"
            style={{ flex: 1, minWidth: 0, flexWrap: 'wrap', rowGap: 8 }}
          >
            <Button
              type="primary"
              className="agent-list-action-primary"
              icon={<PlusOutlined />}
              onClick={openCreate}
            >
              添加模型
            </Button>
          </div>
          <div className="agent-list-search-wrap">
            <Input
              className="agent-list-search-input"
              prefix={<SearchOutlined className="agent-list-search-icon" />}
              placeholder="搜索 model_name / model_code（回车）"
              allowClear
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
              onPressEnter={() => {
                setPage(1)
                setSearchApplied(searchDraft.trim())
              }}
            />
          </div>
        </div>

        {loading ? (
          <div className="agent-list-table-wrap" style={{ padding: 24 }}>
            <Skeleton active paragraph={{ rows: 8 }} />
          </div>
        ) : (
          <div className="agent-list-table-wrap">
            <div className="agent-list-table-inner">
              <Table<LlmModel>
                className="agent-list-table"
                rowKey="id"
                size="small"
                columns={columns}
                dataSource={rows}
                pagination={false}
                scroll={{ x: 920 }}
                locale={{
                  emptyText: (
                    <div style={{ padding: '48px 0', textAlign: 'center' }}>
                      <Text type="secondary">暂无模型；同一提供商可添加多条</Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={openCreate}
                        >
                          添加模型
                        </Button>
                      </div>
                    </div>
                  ),
                }}
              />
            </div>
            <div className="agent-list-pagination-bar">
              <Pagination
                size="small"
                current={page}
                pageSize={PAGE_SIZE}
                total={total}
                showSizeChanger={false}
                showTotal={(t) => `共 ${t} 条`}
                onChange={(p) => setPage(p)}
              />
            </div>
          </div>
        )}
      </div>

      <Drawer
        title={editing ? '编辑模型' : '添加模型'}
        width={600}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        destroyOnClose
        footer={
          <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
            <Button onClick={() => setDrawerOpen(false)}>取消</Button>
            <Button type="primary" className="agent-list-btn-primary" loading={submitting} onClick={() => void submit()}>
              {editing ? '保存' : '创建'}
            </Button>
          </Space>
        }
      >
        <Form form={form} layout="vertical" className="settings-model-form">
          <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
            字段与表 sys_model 一致（路径 id 为主键）；鉴权密钥在提供商表 sys_model_provider。
          </Text>
          <Collapse
            bordered={false}
            ghost
            className="settings-form-collapse"
            defaultActiveKey={['identity', 'bind', 'conn', 'state']}
            items={[
              {
                key: 'identity',
                label: '模型标识',
                children: (
                  <>
                    <Form.Item
                      name="model_code"
                      label="model_code（厂商侧模型 ID）"
                      rules={
                        editing
                          ? []
                          : [
                              { required: true },
                              { pattern: /^[a-zA-Z0-9_.-]{2,128}$/, message: '2-128 位，如 gpt-4o' },
                            ]
                      }
                    >
                      <Input disabled={!!editing} autoComplete="off" placeholder="如 gpt-4o" />
                    </Form.Item>
                    <Form.Item name="model_name" label="model_name（展示名称）" rules={[{ required: true, max: 128 }]}>
                      <Input placeholder="展示名称" />
                    </Form.Item>
                  </>
                ),
              },
              {
                key: 'bind',
                label: '归属与类型',
                children: (
                  <>
                    <Form.Item
                      name="provider_id"
                      label="所属提供商"
                      rules={[{ required: true }]}
                      extra="对应 sys_model.provider_id，选择已配置的提供商编码"
                    >
                      <Select
                        options={providers.map((p) => ({
                          label: p.enabled ? p.name : `${p.name}（未启用）`,
                          value: p.id,
                        }))}
                      />
                    </Form.Item>
                    <Form.Item name="model_type" label="model_type" rules={[{ required: true }]}>
                      <Select
                        options={[
                          { label: `${TYPE_LABELS.llm}（llm）`, value: 'llm' },
                          { label: `${TYPE_LABELS.embedding}（embedding）`, value: 'embedding' },
                          { label: `${TYPE_LABELS.tts}（tts）`, value: 'tts' },
                          { label: `${TYPE_LABELS.stt}（stt）`, value: 'stt' },
                          { label: `${TYPE_LABELS.ocr}（ocr）`, value: 'ocr' },
                          { label: `${TYPE_LABELS.image}（image）`, value: 'image' },
                        ]}
                      />
                    </Form.Item>
                  </>
                ),
              },
              {
                key: 'conn',
                label: '连接与超时',
                children: (
                  <>
                    <Form.Item
                      name="endpoint"
                      label="endpoint（可选覆盖）"
                      extra="可空；空则未单独配置 endpoint，实际请求基址为提供商 Base URL"
                    >
                      <Input placeholder="可选，完整或相对基址由网关约定" />
                    </Form.Item>
                    <Form.Item name="timeout" label="timeout（秒）" rules={[{ required: true }]}>
                      <InputNumber min={1} max={86400} style={{ width: '100%' }} />
                    </Form.Item>
                  </>
                ),
              },
              {
                key: 'state',
                label: '启用状态',
                children: (
                  <Form.Item
                    name="enabled"
                    label="启用该模型"
                    valuePropName="checked"
                    extra="默认关闭；开启前所属提供商须已配置密钥（本地无需密钥的接入除外）"
                  >
                    <Switch />
                  </Form.Item>
                ),
              },
            ]}
          />
        </Form>
      </Drawer>
    </div>
  )
}
