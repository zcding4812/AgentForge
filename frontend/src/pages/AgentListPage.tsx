import {
  DeleteOutlined,
  EditOutlined,
  FilterFilled,
  FilterOutlined,
  PlusOutlined,
  SearchOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import {
  Button,
  Checkbox,
  Form,
  Input,
  type InputRef,
  Modal,
  Pagination,
  Popover,
  Select,
  Skeleton,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import type { AgentKind } from '../api/agentApi'
import {
  createAgent,
  deleteAgent,
  fetchAgentsList,
  updateAgent,
  type AgentOut,
} from '../api/agentsApi'
import { fetchWorkspaceNamespaces } from '../api/workspaceNamespacesApi'
import { useDebouncedValue } from '../hooks'

import './agentListPage.css'

/** 用户可创建的编排类型（系统工作台 agent_kind=workbench 由 POST /agents/workbench 或种子创建，不在此新建） */
const ORCH_OPTIONS: { label: string; value: AgentKind }[] = [
  { label: '简单对话', value: 'simple_chat' },
  { label: 'ReAct', value: 'react' },
  { label: 'Plan & Execute', value: 'plan_execute' },
]

/** 列表「按类型筛选」含工作台，与列表数据一致（不再默认排除 workbench） */
const AGENT_KIND_FILTER_OPTIONS: { label: string; value: AgentKind }[] = [
  ...ORCH_OPTIONS,
  { label: '工作区（系统）', value: 'workbench' },
]

function agentKindLabel(k: AgentKind): string {
  const m: Record<AgentKind, string> = {
    simple_chat: '简单对话',
    react: 'ReAct',
    plan_execute: 'Plan & Execute',
    workbench: '工作区（系统）',
  }
  return m[k] ?? k
}

import { DEFAULT_PAGE_SIZE, DESCRIPTION_MAX_LENGTH } from '../constants'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export type AgentRow = AgentOut

export function AgentListPage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const searchInputRef = useRef<InputRef | null>(null)

  const [loading, setLoading] = useState(true)
  const [agents, setAgents] = useState<AgentOut[]>([])
  const [total, setTotal] = useState(0)
  const [searchDraft, setSearchDraft] = useState('')
  const debouncedSearch = useDebouncedValue(searchDraft.trim())
  const [filterTypes, setFilterTypes] = useState<AgentKind[]>([])
  const [page, setPage] = useState(1)

  /** 与 URL `?workspace_namespace=` 同步，便于从工作台链入「在本命名空间创建 Agent」 */
  const workspaceNamespaceFilter = (searchParams.get('workspace_namespace') ?? '').trim()

  const [namespaceSlugs, setNamespaceSlugs] = useState<string[]>([])
  const [namespacesLoading, setNamespacesLoading] = useState(true)

  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [createSubmitting, setCreateSubmitting] = useState(false)
  const [form] = Form.useForm<{
    name: string
    agent_kind: AgentKind
    workspace_namespace: string
  }>()

  const [editOpen, setEditOpen] = useState(false)
  const [editSubmitting, setEditSubmitting] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editForm] = Form.useForm<{ name: string; description: string; agent_kind: AgentKind }>()

  useEffect(() => {
    setPage(1)
  }, [debouncedSearch, filterTypes])

  useEffect(() => {
    let cancelled = false
    setNamespacesLoading(true)
    void fetchWorkspaceNamespaces()
      .then((data) => {
        if (!cancelled) setNamespaceSlugs(data.items.map((x) => x.slug))
      })
      .catch(() => {
        if (!cancelled) setNamespaceSlugs([])
      })
      .finally(() => {
        if (!cancelled) setNamespacesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const namespaceFilterOptions = useMemo(() => {
    const slugs = [...namespaceSlugs]
    const set = new Set(slugs)
    if (workspaceNamespaceFilter && !set.has(workspaceNamespaceFilter)) {
      slugs.unshift(workspaceNamespaceFilter)
    }
    return slugs.map((s) => ({ label: s, value: s }))
  }, [namespaceSlugs, workspaceNamespaceFilter])

  /** 新建弹窗：仅从已存在命名空间中选择（不新建命名空间、不在此初始化工作台） */
  const createModalNamespaceOptions = useMemo(
    () =>
      [...namespaceSlugs]
        .map((s) => (s || '').trim())
        .filter(Boolean)
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
        .map((slug) => ({ label: slug, value: slug })),
    [namespaceSlugs],
  )

  /** 列表加载较慢时：弹窗已打开且尚未选中命名空间，补全为 URL 对齐项或列表首项 */
  useEffect(() => {
    if (!createModalOpen || namespacesLoading || namespaceSlugs.length === 0) return
    const cur = form.getFieldValue('workspace_namespace') as string | undefined
    if (typeof cur === 'string' && cur.trim()) return
    const f = workspaceNamespaceFilter.trim()
    const match = f && namespaceSlugs.find((s) => s.toLowerCase() === f.toLowerCase())
    form.setFieldValue('workspace_namespace', match ?? namespaceSlugs[0])
  }, [
    createModalOpen,
    namespacesLoading,
    namespaceSlugs,
    workspaceNamespaceFilter,
    form,
  ])

  const setWorkspaceNamespaceParam = useCallback(
    (slug: string | null) => {
      const t = (slug ?? '').trim()
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev)
          if (t) p.set('workspace_namespace', t)
          else p.delete('workspace_namespace')
          return p
        },
        { replace: true },
      )
      setPage(1)
    },
    [setSearchParams],
  )

  const loadList = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchAgentsList({
        page,
        page_size: PAGE_SIZE,
        q: debouncedSearch || undefined,
        agent_kind: filterTypes.length ? filterTypes : undefined,
        workspace_namespace: workspaceNamespaceFilter || undefined,
      })
      setAgents(data.items)
      setTotal(data.total)
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载失败')
      setAgents([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [page, debouncedSearch, filterTypes, workspaceNamespaceFilter])

  useEffect(() => {
    void loadList()
  }, [loadList])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        searchInputRef.current?.input?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const submitCreate = useCallback(async () => {
    try {
      const v = await form.validateFields()
      const ns = (v.workspace_namespace ?? '').trim()
      setCreateSubmitting(true)

      const out = await createAgent({
        name: v.name.trim(),
        description: null,
        agent_kind: v.agent_kind,
        workspace_namespace: ns,
        system_prompt: null,
      })
      message.success('已创建')
      form.resetFields()
      setCreateModalOpen(false)
      navigate(`/agents/${out.id}`)
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) return
      message.error(e instanceof Error ? e.message : '创建失败')
    } finally {
      setCreateSubmitting(false)
    }
  }, [form, navigate])

  const submitEdit = async () => {
    if (editingId == null) return
    try {
      const v = await editForm.validateFields()
      setEditSubmitting(true)
      await updateAgent(editingId, {
        name: v.name.trim(),
        description: (v.description || '').trim() || null,
        agent_kind: v.agent_kind,
      })
      message.success('已保存')
      setEditOpen(false)
      setEditingId(null)
      await loadList()
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) {
        throw e
      }
      message.error(e instanceof Error ? e.message : '保存失败')
      throw e instanceof Error ? e : new Error('保存失败')
    } finally {
      setEditSubmitting(false)
    }
  }

  const openEditModal = useCallback(
    (row: AgentOut) => {
      setEditingId(row.id)
      editForm.setFieldsValue({
        name: row.name,
        description: row.description ?? '',
        agent_kind: row.agent_kind,
      })
      setEditOpen(true)
    },
    [editForm],
  )

  const openCreateModal = useCallback(() => {
    const f = workspaceNamespaceFilter.trim()
    const match =
      f && namespaceSlugs.find((s) => s.toLowerCase() === f.toLowerCase())
    const initialNs = match ?? namespaceSlugs[0] ?? undefined
    form.setFieldsValue({
      ...(initialNs !== undefined ? { workspace_namespace: initialNs } : {}),
    })
    setCreateModalOpen(true)
  }, [form, workspaceNamespaceFilter, namespaceSlugs])

  const confirmDelete = useCallback(
    (row: AgentOut) => {
      const isWorkbench = row.agent_kind === 'workbench'
      Modal.confirm({
        title: isWorkbench ? '确定删除该工作区编排 Agent？' : '确定删除该 Agent？',
        content: isWorkbench
          ? `将删除整个命名空间「${row.workspace_namespace || '—'}」及其下所有 Agent、知识库与相关会话，且不可恢复。`
          : '删除后不可恢复；该 Agent 下的对话会话将一并删除。',
        okText: '删除',
        okType: 'danger',
        cancelText: '取消',
        onOk: async () => {
          try {
            await deleteAgent(row.id)
            message.success('已删除')
            await loadList()
          } catch (e) {
            message.error(e instanceof Error ? e.message : '删除失败')
            throw e
          }
        },
      })
    },
    [loadList],
  )

  const columns: ColumnsType<AgentOut> = useMemo(
    () => [
      {
        title: '名称',
        dataIndex: 'name',
        ellipsis: true,
        render: (name: string) => (
          <Tooltip title={name}>
            <strong style={{ color: '#1d2129', fontWeight: 600 }}>{name}</strong>
          </Tooltip>
        ),
      },
      {
        title: '描述',
        dataIndex: 'description',
        ellipsis: true,
        render: (text: string | null) => {
          const t = text || '—'
          return (
            <Tooltip title={t}>
              <span style={{ color: '#4e5969' }}>{t}</span>
            </Tooltip>
          )
        },
      },
      {
        title: (
          <div className="agent-list-type-col-header">
            <div className="agent-list-type-col-label-row">
              <span className="agent-list-type-col-label">Agent 类型</span>
              <Popover
                trigger="click"
                placement="bottomRight"
                getPopupContainer={() => document.body}
                content={
                  <div className="agent-list-type-popover">
                    <Checkbox.Group
                      className="agent-list-type-checkbox-group"
                      value={filterTypes}
                      onChange={(v) => setFilterTypes(v as AgentKind[])}
                    >
                      {AGENT_KIND_FILTER_OPTIONS.map((o) => (
                        <Checkbox key={o.value} value={o.value}>
                          {o.label}
                        </Checkbox>
                      ))}
                    </Checkbox.Group>
                    {filterTypes.length > 0 ? (
                      <Button type="link" size="small" className="agent-list-type-clear" onClick={() => setFilterTypes([])}>
                        清空
                      </Button>
                    ) : null}
                  </div>
                }
              >
                <Tooltip title="按类型筛选">
                  <button type="button" className="agent-list-type-icon-trigger" aria-label="按类型筛选">
                    {filterTypes.length > 0 ? (
                      <FilterFilled className="agent-list-type-filter-icon agent-list-type-filter-icon--active" />
                    ) : (
                      <FilterOutlined className="agent-list-type-filter-icon" />
                    )}
                  </button>
                </Tooltip>
              </Popover>
            </div>
          </div>
        ),
        dataIndex: 'agent_kind',
        width: 168,
        render: (_: unknown, row) => <span>{agentKindLabel(row.agent_kind)}</span>,
      },
      {
        title: (
          <div className="agent-list-type-col-header">
            <div className="agent-list-type-col-label-row">
              <span className="agent-list-type-col-label">命名空间</span>
              <Popover
                trigger="click"
                placement="bottomRight"
                getPopupContainer={() => document.body}
                content={
                  <div className="agent-list-type-popover">
                    <Select
                      showSearch
                      allowClear
                      placeholder="全部命名空间"
                      loading={namespacesLoading}
                      style={{ width: 240 }}
                      options={namespaceFilterOptions}
                      value={workspaceNamespaceFilter || undefined}
                      onChange={(v) => setWorkspaceNamespaceParam(v ?? null)}
                      filterOption={(input, opt) =>
                        String(opt?.label ?? '')
                          .toLowerCase()
                          .includes(input.toLowerCase())
                      }
                      getPopupContainer={() => document.body}
                    />
                  </div>
                }
              >
                <Tooltip title="按命名空间筛选">
                  <button
                    type="button"
                    className="agent-list-type-icon-trigger"
                    aria-label="按命名空间筛选"
                  >
                    {workspaceNamespaceFilter ? (
                      <FilterFilled className="agent-list-type-filter-icon agent-list-type-filter-icon--active" />
                    ) : (
                      <FilterOutlined className="agent-list-type-filter-icon" />
                    )}
                  </button>
                </Tooltip>
              </Popover>
            </div>
          </div>
        ),
        dataIndex: 'workspace_namespace',
        width: 148,
        ellipsis: true,
        render: (ns: string | undefined) => (
          <Tooltip title={ns || 'default'}>
            <span className="agent-list-ns-cell">{ns?.trim() || 'default'}</span>
          </Tooltip>
        ),
      },
      {
        title: '状态',
        dataIndex: 'status',
        width: 96,
        render: (s: string) => (
          <Tag color={s === 'active' ? 'blue' : 'default'}>{s === 'active' ? '启用' : s}</Tag>
        ),
      },
      {
        title: '操作',
        key: 'actions',
        width: 200,
        align: 'center',
        render: (_, row) => (
          <Space size={4} className="agent-list-actions-cell">
            <Tooltip title="配置 Agent（编排、工具等）">
              <Button
                type="text"
                className="agent-list-row-action"
                icon={<SettingOutlined />}
                aria-label="配置 Agent"
                onClick={(e) => {
                  e.stopPropagation()
                  navigate(`/agents/${row.id}`)
                }}
              />
            </Tooltip>
            <Tooltip title="编辑">
              <Button
                type="text"
                className="agent-list-row-action"
                icon={<EditOutlined />}
                aria-label="编辑"
                onClick={(e) => {
                  e.stopPropagation()
                  openEditModal(row)
                }}
              />
            </Tooltip>
            <Tooltip title="删除">
              <Button
                type="text"
                danger
                className="agent-list-row-action"
                icon={<DeleteOutlined />}
                aria-label="删除"
                onClick={(e) => {
                  e.stopPropagation()
                  confirmDelete(row)
                }}
              />
            </Tooltip>
          </Space>
        ),
      },
    ],
    [
      confirmDelete,
      filterTypes,
      navigate,
      namespaceFilterOptions,
      namespacesLoading,
      openEditModal,
      setWorkspaceNamespaceParam,
      workspaceNamespaceFilter,
    ],
  )

  return (
    <div className="agent-list-page">
      <div className="agent-list-table-shell">
        <div className="agent-list-toolbar">
          <div className="agent-list-toolbar-left">
            <Button
              type="primary"
              className="agent-list-action-primary"
              icon={<PlusOutlined />}
              onClick={() => openCreateModal()}
            >
              新建 Agent
            </Button>
          </div>
          <div className="agent-list-search-wrap">
            <Input
              ref={searchInputRef}
              className="agent-list-search-input"
              prefix={<SearchOutlined className="agent-list-search-icon" />}
              placeholder="搜索名称、描述（Ctrl+K）"
              allowClear
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
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
              <Table<AgentOut>
                className="agent-list-table"
                rowKey="id"
                size="small"
                columns={columns}
                dataSource={agents}
                pagination={false}
                locale={{
                  emptyText: (
                    <div style={{ padding: '48px 0', textAlign: 'center' }}>
                      <Typography.Text type="secondary">暂无 Agent 数据</Typography.Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={() => openCreateModal()}
                        >
                          新建 Agent
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

      <Modal
        title="新建 Agent"
        open={createModalOpen}
        width={480}
        destroyOnClose
        maskClosable={!createSubmitting}
        onCancel={() => {
          if (createSubmitting) return
          setCreateModalOpen(false)
        }}
        footer={
          <Space style={{ justifyContent: 'flex-end', width: '100%' }}>
            <Button onClick={() => setCreateModalOpen(false)} disabled={createSubmitting}>
              取消
            </Button>
            <Button
              type="primary"
              className="agent-list-btn-primary"
              loading={createSubmitting}
              disabled={!namespacesLoading && namespaceSlugs.length === 0}
              onClick={() => void submitCreate()}
            >
              创建并进入工作区
            </Button>
          </Space>
        }
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            agent_kind: 'simple_chat' satisfies AgentKind,
          }}
        >
          <Form.Item
            name="workspace_namespace"
            label="工作区命名空间"
            normalize={(v) => (typeof v === 'string' ? v.trim() : v)}
            rules={[{ required: true, message: '请选择命名空间' }]}
          >
            <Select
              className="agent-list-create-ns-select"
              options={createModalNamespaceOptions}
              placeholder={
                namespacesLoading
                  ? '加载命名空间…'
                  : namespaceSlugs.length
                    ? '请选择命名空间'
                    : '暂无可用命名空间'
              }
              loading={namespacesLoading}
              disabled={!namespacesLoading && namespaceSlugs.length === 0}
              showSearch
              optionFilterProp="label"
            />
          </Form.Item>
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="例如：运维总控" maxLength={64} showCount />
          </Form.Item>
          <Form.Item name="agent_kind" label="编排类型" rules={[{ required: true, message: '请选择编排类型' }]}>
            <Select options={ORCH_OPTIONS} placeholder="选择编排策略" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="编辑 Agent"
        open={editOpen}
        width={480}
        destroyOnClose
        confirmLoading={editSubmitting}
        okText="保存"
        cancelText="取消"
        onCancel={() => {
          setEditOpen(false)
          setEditingId(null)
        }}
        onOk={() => submitEdit()}
      >
        <Form form={editForm} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="例如：运维总控" maxLength={128} showCount />
          </Form.Item>
          <Form.Item name="agent_kind" label="编排类型" rules={[{ required: true, message: '请选择编排类型' }]}>
            <Select options={AGENT_KIND_FILTER_OPTIONS} placeholder="选择编排策略" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={4} placeholder="可选：用途与使用场景" maxLength={DESCRIPTION_MAX_LENGTH} showCount />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
