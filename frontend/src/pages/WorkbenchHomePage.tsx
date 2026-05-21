import { LoginOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons'
import {
  Button,
  Input,
  type InputRef,
  Pagination,
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
import { useNavigate } from 'react-router-dom'

import type { AgentOut } from '../api/agentsApi'
import { CreateWorkbenchWorkspaceModal } from '../components/CreateWorkbenchWorkspaceModal'
import { useWorkbench } from '../contexts/useWorkbench'
import { workbenchPathForNamespace } from '../utils/workbenchRoutes'

import './agentListPage.css'

import { DEFAULT_PAGE_SIZE } from '../constants'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

const WORKBENCH_TYPE_LABEL = '工作区（系统）'

/**
 * 工作区列表（路由 ``/workbench``）：与 Agent 中心同构；新建即创建 ``agent_kind=workbench`` 的编排 Agent。
 */
export function WorkbenchHomePage() {
  const navigate = useNavigate()
  const searchInputRef = useRef<InputRef | null>(null)
  const { workbenchAgents, loading, error, refresh, selectWorkbenchAgent } = useWorkbench()

  const [searchDraft, setSearchDraft] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [page, setPage] = useState(1)
  const [createOpen, setCreateOpen] = useState(false)

  useEffect(() => {
    const t = window.setTimeout(() => {
      const next = searchDraft.trim()
      setDebouncedSearch((prev) => {
        if (prev !== next) setPage(1)
        return next
      })
    }, 350)
    return () => window.clearTimeout(t)
  }, [searchDraft])

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

  const filteredAgents = useMemo(() => {
    const q = debouncedSearch.toLowerCase()
    if (!q) return workbenchAgents
    return workbenchAgents.filter((a) => {
      const ns = (a.workspace_namespace ?? 'default').trim()
      const parts = [
        a.name,
        a.description ?? '',
        ns,
        String(a.id),
      ]
      return parts.some((p) => p.toLowerCase().includes(q))
    })
  }, [workbenchAgents, debouncedSearch])

  const pagedAgents = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE
    return filteredAgents.slice(start, start + PAGE_SIZE)
  }, [filteredAgents, page])

  const enterWorkspace = useCallback(
    (row: AgentOut) => {
      selectWorkbenchAgent(row.id)
      navigate(workbenchPathForNamespace(row.workspace_namespace || 'default'))
    },
    [navigate, selectWorkbenchAgent],
  )

  const onWorkspaceCreated = useCallback(
    (agent: AgentOut) => {
      void refresh().then(() => {
        selectWorkbenchAgent(agent.id)
        navigate(workbenchPathForNamespace(agent.workspace_namespace || 'default'))
        message.success('已创建工作区（系统）并进入编排')
      })
    },
    [navigate, refresh, selectWorkbenchAgent],
  )

  const columns: ColumnsType<AgentOut> = useMemo(
    () => [
      {
        title: '命名空间',
        dataIndex: 'workspace_namespace',
        width: 148,
        ellipsis: true,
        render: (ns: string | undefined) => (
          <Tooltip title={ns?.trim() || 'default'}>
            <span className="agent-list-ns-cell">{ns?.trim() || 'default'}</span>
          </Tooltip>
        ),
      },
      {
        title: '编排名称',
        dataIndex: 'name',
        ellipsis: true,
        render: (name: string) => (
          <Tooltip title={name}>
            <strong style={{ color: '#1d2129', fontWeight: 600 }}>{name}</strong>
          </Tooltip>
        ),
      },
      {
        title: '类型',
        dataIndex: 'agent_kind',
        width: 128,
        render: () => <Tag color="cyan">{WORKBENCH_TYPE_LABEL}</Tag>,
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
        title: 'ID',
        dataIndex: 'id',
        width: 72,
        align: 'center',
        render: (id: number) => (
          <span style={{ color: '#86909c', fontVariantNumeric: 'tabular-nums' }}>{id}</span>
        ),
      },
      {
        title: '操作',
        key: 'actions',
        width: 88,
        align: 'center',
        render: (_: unknown, row) => (
          <Space size={4} className="agent-list-actions-cell">
            <Tooltip title="进入工作区">
              <Button
                type="primary"
                size="small"
                className="agent-list-btn-primary"
                icon={<LoginOutlined />}
                aria-label="进入工作区"
                onClick={(e) => {
                  e.stopPropagation()
                  enterWorkspace(row)
                }}
              />
            </Tooltip>
          </Space>
        ),
      },
    ],
    [enterWorkspace],
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
              onClick={() => setCreateOpen(true)}
            >
              新建工作区
            </Button>
          </div>
          <div className="agent-list-search-wrap">
            <Input
              ref={searchInputRef}
              className="agent-list-search-input"
              prefix={<SearchOutlined className="agent-list-search-icon" />}
              placeholder="搜索命名空间、名称、描述、ID（Ctrl+K）"
              allowClear
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
            />
          </div>
        </div>

        {error ? (
          <div className="agent-list-table-wrap" style={{ padding: 24 }}>
            <Typography.Text type="danger">{error}</Typography.Text>
          </div>
        ) : loading ? (
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
                dataSource={pagedAgents}
                pagination={false}
                onRow={(row) => ({
                  onClick: () => enterWorkspace(row),
                  style: { cursor: 'pointer' },
                })}
                locale={{
                  emptyText: (
                    <div style={{ padding: '48px 0', textAlign: 'center' }}>
                      <Typography.Text type="secondary">暂无工作区</Typography.Text>
                      <div style={{ marginTop: 16 }}>
                        <Button
                          type="primary"
                          className="agent-list-btn-primary"
                          icon={<PlusOutlined />}
                          onClick={() => setCreateOpen(true)}
                        >
                          新建工作区
                        </Button>
                      </div>
                      <Typography.Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0 }}>
                        将在新命名空间下创建系统工作区（{WORKBENCH_TYPE_LABEL}）编排 Agent。
                      </Typography.Paragraph>
                    </div>
                  ),
                }}
              />
            </div>
            {filteredAgents.length > 0 ? (
              <div className="agent-list-pagination-bar">
                <Pagination
                  size="small"
                  current={page}
                  pageSize={PAGE_SIZE}
                  total={filteredAgents.length}
                  showSizeChanger={false}
                  showTotal={(t) => `共 ${t} 条`}
                  onChange={(p) => setPage(p)}
                />
              </div>
            ) : null}
          </div>
        )}
      </div>

      <CreateWorkbenchWorkspaceModal
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onCreated={onWorkspaceCreated}
      />
    </div>
  )
}
