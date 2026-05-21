import {
  AppstoreOutlined,
  BuildOutlined,
  CompassOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  HomeOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MonitorOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { Button, Layout, Menu, Space, Tag, Tooltip, Typography } from 'antd'
import type { MenuProps } from 'antd'
import { useMemo, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'

import './AppLayout.css'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

const mainMenuItems: MenuProps['items'] = [
  { key: '/workbench', icon: <HomeOutlined />, label: '工作区' },
  { key: '/agents', icon: <AppstoreOutlined />, label: 'Agent 中心' },
  {
    key: 'resources-group',
    icon: <DatabaseOutlined />,
    label: '资源管理',
    children: [
      { key: '/resources/knowledge', label: '知识库' },
      { key: '/resources/tools', label: '工具库' },
    ],
  },
  {
    key: 'monitor-group',
    icon: <MonitorOutlined />,
    label: '监控',
    children: [
      { key: '/monitor/system', label: '系统监控' },
      { key: '/monitor/tracing', label: '链路追踪' },
    ],
  },
  {
    key: 'settings-group',
    icon: <SettingOutlined />,
    label: '系统设置',
    children: [
      { key: 'settings-models', label: '模型列表' },
      { key: 'settings-providers', label: '模型提供商' },
    ],
  },
]

const routeMeta: Record<string, { title: string; subtitle: string }> = {
  '/workbench': {
    title: '工作区',
    subtitle: '列表管理命名空间与系统编排 Agent，进入后对话与编排子 Agent',
  },
  '/workbench/:workspaceNamespace': {
    title: '工作区',
    subtitle: '按命名空间进入，编排同区子 Agent',
  },
  '/agents': {
    title: 'Agent 中心',
    subtitle: '创建、管理并进入双栏工作区',
  },
  '/agents/:agentId': {
    title: 'Agent 配置与试聊',
    subtitle: '侧栏顶栏为入库 Agent 名称与类型，左侧为参数与能力配置，右侧为对话',
  },
  '/resources/knowledge': {
    title: '知识库管理',
    subtitle: '列表检索与五步运维入口',
  },
  '/resources/knowledge/:kbId': {
    title: '知识库详情',
    subtitle: '文档上传到索引验证的完整流程',
  },
  '/resources/tools': {
    title: '工具库管理',
    subtitle: '统一维护工具目录与同步状态',
  },
  '/monitor/system': {
    title: '系统监控',
    subtitle: '服务健康、运行概览与关键指标',
  },
  '/monitor/tracing': {
    title: '链路追踪',
    subtitle: '按请求链路排查耗时与异常节点',
  },
  '/monitor/tracing/:traceId': {
    title: '链路详情',
    subtitle: '单条链路的执行时序与节点耗时分布',
  },
  '/settings': {
    title: '系统设置',
    subtitle: '模型与提供商配置',
  },
  '/settings/models': {
    title: '模型配置',
    subtitle: '配置可用模型，与提供商关联；同一提供商可添加多个模型',
  },
  '/settings/providers': {
    title: '提供商与模型',
    subtitle: '维护提供商、连接与密钥；模型在「模型列表」中挂到提供商',
  },
}

function getMeta(pathname: string) {
  if (pathname === '/workbench') {
    return routeMeta['/workbench']
  }
  if (pathname.startsWith('/workbench/')) {
    return routeMeta['/workbench/:workspaceNamespace']
  }
  if (pathname.startsWith('/agents/')) return routeMeta['/agents/:agentId']
  if (pathname.startsWith('/resources/knowledge/')) {
    return routeMeta['/resources/knowledge/:kbId']
  }
  if (pathname.startsWith('/monitor/tracing/')) {
    return routeMeta['/monitor/tracing/:traceId']
  }
  return routeMeta[pathname] ?? { title: 'AI Agents', subtitle: '运维型后台控制台' }
}

export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const meta = useMemo(() => getMeta(location.pathname), [location.pathname])

  /** 仅列表页 ``/agents`` 可按查询参数展示命名空间；进入 ``/agents/:id`` 配置工作区时侧栏保持「Agent 中心」不追加（xxx）。 */
  const agentCenterMenuLabel = useMemo(() => {
    let ns = ''
    if (location.pathname === '/agents') {
      const raw = new URLSearchParams(location.search).get('workspace_namespace')?.trim() ?? ''
      if (raw) {
        try {
          ns = decodeURIComponent(raw).trim()
        } catch {
          ns = raw
        }
      }
    }
    return ns ? `Agent 中心（${ns}）` : 'Agent 中心'
  }, [location.pathname, location.search])

  const mainMenuItemsResolved = useMemo((): MenuProps['items'] => {
    return (mainMenuItems ?? []).map((item) => {
      if (item && typeof item === 'object' && 'key' in item && item.key === '/agents') {
        return { ...item, label: agentCenterMenuLabel }
      }
      return item
    })
  }, [agentCenterMenuLabel])

  const mainSelectedKeys = useMemo(() => {
    if (location.pathname === '/settings/models') {
      return ['settings-models']
    }
    if (location.pathname === '/settings/providers') {
      return ['settings-providers']
    }
    if (location.pathname === '/workbench' || location.pathname.startsWith('/workbench/')) {
      return ['/workbench']
    }
    if (location.pathname.startsWith('/agents/')) return ['/agents']
    if (location.pathname.startsWith('/resources/knowledge')) return ['/resources/knowledge']
    if (location.pathname.startsWith('/monitor/system')) return ['/monitor/system']
    if (location.pathname.startsWith('/monitor/tracing')) return ['/monitor/tracing']
    return [location.pathname]
  }, [location.pathname])

  return (
    <Layout className="app-shell">
      <Sider
        collapsible
        trigger={null}
        collapsed={collapsed}
        width={220}
        collapsedWidth={64}
        className="app-sider"
      >
        <div className="sider-brand">
          <div className="sider-brand-text">
            <BuildOutlined />
            {!collapsed && <span>AI Agents</span>}
          </div>
          <Tooltip title={collapsed ? '展开侧栏' : '折叠侧栏'}>
            <Button
              type="text"
              size="small"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed((v) => !v)}
            />
          </Tooltip>
        </div>
        <div className="sider-nav">
          <Menu
            mode="inline"
            items={mainMenuItemsResolved}
            selectedKeys={mainSelectedKeys}
            defaultOpenKeys={['resources-group', 'monitor-group', 'settings-group']}
            onClick={({ key }) => {
              if (key === 'settings-models') {
                navigate('/settings/models')
                return
              }
              if (key === 'settings-providers') {
                navigate('/settings/providers')
                return
              }
              if (typeof key === 'string' && key.startsWith('/')) {
                navigate(key)
              }
            }}
          />
        </div>
      </Sider>
      <Layout>
        <Header className="app-header">
          <div className="header-left">
            <Title level={4} className="header-title">
              {meta.title}
            </Title>
            <Text className="header-subtitle">{meta.subtitle}</Text>
          </div>
          <Space align="center">
            <Tooltip title="工作区对话落地页（选择命名空间后开聊）">
              <Tag
                color="blue"
                bordered={false}
                className="app-header-home-tag"
                role="button"
                tabIndex={0}
                onClick={() => navigate('/studio')}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    navigate('/studio')
                  }
                }}
              >
                <CompassOutlined className="app-header-home-tag__icon" aria-hidden />
                首页
              </Tag>
            </Tooltip>
            <Tooltip title="接口文档">
              <Button
                href="/docs"
                target="_blank"
                rel="noreferrer"
                shape="circle"
                icon={<FileTextOutlined />}
              />
            </Tooltip>
          </Space>
        </Header>
        <Content className="app-content">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
