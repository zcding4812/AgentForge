import { FilterFilled, FilterOutlined } from '@ant-design/icons'
import { Popover, Tooltip } from 'antd'
import type { ReactNode } from 'react'

/** 列头：标题 + 筛选图标（Popover 内选条件；与模型列表、提供商列表一致） */
export function TableColumnFilterHeader({
  label,
  tooltip,
  active,
  children,
}: {
  label: string
  tooltip: string
  active: boolean
  children: ReactNode
}) {
  return (
    <div className="provider-col-header provider-col-header--inline">
      <span className="provider-col-header-label">{label}</span>
      <Popover
        trigger="click"
        placement="bottomRight"
        destroyOnHidden
        content={<div style={{ minWidth: 220, maxHeight: 320, overflow: 'auto' }}>{children}</div>}
      >
        <Tooltip title={tooltip}>
          <button type="button" className="agent-list-type-icon-trigger" aria-label={tooltip}>
            {active ? (
              <FilterFilled className="agent-list-type-filter-icon agent-list-type-filter-icon--active" />
            ) : (
              <FilterOutlined className="agent-list-type-filter-icon" />
            )}
          </button>
        </Tooltip>
      </Popover>
    </div>
  )
}
