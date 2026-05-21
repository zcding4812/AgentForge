import { Form, Input, Modal, message } from 'antd'
import { useEffect, useState } from 'react'

import { createWorkbenchWorkspace, type AgentOut } from '../api/agentsApi'
import { DESCRIPTION_MAX_LENGTH } from '../constants'

type Props = {
  open: boolean
  onCancel: () => void
  /** 创建成功：返回新入库的工作台 Agent */
  onCreated: (agent: AgentOut) => void
}

/**
 * 调用 POST /api/agents/workbench，在空命名空间下创建唯一 workbench 行。
 */
export function CreateWorkbenchWorkspaceModal({ open, onCancel, onCreated }: Props) {
  const [form] = Form.useForm<{
    workspace_namespace: string
    name: string
    description?: string
  }>()
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (open) {
      form.setFieldsValue({ name: '工作区', workspace_namespace: '', description: '' })
    }
  }, [open, form])

  const submit = async () => {
    try {
      const v = await form.validateFields()
      setSubmitting(true)
      const out = await createWorkbenchWorkspace({
        workspace_namespace: v.workspace_namespace.trim(),
        name: (v.name || '工作区').trim(),
        description: (v.description || '').trim() || null,
      })
      onCreated(out)
      onCancel()
      form.resetFields()
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) return
      message.error(e instanceof Error ? e.message : '创建失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      title="新建工作区"
      open={open}
      onCancel={onCancel}
      onOk={() => void submit()}
      confirmLoading={submitting}
      okText="创建并进入"
      destroyOnClose
      width={480}
    >
      <Form form={form} layout="vertical" requiredMark={false}>
        <Form.Item
          name="workspace_namespace"
          label="命名空间"
          extra="与知识库、子 Agent 隔离；同一命名空间仅能有一条编排 Agent（workbench）。仅字母、数字、下划线、连字符，勿含空格。"
          rules={[
            { required: true, message: '请输入命名空间' },
            { max: 64, type: 'string' },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message: '仅允许字母、数字、下划线与连字符',
            },
          ]}
        >
          <Input placeholder="例如 finance、lab_01" maxLength={64} autoComplete="off" />
        </Form.Item>
        <Form.Item name="name" label="编排名称" rules={[{ required: true, message: '请输入名称' }]}>
          <Input placeholder="工作区" maxLength={128} />
        </Form.Item>
        <Form.Item name="description" label="描述">
          <Input.TextArea rows={2} placeholder="可选：用途说明" maxLength={DESCRIPTION_MAX_LENGTH} showCount />
        </Form.Item>
      </Form>
    </Modal>
  )
}
