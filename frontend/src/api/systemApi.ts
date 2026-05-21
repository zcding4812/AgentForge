/**
 * Agent 域：默认系统提示词（按类型拉取正文），见 GET /api/agents/default-prompts/{kind}。
 * 下拉菜单项与后端 ``app.agent.kernel.default_prompts.DEFAULT_PROMPT_ENTRIES`` 的 id/文案应对齐；增删类型时请同步两端。
 */

import { parseApiEnvelope } from './parseEnvelope'

export type DefaultPromptMeta = {
  id: string
  label: string
  description: string
}

export type DefaultPromptDetailData = DefaultPromptMeta & {
  text: string
}

const base = '/api/agents'

/** 工作台根 Agent 可编辑的「流程提示词」模板（基础编排边界由服务端 ``WORKBENCH_ORCHESTRATOR_PROMPT`` 固定注入）。 */
export const WORKBENCH_FLOW_PROMPT_TEMPLATE = `## 本工作区流程约定（可按需修改）

- **领域侧重**：说明本命名空间主要服务哪类用户问题（例如内部知识答疑、研发辅助、运营报表等）。
- **委派与分域**：多类交付（文档 / 验证 / 反馈等）应对应**不同子 Agent**（或先建专职再派）；禁止把根台自身 \`agent_id\` 当作子任务执行方；禁止长期用同一子 Agent 冒充「全角色」。
- **汇总风格**：答复长度、是否结论前置、是否在最终答复中简要列出使用过的子 Agent（仅展示名或 ID）。
- **模型与新建子 Agent**：新建专职子 Agent 前**先** \`workbench_list_sys_models\`，将返回中的 **sys_model_id** 写入 \`create_json\`；平台增删模型后，流程提示词或约定中若写死了旧模型名/旧 id，须随本工具结果**人工或编排侧修订**，勿沿用过期配置。
- **术语与禁区**：团队专有名词口径；哪些内容禁止臆测或须提示用户转人工。
- **收敛与重试**：同一工具/同一子 Agent 相同问法失败后的改派策略；禁止无参数变化的重复调用；何时必须无工具收尾向用户说明阻塞。

以上仅补充平台固定编排职责之外的行为偏好；不得要求根 Agent 绕过 workbench_* 直接执行业务工具或 MCP。`

/** 工作台「默认提示词」下拉：与后端 ``kernel.default_prompts.DEFAULT_PROMPT_ENTRIES`` 顺序、id 一致 */
export const WORKSPACE_DEFAULT_PROMPT_MENU: readonly DefaultPromptMeta[] = [
  {
    id: 'system',
    label: '通用助手',
    description: '平台默认第 1 类 System Prompt；与 MessageBuilder 无自定义 system 时一致。',
  },
  {
    id: 'role_analyst',
    label: '分析 / 咨询',
    description: '侧重结构化分析、假设与结论表述；适合报表、调研、故障初判等场景。',
  },
  {
    id: 'role_coder',
    label: '编程 / 工程',
    description: '侧重可读代码、边界条件与可执行步骤；适合脚本、接口与排错场景。',
  },
  {
    id: 'workbench',
    label: '工作区 / 多 Agent',
    description: '资源管理、子 Agent 编排与结果汇总；与 agent_kind=workbench 配套。',
  },
]

/** GET /api/agents/default-prompts/{kind} */
export async function fetchDefaultPromptByKind(kind: string): Promise<DefaultPromptDetailData> {
  const res = await fetch(`${base}/default-prompts/${encodeURIComponent(kind)}`, {
    headers: { Accept: 'application/json' },
  })
  return parseApiEnvelope<DefaultPromptDetailData>(res)
}
