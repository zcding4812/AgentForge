/** 工作台独立路由（与 `/agents/:id` 区分），以命名空间为路径段。 */

/** ``/workbench/:ns?wb_config=1``：hub 落地后展开编排侧栏并移除参数（可手动深链）。 */
export const WORKBENCH_OPEN_CONFIG_QUERY_PARAM = 'wb_config'

/** 将命名空间编码为路径段（保留 default 等常见名）。 */
export function workbenchPathForNamespace(namespace: string): string {
  const ns = (namespace || 'default').trim() || 'default'
  return `/workbench/${encodeURIComponent(ns)}`
}

