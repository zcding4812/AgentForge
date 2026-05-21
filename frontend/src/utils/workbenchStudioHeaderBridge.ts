/** 左侧工作台页向顶栏同步「工作室 → 像素工作台」入口显隐与命名空间。 */

export const WORKBENCH_STUDIO_CTA_EVENT = 'app:workbench-studio-cta'

export type WorkbenchStudioCtaDetail = {
  show: boolean
  namespace: string
}
