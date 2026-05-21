/**
 * 工作台等场景：react-markdown 内识别 Mermaid（```mermaid 或裸 graph LR 等）并渲染为 SVG。
 */
import mermaid from 'mermaid'
import { Children, isValidElement, useEffect, useId, useRef } from 'react'
import type { Components } from 'react-markdown'

const MERMAID_LEADING =
  /^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram-v2|stateDiagram|erDiagram|journey|gantt|pie|mindmap|timeline|sankey-beta|gitGraph|C4Context|block-beta)\b/i

let mermaidInitialized = false

function ensureMermaidInit() {
  if (mermaidInitialized) return
  mermaid.initialize({
    startOnLoad: false,
    theme: 'neutral',
    // 允许节点标签内 <br> 等（模型常见）；仍拦截脚本类注入
    securityLevel: 'antiscript',
    fontFamily: 'inherit',
  })
  mermaidInitialized = true
}

/** Mermaid 11+ 在部分语法错误时不抛异常，而是返回含错误文案的 SVG。 */
function looksLikeMermaidRenderError(svg: string): boolean {
  return (
    /Syntax error in text/i.test(svg) ||
    /Parse error on line/i.test(svg) ||
    /No diagram type detected/i.test(svg)
  )
}

function mountMermaidFallback(container: HTMLDivElement, source: string): void {
  container.replaceChildren()
  const wrap = document.createElement('div')
  wrap.className = 'agent-workspace-msg__mermaid-fallback-wrap'
  wrap.setAttribute('role', 'status')
  const hint = document.createElement('p')
  hint.className = 'agent-workspace-msg__mermaid-fallback-hint'
  hint.textContent =
    '无法渲染为 Mermaid 图（语法有误或与当前库不兼容）。以下为原始内容，可复制后修正：'
  const pre = document.createElement('pre')
  pre.className = 'agent-workspace-msg__mermaid-fallback'
  pre.textContent = source
  wrap.appendChild(hint)
  wrap.appendChild(pre)
  container.appendChild(wrap)
}

export function MermaidDiagram({ definition }: { definition: string }) {
  const reactId = useId().replace(/:/g, '')
  const containerRef = useRef<HTMLDivElement>(null)
  const seqRef = useRef(0)

  useEffect(() => {
    ensureMermaidInit()
    const el = containerRef.current
    if (!el) return
    const def = definition.trim()
    if (!def) return

    let cancelled = false
    const renderId = `wb-mer-${reactId}-${seqRef.current++}`

    void (async () => {
      try {
        const { svg, bindFunctions } = await mermaid.render(renderId, def)
        if (cancelled || !containerRef.current) return
        if (looksLikeMermaidRenderError(svg)) {
          mountMermaidFallback(containerRef.current, definition)
          return
        }
        containerRef.current.innerHTML = svg
        bindFunctions?.(containerRef.current)
      } catch {
        if (cancelled || !containerRef.current) return
        mountMermaidFallback(containerRef.current, definition)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [definition, reactId])

  return (
    <div
      ref={containerRef}
      className="agent-workspace-msg__mermaid-wrap"
      role="img"
      aria-label="Mermaid 流程图"
    />
  )
}

function isMermaidCodeBlock(text: string, lang: string | undefined): boolean {
  const t = text.replace(/\n$/, '')
  if (lang === 'mermaid') return true
  // 已明确为其它代码语言（非 text/plain）时不抢判
  if (
    lang &&
    lang !== 'mermaid' &&
    lang !== 'text' &&
    lang !== 'plaintext' &&
    lang !== 'txt'
  ) {
    return false
  }
  const trimmed = t.trimStart()
  if (!MERMAID_LEADING.test(trimmed)) return false
  // 避免把行内 `graph` 误判为图：多行或典型流程图长度
  const looksBlock = t.includes('\n') || t.length > 40
  return looksBlock
}

/** 供 react-markdown components 使用 */
export const workspaceMarkdownComponents: Components = {
  pre({ children }) {
    const arr = Children.toArray(children)
    if (arr.length === 1 && isValidElement(arr[0])) {
      const cn = (arr[0].props as { className?: string }).className
      if (cn?.includes('agent-workspace-msg__mermaid-host')) {
        return <>{children}</>
      }
    }
    return <pre>{children}</pre>
  },
  code({ className, children, node: _node, ...props }) {
    const text = String(children).replace(/\n$/, '')
    const langMatch = /language-(\w+)/.exec(className ?? '')
    const lang = langMatch?.[1]

    if (isMermaidCodeBlock(text, lang)) {
      return (
        <div className="agent-workspace-msg__mermaid-host">
          <MermaidDiagram definition={text} />
        </div>
      )
    }

    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
}
