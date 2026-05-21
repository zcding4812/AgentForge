import { useCallback, useLayoutEffect, useRef } from 'react'

const BOTTOM_THRESHOLD_PX = 80

/**
 * Manages auto-scroll-to-bottom for a chat message container.
 *
 * - While the user is near the bottom, new messages auto-scroll the container.
 * - When the user scrolls up to read history, auto-scroll is suppressed.
 *
 * @param deps - values whose change should trigger a scroll-to-bottom check (e.g. messages array, loading flag)
 * @returns `{ scrollRef, onScroll }` — attach `scrollRef` to the scrollable container and `onScroll` to its `onScroll` handler.
 */
export function useChatScrollAnchor(deps: readonly unknown[]) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const stickRef = useRef(true)

  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const { scrollTop, scrollHeight, clientHeight } = el
    stickRef.current = scrollHeight - scrollTop - clientHeight <= BOTTOM_THRESHOLD_PX
  }, [])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el || !stickRef.current) return
    el.scrollTop = el.scrollHeight
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { scrollRef, onScroll } as const
}
