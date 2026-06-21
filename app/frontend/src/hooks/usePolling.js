import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Poll an async function at a fixed interval.
 * Pauses on window blur to avoid waking the API unnecessarily.
 *
 * Returns `{ data, error, loading, refresh }` where `refresh()` triggers
 * an immediate re-fetch. Use it after a mutation (e.g. verifying a
 * capture) so the UI reflects the new state without waiting up to
 * `intervalMs` for the next scheduled tick.
 */
export function usePolling(fn, intervalMs, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const timer = useRef(null)
  const alive = useRef(true)
  // Keep `fn` in a ref so `refresh()` can call the latest closure
  // without re-running the polling effect on every render.
  const fnRef = useRef(fn)
  useEffect(() => {
    fnRef.current = fn
  })

  const tick = useCallback(async () => {
    try {
      const res = await fnRef.current()
      if (alive.current) {
        setData(res)
        setError(null)
      }
    } catch (e) {
      if (alive.current) setError(e)
    } finally {
      if (alive.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    tick()
    timer.current = setInterval(tick, intervalMs)

    const onVisibility = () => {
      if (document.hidden) {
        if (timer.current) clearInterval(timer.current)
      } else {
        tick()
        timer.current = setInterval(tick, intervalMs)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      alive.current = false
      if (timer.current) clearInterval(timer.current)
      document.removeEventListener('visibilitychange', onVisibility)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, loading, refresh: tick }
}
