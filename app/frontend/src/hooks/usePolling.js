import { useEffect, useRef, useState } from 'react'

/**
 * Poll an async function at a fixed interval.
 * Pauses on window blur to avoid waking the API unnecessarily.
 */
export function usePolling(fn, intervalMs, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const timer = useRef(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true

    const tick = async () => {
      try {
        const res = await fn()
        if (alive.current) {
          setData(res)
          setError(null)
        }
      } catch (e) {
        if (alive.current) setError(e)
      } finally {
        if (alive.current) setLoading(false)
      }
    }

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

  return { data, error, loading }
}
