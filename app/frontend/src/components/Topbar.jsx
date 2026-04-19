import { Activity, CircleDot } from 'lucide-react'
import { usePolling } from '../hooks/usePolling.js'
import { api } from '../lib/api.js'

export default function Topbar() {
  const { data, error } = usePolling(api.health, 5000)
  const ok = data?.status === 'ok' && !error

  return (
    <header className="sticky top-0 z-10 flex items-center justify-between gap-4 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="glass flex items-center gap-2 px-3 py-1.5">
          <Activity className="h-4 w-4 text-accent-blue" />
          <span className="text-xs text-slate-300">AI-NGFW / IDS Inference</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="glass flex items-center gap-2 px-3 py-1.5 text-xs">
          <CircleDot
            className={`h-3.5 w-3.5 ${ok ? 'text-accent-green' : 'text-accent-red'}`}
          />
          <span className="text-slate-300">
            API {ok ? 'online' : error ? 'offline' : 'loading…'}
          </span>
          {data?.models_loaded && (
            <span className="text-slate-400">· {data.n_features} features</span>
          )}
        </div>
      </div>
    </header>
  )
}
