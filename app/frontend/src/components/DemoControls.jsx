import { useState, useEffect } from 'react'
import { Play, Square, Zap } from 'lucide-react'
import { api } from '../lib/api.js'

const SCENARIOS = [
  { id: 'normal_traffic', label: 'Normal', tone: 'bg-emerald-500/10 border-emerald-500/30 text-accent-green' },
  { id: 'ddos_attack', label: 'DDoS', tone: 'bg-red-500/10 border-red-500/30 text-accent-red' },
  { id: 'port_scan', label: 'Port Scan', tone: 'bg-amber-500/10 border-amber-500/30 text-accent-yellow' },
  { id: 'brute_force', label: 'Brute Force', tone: 'bg-orange-500/10 border-orange-500/30 text-orange-400' },
  { id: 'mixed_realistic', label: 'Mixed', tone: 'bg-sky-500/10 border-sky-500/30 text-accent-blue' },
]

export default function DemoControls() {
  const [status, setStatus] = useState(null)
  const [scenario, setScenario] = useState('mixed_realistic')
  const [speed, setSpeed] = useState(10)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const poll = async () => {
      try {
        setStatus(await api.demoStatus())
      } catch {
        setStatus({ running: false, available: false })
      }
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [])

  const start = async () => {
    setBusy(true)
    try {
      await api.demoStart(scenario, Number(speed))
    } catch (e) {
      alert(`Start failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      await api.demoStop()
    } catch (e) {
      alert(`Stop failed: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  const running = status?.running

  if (status?.available === false) {
    return (
      <div className="glass p-5">
        <div className="mb-2 flex items-center gap-2 text-sm">
          <Zap className="h-4 w-4 text-slate-500" />
          <span className="font-medium text-slate-300">Demo Simulator</span>
        </div>
        <div className="text-xs text-slate-500">
          Simulator endpoints unavailable — start the API with the simulator enabled.
        </div>
      </div>
    )
  }

  return (
    <div className="glass p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm">
          <Zap className="h-4 w-4 text-accent-purple" />
          <span className="font-medium">Demo Simulator</span>
        </div>
        <span
          className={`pill border ${running ? 'border-emerald-500/30 bg-emerald-500/10 text-accent-green' : 'border-white/10 bg-white/5 text-slate-400'}`}
        >
          {running ? 'running' : 'idle'}
          {status?.sent != null && (
            <span className="ml-1 font-mono">· {status.sent.toLocaleString()}</span>
          )}
        </span>
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {SCENARIOS.map((s) => (
          <button
            key={s.id}
            onClick={() => setScenario(s.id)}
            disabled={running || busy}
            className={`pill cursor-pointer border transition ${
              scenario === s.id ? s.tone : 'border-white/10 bg-white/5 text-slate-300'
            } ${running || busy ? 'opacity-60 cursor-not-allowed' : 'hover:bg-white/10'}`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="mb-4">
        <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
          <span>Speed</span>
          <span className="font-mono text-slate-200">{speed}×</span>
        </div>
        <input
          type="range"
          min="1"
          max="100"
          step="1"
          value={speed}
          onChange={(e) => setSpeed(e.target.value)}
          disabled={running || busy}
          className="w-full accent-accent-blue"
        />
      </div>

      <div className="flex gap-2">
        {!running ? (
          <button
            onClick={start}
            disabled={busy}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-accent-blue px-3 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
          >
            <Play className="h-4 w-4" /> Start
          </button>
        ) : (
          <button
            onClick={stop}
            disabled={busy}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-accent-red px-3 py-2 text-sm font-medium text-white transition hover:bg-red-500 disabled:opacity-50"
          >
            <Square className="h-4 w-4" /> Stop
          </button>
        )}
      </div>
    </div>
  )
}
