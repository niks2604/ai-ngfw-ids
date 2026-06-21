import { useState } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  ArrowUpRight,
  CheckCircle2,
  RefreshCcw,
  Gauge,
} from 'lucide-react'
import StatCard from './StatCard.jsx'
import { api } from '../lib/api.js'
import { usePolling } from '../hooks/usePolling.js'

// Minimum verified captures required to enable manual retraining.
// Below this the queue is too small to move the needle, so we keep
// the button greyed out to avoid wasted retrains.
const RETRAIN_THRESHOLD = 100
// Target the progress bar fills toward — purely cosmetic copy.
const RETRAIN_TARGET = 1000

function fmtPct(n) {
  if (n == null || isNaN(n)) return '—'
  return `${Number(n).toFixed(1)}%`
}

function fmtSigned(n) {
  if (n == null || isNaN(n)) return '—'
  const v = Number(n)
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
}

export default function ModelPerformance() {
  const { data: metrics } = usePolling(api.modelMetrics, 5000)
  const [retraining, setRetraining] = useState(false)
  const [error, setError] = useState(null)
  const [lastResult, setLastResult] = useState(null)

  const baseline = metrics?.baseline_accuracy ?? 0
  const current = metrics?.current_accuracy ?? 0
  const improvement = metrics?.improvement ?? current - baseline
  const verified = metrics?.verified_captures ?? 0
  const queueSize = metrics?.training_queue_size ?? verified
  const history = metrics?.history ?? []
  const versions = metrics?.versions ?? []
  const canRetrain = queueSize >= RETRAIN_THRESHOLD && !retraining

  const pct = Math.min(100, (queueSize / RETRAIN_TARGET) * 100)

  const handleRetrain = async () => {
    setError(null)
    setRetraining(true)
    try {
      const r = await api.modelRetrain()
      setLastResult(r)
      // The 5s poll on metrics will pick up the new accuracy /
      // version on its next tick; no manual refetch needed.
    } catch (e) {
      setError(e.message)
    } finally {
      setRetraining(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          icon={Gauge}
          label="Baseline accuracy"
          value={fmtPct(baseline)}
          tone="blue"
          hint="On CICIDS-2018 (out-of-distribution)"
        />
        <StatCard
          icon={Activity}
          label="Current accuracy"
          value={fmtPct(current)}
          tone="green"
          hint={metrics?.current_version ? `Model ${metrics.current_version}` : '—'}
        />
        <StatCard
          icon={ArrowUpRight}
          label="Improvement"
          value={fmtSigned(improvement)}
          tone={improvement >= 0 ? 'green' : 'red'}
          hint="vs. shipped baseline"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {/* Accuracy over time chart */}
        <div className="glass p-5 xl:col-span-2">
          <div className="mb-4 flex items-center gap-2">
            <Activity className="h-5 w-5 text-accent-green" />
            <div>
              <div className="text-sm font-medium">Accuracy over retrains</div>
              <div className="text-xs text-slate-400">
                Each point is one retrain on verified honeypot captures
              </div>
            </div>
          </div>
          <div className="h-56">
            {history.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                No retrain history yet.
              </div>
            ) : (
              <ResponsiveContainer>
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="accFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis
                    domain={[
                      (dataMin) => Math.max(0, Math.floor(dataMin - 5)),
                      (dataMax) => Math.min(100, Math.ceil(dataMax + 5)),
                    ]}
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'rgba(17,24,39,0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    formatter={(v) => [`${v.toFixed(1)}%`, 'Accuracy']}
                  />
                  <Area
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#10b981"
                    fill="url(#accFill)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Retrain control card */}
        <div className="glass p-5">
          <div className="mb-4 flex items-center gap-2">
            <RefreshCcw className="h-5 w-5 text-accent-purple" />
            <div className="text-sm font-medium">Retrain trigger</div>
          </div>

          <div className="mb-2 flex items-center justify-between text-xs text-slate-300">
            <span>Verified captures for retrain</span>
            <span className="font-mono">
              {queueSize.toLocaleString()} / {RETRAIN_TARGET.toLocaleString()}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className={`h-full rounded-full transition-all ${
                canRetrain ? 'bg-accent-green' : 'bg-accent-blue'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="mt-2 text-[11px] text-slate-500">
            {queueSize < RETRAIN_THRESHOLD
              ? `${RETRAIN_THRESHOLD - queueSize} more needed to enable retraining.`
              : 'Threshold reached — retrain available.'}
          </div>

          <button
            type="button"
            onClick={handleRetrain}
            disabled={!canRetrain}
            className="mt-4 w-full rounded-xl border border-white/10 bg-gradient-to-r from-accent-purple/30 to-accent-blue/30 px-3 py-2 text-xs font-medium text-slate-100 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {retraining ? 'Retraining…' : 'Retrain Model Now'}
          </button>

          {error && (
            <div className="mt-3 text-[11px] text-accent-red">Error: {error}</div>
          )}
          {lastResult && !error && (
            <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2 text-[11px] text-emerald-200">
              <div>
                {lastResult.previous_version} → <strong>{lastResult.current_version}</strong>{' '}
                in {lastResult.seconds}s
              </div>
              <div>
                Acc {fmtPct(lastResult.previous_accuracy)} →{' '}
                <strong>{fmtPct(lastResult.current_accuracy)}</strong>{' '}
                ({fmtSigned(lastResult.current_accuracy - lastResult.previous_accuracy)})
              </div>
              <div>Trained on {lastResult.samples_added.toLocaleString()} new samples</div>
            </div>
          )}
        </div>
      </div>

      {/* Version history table */}
      <div className="glass p-5">
        <div className="mb-3 flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-accent-blue" />
          <div className="text-sm font-medium">Model version history</div>
        </div>
        {versions.length === 0 ? (
          <div className="text-xs text-slate-500">No versions recorded.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                  <th className="px-3 py-2 font-medium">Version</th>
                  <th className="px-3 py-2 font-medium">Date</th>
                  <th className="px-3 py-2 font-medium">Accuracy</th>
                  <th className="px-3 py-2 font-medium">Samples added</th>
                  <th className="px-3 py-2 font-medium">Note</th>
                </tr>
              </thead>
              <tbody>
                {versions
                  .slice()
                  .reverse()
                  .map((v) => (
                    <tr key={v.version} className="border-t border-white/5">
                      <td className="px-3 py-2 font-mono text-xs">{v.version}</td>
                      <td className="px-3 py-2 font-mono text-xs text-slate-400">
                        {v.date}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-accent-green">
                        {fmtPct(v.accuracy)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {(v.samples || 0).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-300">{v.note}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
