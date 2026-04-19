import { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import DecisionBadge from '../components/DecisionBadge.jsx'
import DemoControls from '../components/DemoControls.jsx'
import { usePolling } from '../hooks/usePolling.js'
import { api, trustColor } from '../lib/api.js'

function ScoreBar({ score }) {
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? '#ef4444' : score >= 0.3 ? '#f59e0b' : '#10b981'
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="font-mono text-xs text-slate-300">{score.toFixed(3)}</span>
    </div>
  )
}

export default function LiveFlows() {
  const { data } = usePolling(() => api.demoRecent(100), 2000)
  const [selected, setSelected] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)

  const flows = data?.flows ?? []

  const openDetail = async (flow) => {
    setSelected(flow)
    setExplanation(null)
    setLoading(true)
    try {
      const r = await api.explain(flow.features || {}, flow.context || null)
      setExplanation(r)
    } catch (e) {
      setExplanation({ error: e.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-4">
      <div className="xl:col-span-3">
        <div className="glass overflow-hidden">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
            <div>
              <div className="text-sm font-medium">Live Flows</div>
              <div className="text-xs text-slate-400">Polling every 2 s · click a row for SHAP</div>
            </div>
            <span className="pill border border-white/10 bg-white/5 text-slate-300">
              {flows.length} shown
            </span>
          </div>
          <div className="max-h-[70vh] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-bg-panel/90 backdrop-blur">
                <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                  <th className="px-5 py-2 font-medium">Time</th>
                  <th className="px-5 py-2 font-medium">Source</th>
                  <th className="px-5 py-2 font-medium">Dest</th>
                  <th className="px-5 py-2 font-medium">Port</th>
                  <th className="px-5 py-2 font-medium">Score</th>
                  <th className="px-5 py-2 font-medium">Decision</th>
                  <th className="px-5 py-2 font-medium">Trust</th>
                  <th className="px-5 py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {flows.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-5 py-8 text-center text-sm text-slate-500">
                      No flows yet — start the demo simulator →
                    </td>
                  </tr>
                )}
                {flows.map((f, i) => (
                  <tr
                    key={i}
                    onClick={() => openDetail(f)}
                    className="cursor-pointer border-t border-white/5 hover:bg-white/5"
                  >
                    <td className="px-5 py-2 font-mono text-xs text-slate-400">{f.ts}</td>
                    <td className="px-5 py-2 font-mono text-xs">{f.src_ip || '—'}</td>
                    <td className="px-5 py-2 font-mono text-xs">{f.dst_ip || '—'}</td>
                    <td className="px-5 py-2 font-mono text-xs">{f.dst_port ?? '—'}</td>
                    <td className="px-5 py-2"><ScoreBar score={f.score} /></td>
                    <td className="px-5 py-2"><DecisionBadge decision={f.decision} /></td>
                    <td className={`px-5 py-2 text-xs font-medium ${trustColor(f.trust_level)}`}>
                      {f.trust_level || '—'}
                    </td>
                    <td className="px-5 py-2 text-slate-500">
                      <ChevronRight className="h-4 w-4" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="space-y-4 xl:col-span-1">
        <DemoControls />
        <div className="glass p-5">
          <div className="mb-3 text-sm font-medium">Flow Detail</div>
          {!selected && (
            <div className="text-xs text-slate-500">Click a flow to see SHAP explanation.</div>
          )}
          {selected && (
            <div className="space-y-3 text-xs">
              <div className="flex items-center gap-2">
                <DecisionBadge decision={selected.decision} />
                <span className="font-mono text-slate-300">score {selected.score.toFixed(3)}</span>
              </div>
              <div className="font-mono text-[11px] text-slate-400">
                {selected.src_ip} → {selected.dst_ip}:{selected.dst_port}
              </div>
              {loading && <div className="text-slate-500">Computing SHAP…</div>}
              {explanation?.explanation && (
                <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-slate-200">
                  {explanation.explanation}
                </div>
              )}
              {explanation?.top_features && (
                <ul className="space-y-1">
                  {explanation.top_features.map((tf) => (
                    <li key={tf.feature} className="flex items-center justify-between">
                      <span className="truncate text-slate-300">{tf.feature}</span>
                      <span
                        className={`font-mono ${
                          tf.direction === '+' ? 'text-accent-red' : 'text-accent-green'
                        }`}
                      >
                        {tf.direction}
                        {Math.abs(tf.shap_value).toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {explanation?.error && (
                <div className="text-accent-red">Error: {explanation.error}</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
