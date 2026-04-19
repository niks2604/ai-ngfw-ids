import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Brain } from 'lucide-react'
import DecisionBadge from '../components/DecisionBadge.jsx'
import { usePolling } from '../hooks/usePolling.js'
import { api, trustColor } from '../lib/api.js'

function pickLastBlockedOrHighest(flows) {
  if (!flows?.length) return null
  const blocked = flows.find((f) => f.decision === 'BLOCK' || f.decision === 'QUARANTINE')
  if (blocked) return blocked
  return flows.reduce((a, b) => (a.score > b.score ? a : b), flows[0])
}

export default function Explainability() {
  const { data } = usePolling(() => api.demoRecent(50), 2000)
  const selected = useMemo(() => pickLastBlockedOrHighest(data?.flows), [data])
  const { data: exp } = usePolling(
    async () => (selected ? api.explain(selected.features || {}, selected.context || null) : null),
    3000,
    [selected?.ts, selected?.src_ip],
  )

  const chartData =
    exp?.top_features?.map((tf) => ({
      feature: tf.feature.length > 22 ? tf.feature.slice(0, 22) + '…' : tf.feature,
      shap: tf.shap_value,
      direction: tf.direction,
      value: tf.value,
    })) || []

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
      <div className="glass p-5 xl:col-span-2">
        <div className="mb-4 flex items-center gap-2">
          <Brain className="h-5 w-5 text-accent-purple" />
          <div>
            <div className="text-sm font-medium">SHAP Feature Contributions</div>
            <div className="text-xs text-slate-400">Signed weighted tree-SHAP (RF + XGBoost)</div>
          </div>
        </div>

        <div className="h-80">
          {chartData.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              Waiting for a scored flow…
            </div>
          ) : (
            <ResponsiveContainer>
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ left: 24, right: 24, top: 8, bottom: 8 }}
              >
                <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  stroke="#cbd5e1"
                  tick={{ fontSize: 11 }}
                  width={160}
                />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.25)" />
                <Tooltip
                  cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                  contentStyle={{
                    background: 'rgba(17,24,39,0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="shap" radius={[0, 6, 6, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.shap >= 0 ? '#ef4444' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="space-y-4 xl:col-span-1">
        <div className="glass p-5">
          <div className="mb-3 text-sm font-medium">Why was this decided?</div>
          {!selected ? (
            <div className="text-xs text-slate-500">No flow selected yet.</div>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <DecisionBadge decision={selected.decision} />
                <span className="font-mono text-xs text-slate-300">
                  score {selected.score.toFixed(3)}
                </span>
                <span className={`text-xs font-medium ${trustColor(selected.trust_level)}`}>
                  {selected.trust_level}
                </span>
              </div>
              {exp?.explanation && (
                <p className="rounded-xl border border-white/10 bg-white/5 p-3 text-sm leading-relaxed text-slate-200">
                  {exp.explanation}
                </p>
              )}
              {exp?.zero_trust && (
                <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-xs">
                  <div className="mb-1 font-medium text-slate-200">Zero Trust</div>
                  <div className="text-slate-400">
                    primary: <span className="text-slate-100">{exp.zero_trust.primary_action}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {exp.zero_trust.principles_applied?.map((p) => (
                      <span
                        key={p}
                        className="pill border border-white/10 bg-white/5 text-[10px] text-slate-300"
                      >
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="glass p-5">
          <div className="mb-3 text-sm font-medium">Model scores</div>
          {selected?.model_scores ? (
            <ul className="space-y-2 text-xs">
              {Object.entries(selected.model_scores).map(([k, v]) => (
                <li key={k} className="flex items-center justify-between">
                  <span className="text-slate-300">{k}</span>
                  <span className="font-mono text-slate-100">{v.toFixed(3)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-slate-500">—</div>
          )}
        </div>
      </div>
    </div>
  )
}
