import { useEffect, useState } from 'react'
import {
  Brain,
  ChevronRight,
  Network,
  AlertTriangle,
  Activity,
  Shield,
  ShieldOff,
  Eye,
} from 'lucide-react'
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

function ModelScoreRow({ label, score }) {
  const pct = Math.round(Math.min(1, Math.max(0, score)) * 100)
  const color = score >= 0.7 ? '#ef4444' : score >= 0.3 ? '#f59e0b' : '#10b981'
  return (
    <div className="flex items-center gap-3 text-xs">
      <div className="w-32 text-slate-300">{label}</div>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="w-10 text-right font-mono text-slate-200">{score.toFixed(2)}</div>
    </div>
  )
}

function ShapBar({ feature, value, shap }) {
  const pct = Math.min(100, Math.abs(shap) * 200)
  const isUp = shap >= 0
  const color = isUp ? '#ef4444' : '#10b981'
  return (
    <div className="flex items-center gap-3 text-[11px]">
      <div className="w-40 truncate text-slate-300" title={feature}>
        {feature}
      </div>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/10">
        <div
          className="absolute inset-y-0 left-1/2 origin-left"
          style={{
            width: `${pct / 2}%`,
            background: color,
            transform: isUp ? 'translateX(0)' : 'translateX(-100%) scaleX(-1)',
          }}
        />
      </div>
      <div
        className="w-16 text-right font-mono"
        style={{ color }}
        title={`value=${value}`}
      >
        {isUp ? '+' : ''}
        {shap.toFixed(3)}
      </div>
    </div>
  )
}

function CtxStat({ label, value, hint, tone }) {
  const toneCls =
    tone === 'red'
      ? 'text-accent-red'
      : tone === 'amber'
      ? 'text-accent-yellow'
      : tone === 'green'
      ? 'text-accent-green'
      : 'text-slate-100'
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-2.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`font-mono text-sm ${toneCls}`}>{value}</div>
      {hint && <div className="text-[10px] text-slate-400">{hint}</div>}
    </div>
  )
}

// Pretty-print the few features we actually want to surface in the
// "Flow info" header without dumping the full 51-column vector.
function summarizeFlow(flow) {
  const f = flow?.features || {}
  const dur = f['Flow Duration']
  const fwd = (f['Total Fwd Packets'] ?? 0) + (f['Total Backward Packets'] ?? 0)
  const bytes =
    (f['Total Length of Fwd Packets'] ?? 0) + (f['Total Length of Bwd Packets'] ?? 0)
  const pktRate = f['Flow Packets/s']
  return {
    durationSec: dur ? dur / 1_000_000 : null,
    packets: Math.round(fwd),
    bytes: Math.round(bytes),
    pktRate: pktRate ?? null,
  }
}

function detectPatternFromGraph(graph, ip) {
  if (!ip || !graph?.nodes) return null
  const node = graph.nodes.find((n) => n.ip === ip)
  if (!node) return null
  if (node.out_degree >= 20 && node.out_degree > node.in_degree * 3) {
    return {
      label: 'POTENTIAL PORT SCAN',
      detail: `Source connecting to ${node.out_degree} unique destinations`,
      confidence: Math.min(0.99, node.out_degree / 50),
    }
  }
  if (node.in_degree >= 20 && node.in_degree > node.out_degree * 3) {
    return {
      label: 'POTENTIAL DDoS TARGET',
      detail: `Destination receiving ${node.in_degree} unique sources`,
      confidence: Math.min(0.99, node.in_degree / 50),
    }
  }
  return null
}

const RECO_STYLE = {
  ALLOW: { icon: Shield, color: 'text-accent-green', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', label: 'ALLOW' },
  INSPECT: { icon: Eye, color: 'text-accent-yellow', bg: 'bg-amber-500/10', border: 'border-amber-500/30', label: 'INSPECT' },
  BLOCK: { icon: ShieldOff, color: 'text-accent-red', bg: 'bg-red-500/10', border: 'border-red-500/30', label: 'BLOCK' },
}

export default function LiveFlows() {
  const { data } = usePolling(() => api.demoRecent(100), 2000)
  const [selected, setSelected] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [gnnResult, setGnnResult] = useState(null)
  const [gnnLoading, setGnnLoading] = useState(false)
  const [graph, setGraph] = useState(null)

  const flows = data?.flows ?? []

  const openDetail = (flow) => {
    setSelected(flow)
    setExplanation(null)
    setGnnResult(null)
  }

  // Auto-fetch SHAP and GNN whenever a flow is selected. GNN only meaningfully
  // overrides INSPECT, but we fetch it for any flow so the user can always
  // see the topology view.
  useEffect(() => {
    if (!selected) return
    let cancelled = false
    setLoading(true)
    api
      .explain(selected.features || {}, selected.context || null)
      .then((r) => !cancelled && setExplanation(r))
      .catch((e) => !cancelled && setExplanation({ error: e.message }))
      .finally(() => !cancelled && setLoading(false))

    setGnnLoading(true)
    api
      .predictGnn(selected.features || {}, selected.context || null)
      .then((r) => !cancelled && setGnnResult(r))
      .catch((e) => !cancelled && setGnnResult({ error: e.message }))
      .finally(() => !cancelled && setGnnLoading(false))

    api
      .networkGraph()
      .then((g) => !cancelled && setGraph(g))
      .catch(() => !cancelled && setGraph(null))

    return () => {
      cancelled = true
    }
  }, [selected])

  const summary = selected ? summarizeFlow(selected) : null
  const ensemble = explanation?.ensemble || explanation?.model_scores
  const rfScore = ensemble?.rf_score ?? ensemble?.random_forest ?? selected?.model_scores?.random_forest ?? 0
  const xgbScore = ensemble?.xgb_score ?? ensemble?.xgboost ?? selected?.model_scores?.xgboost ?? 0
  const ifScore = ensemble?.if_score ?? ensemble?.isolation_forest ?? selected?.model_scores?.isolation_forest ?? 0
  const ensembleSpread = Math.max(rfScore, xgbScore, ifScore) - Math.min(rfScore, xgbScore, ifScore)

  const srcNode = graph?.nodes?.find((n) => n.ip === selected?.src_ip)
  const dstNode = graph?.nodes?.find((n) => n.ip === selected?.dst_ip)
  const gnnPayload = gnnResult?.gnn
  const ensembleDecision = selected?.decision
  const gnnRecommendation = gnnResult?.decision || ensembleDecision
  const overrode =
    ensembleDecision === 'INSPECT' &&
    gnnRecommendation &&
    gnnRecommendation !== 'INSPECT'
  const pattern =
    detectPatternFromGraph(graph, selected?.src_ip) ||
    detectPatternFromGraph(graph, selected?.dst_ip)
  const reco = RECO_STYLE[gnnRecommendation] || RECO_STYLE.INSPECT
  const RecoIcon = reco.icon

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <div className="xl:col-span-3">
        <div className="glass overflow-hidden">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
            <div>
              <div className="text-sm font-medium">Live Flows</div>
              <div className="text-xs text-slate-400">
                Polling every 2 s · click a row for SHAP + GNN
              </div>
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
                    className={`cursor-pointer border-t border-white/5 hover:bg-white/5 ${
                      selected === f ? 'bg-white/5' : ''
                    }`}
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

      <div className="space-y-4 xl:col-span-2">
        <DemoControls />

        {!selected && (
          <div className="glass p-5 text-xs text-slate-500">
            Click a flow row to see ensemble breakdown, GNN topology analysis, and SHAP explanation.
          </div>
        )}

        {selected && (
          <>
            {/* FLOW INFO ---------------------------------------------- */}
            <div className="glass p-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-sm font-medium">Flow details</div>
                <DecisionBadge decision={selected.decision} />
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <CtxStat
                  label="Source"
                  value={selected.src_ip || '—'}
                  hint={selected.context?.is_internal_src ? 'Internal' : 'External'}
                />
                <CtxStat
                  label="Destination"
                  value={`${selected.dst_ip || '—'}:${selected.dst_port ?? '—'}`}
                />
                <CtxStat label="Protocol" value={selected.protocol ?? '—'} />
                <CtxStat
                  label="Duration"
                  value={
                    summary?.durationSec != null
                      ? `${summary.durationSec.toFixed(2)} s`
                      : '—'
                  }
                />
                <CtxStat
                  label="Bytes"
                  value={summary?.bytes != null ? summary.bytes.toLocaleString() : '—'}
                />
                <CtxStat
                  label="Packets"
                  value={summary?.packets != null ? summary.packets.toLocaleString() : '—'}
                  hint={
                    summary?.pktRate != null
                      ? `${summary.pktRate.toFixed(0)} pkts/s`
                      : null
                  }
                  tone={summary?.pktRate > 1000 ? 'red' : null}
                />
              </div>
            </div>

            {/* ENSEMBLE DECISION ------------------------------------- */}
            <div className="glass p-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-sm font-medium">Ensemble decision</div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-slate-300">
                    {selected.score.toFixed(3)}
                  </span>
                  <span className="text-slate-500">→</span>
                  <DecisionBadge decision={selected.decision} />
                </div>
              </div>
              <div className="space-y-2">
                <ModelScoreRow label="Random Forest" score={rfScore} />
                <ModelScoreRow label="XGBoost" score={xgbScore} />
                <ModelScoreRow label="Isolation Forest" score={ifScore} />
              </div>
              {selected.decision === 'INSPECT' && ensembleSpread >= 0.1 && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 p-2 text-[11px] text-violet-100">
                  <Brain className="mt-0.5 h-3 w-3 shrink-0" />
                  Models disagree (spread {(ensembleSpread * 100).toFixed(0)} pp) → sent to GNN
                  for deeper analysis.
                </div>
              )}
            </div>

            {/* GNN ANALYSIS ------------------------------------------ */}
            <div
              className={`glass border ${reco.border} bg-gradient-to-br from-violet-500/5 to-transparent p-5`}
            >
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-accent-purple" />
                  <div className="text-sm font-medium">GNN analysis</div>
                </div>
                <span
                  className={`pill border text-[10px] font-medium ${reco.border} ${reco.bg} ${reco.color}`}
                >
                  <RecoIcon className="mr-1 inline h-3 w-3" />
                  {reco.label}
                </span>
              </div>

              {gnnLoading && (
                <div className="text-[11px] text-slate-500">Running topology analysis…</div>
              )}
              {gnnResult?.error && (
                <div className="text-[11px] text-accent-red">GNN error: {gnnResult.error}</div>
              )}

              {!gnnLoading && !gnnResult?.error && (
                <>
                  {/* Network context */}
                  <div className="mb-3">
                    <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-slate-400">
                      <Network className="h-3 w-3" /> Network context (last 60 s)
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-[11px]">
                      <CtxStat label="Active IPs" value={graph?.node_count ?? '—'} />
                      <CtxStat label="Connections" value={graph?.flow_count ?? '—'} />
                      <CtxStat label="Edges" value={graph?.edge_count ?? '—'} />
                    </div>
                  </div>

                  {/* Source IP behavior */}
                  {srcNode && (
                    <div className="mb-3">
                      <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-slate-400">
                        <Activity className="h-3 w-3" /> Source IP behavior:{' '}
                        <span className="font-mono text-slate-300">{srcNode.ip}</span>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <CtxStat
                          label="Outbound"
                          value={srcNode.out_degree}
                          tone={srcNode.out_degree >= 20 ? 'red' : null}
                          hint={srcNode.out_degree >= 20 ? 'HIGH' : null}
                        />
                        <CtxStat label="Inbound" value={srcNode.in_degree} />
                        <CtxStat
                          label="Threat"
                          value={srcNode.score.toFixed(2)}
                          tone={srcNode.score >= 0.7 ? 'red' : srcNode.score >= 0.3 ? 'amber' : null}
                        />
                      </div>
                    </div>
                  )}

                  {dstNode && !srcNode && (
                    <div className="mb-3">
                      <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-slate-400">
                        <Activity className="h-3 w-3" /> Destination behavior:{' '}
                        <span className="font-mono text-slate-300">{dstNode.ip}</span>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <CtxStat
                          label="Inbound"
                          value={dstNode.in_degree}
                          tone={dstNode.in_degree >= 20 ? 'red' : null}
                          hint={dstNode.in_degree >= 20 ? 'HIGH' : null}
                        />
                        <CtxStat label="Outbound" value={dstNode.out_degree} />
                        <CtxStat
                          label="Threat"
                          value={dstNode.score.toFixed(2)}
                          tone={dstNode.score >= 0.7 ? 'red' : dstNode.score >= 0.3 ? 'amber' : null}
                        />
                      </div>
                    </div>
                  )}

                  {/* Pattern detection */}
                  {pattern && (
                    <div className="mb-3 rounded-xl border border-accent-purple/40 bg-accent-purple/10 p-3 text-[11px]">
                      <div className="flex items-center justify-between">
                        <div className="font-medium text-accent-purple">
                          🔍 {pattern.label}
                        </div>
                        <div className="font-mono text-slate-300">
                          {(pattern.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="mt-1 text-slate-400">{pattern.detail}</div>
                    </div>
                  )}

                  {/* GNN scores */}
                  {gnnPayload && (
                    <div className="mb-3 grid grid-cols-2 gap-2 text-[11px]">
                      <CtxStat
                        label="GNN endpoint risk"
                        value={(gnnPayload.gnn_endpoint_risk ?? 0).toFixed(2)}
                      />
                      <CtxStat
                        label="GNN graph score"
                        value={(gnnPayload.gnn_graph_score ?? 0).toFixed(2)}
                      />
                    </div>
                  )}

                  {/* Override note */}
                  {overrode && (
                    <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-[11px] text-amber-100">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-accent-yellow" />
                      <div>
                        <div className="font-medium">
                          Override from INSPECT → {gnnRecommendation}
                        </div>
                        <div className="mt-0.5 text-slate-300">
                          Based on network pattern analysis.
                        </div>
                      </div>
                    </div>
                  )}

                  {ensembleDecision !== 'INSPECT' && !gnnResult?.error && (
                    <div className="text-[11px] text-slate-500">
                      GNN only overrides INSPECT decisions; this flow was {ensembleDecision}.
                      The topology view above is informational.
                    </div>
                  )}
                </>
              )}
            </div>

            {/* SHAP EXPLANATION ------------------------------------- */}
            <div className="glass p-5">
              <div className="mb-3 flex items-center gap-2">
                <div className="text-sm font-medium">SHAP explanation</div>
                <span className="text-[10px] text-slate-500">why this score?</span>
              </div>
              {loading && <div className="text-[11px] text-slate-500">Computing SHAP…</div>}
              {explanation?.error && (
                <div className="text-[11px] text-accent-red">Error: {explanation.error}</div>
              )}
              {explanation?.explanation && (
                <div className="mb-3 rounded-lg border border-white/10 bg-white/5 p-2.5 text-[11px] text-slate-200">
                  {explanation.explanation}
                </div>
              )}
              {explanation?.top_features?.length > 0 && (
                <div className="space-y-1.5">
                  {explanation.top_features.slice(0, 6).map((tf) => (
                    <ShapBar
                      key={tf.feature}
                      feature={tf.feature}
                      value={tf.value}
                      shap={tf.direction === '+' ? Math.abs(tf.shap_value) : -Math.abs(tf.shap_value)}
                    />
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
