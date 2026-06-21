import { Brain, Network, AlertTriangle, Shield, ShieldOff, Eye } from 'lucide-react'

// Render a single GNN-analysis payload as a self-contained panel.
// Accepts the shape returned by /gnn/demo/{scenario}:
//   { network_threat_score, flow_threat_score, total_nodes, total_edges,
//     inference_time_ms, patterns_detected, source_analysis,
//     recommendation, override, explanation }
//
// /predict/gnn returns a similar but not identical shape; the existing
// LiveFlows page already adapts that on its own.

const RECO_STYLE = {
  ALLOW: {
    icon: Shield,
    classes: 'border-emerald-500/30 bg-emerald-500/10 text-accent-green',
    label: 'ALLOW',
  },
  INSPECT: {
    icon: Eye,
    classes: 'border-amber-500/30 bg-amber-500/10 text-accent-yellow',
    label: 'INSPECT',
  },
  BLOCK: {
    icon: ShieldOff,
    classes: 'border-red-500/30 bg-red-500/10 text-accent-red',
    label: 'BLOCK',
  },
}

function StatBox({ label, value, tone }) {
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
    </div>
  )
}

function ThreatBar({ score, label }) {
  const pct = Math.min(100, Math.max(0, score * 100))
  const level = score > 0.7 ? 'HIGH' : score > 0.3 ? 'MEDIUM' : 'LOW'
  const color = score > 0.7 ? '#ef4444' : score > 0.3 ? '#f59e0b' : '#10b981'
  const txt = score > 0.7 ? 'text-accent-red' : score > 0.3 ? 'text-accent-yellow' : 'text-accent-green'
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className={`font-mono ${txt}`}>
          {pct.toFixed(1)}% · {level}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

export default function GNNResultPanel({ result }) {
  if (!result) return null
  const reco = RECO_STYLE[result.recommendation] || RECO_STYLE.INSPECT
  const RecoIcon = reco.icon
  const networkScore = result.network_threat_score ?? result.flow_threat_score ?? 0
  const flowScore = result.flow_threat_score ?? null

  return (
    <div className="rounded-xl border border-violet-500/30 bg-gradient-to-br from-violet-500/10 to-violet-500/5 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-accent-purple" />
          <div className="text-sm font-medium text-slate-100">GNN network analysis</div>
        </div>
        <span className={`pill border text-[10px] font-medium ${reco.classes}`}>
          <RecoIcon className="mr-1 inline h-3 w-3" />
          {reco.label}
        </span>
      </div>

      <div className="mb-4 grid grid-cols-4 gap-2 text-[11px]">
        <StatBox label="Active IPs" value={result.total_nodes ?? '—'} />
        <StatBox label="Connections" value={result.total_edges ?? '—'} />
        <StatBox label="Window" value="60s" />
        <StatBox
          label="Inference"
          value={result.inference_time_ms != null ? `${result.inference_time_ms} ms` : '—'}
        />
      </div>

      <div className="mb-3 space-y-3">
        <ThreatBar score={networkScore} label="Network threat score" />
        {flowScore != null && (
          <ThreatBar score={flowScore} label="Flow threat score" />
        )}
      </div>

      {result.source_analysis && (
        <div className="mb-3 rounded-lg border border-white/10 bg-white/5 p-3">
          <div className="mb-2 text-[11px] text-slate-400">
            Source IP behaviour:{' '}
            <span className="font-mono text-slate-200">
              {result.source_analysis.ip}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <StatBox
              label="Out-degree"
              value={result.source_analysis.out_degree}
              tone={result.source_analysis.out_degree >= 20 ? 'red' : null}
            />
            <StatBox
              label="In-degree"
              value={result.source_analysis.in_degree}
              tone={result.source_analysis.in_degree >= 20 ? 'red' : null}
            />
            <StatBox
              label="Threat"
              value={(result.source_analysis.threat_score ?? 0).toFixed(2)}
              tone={
                result.source_analysis.threat_score > 0.7
                  ? 'red'
                  : result.source_analysis.threat_score > 0.3
                  ? 'amber'
                  : null
              }
            />
          </div>
        </div>
      )}

      {result.patterns_detected?.length > 0 && (
        <div className="mb-3">
          <div className="mb-2 flex items-center gap-1.5 text-[11px] text-slate-400">
            <Network className="h-3 w-3" /> Patterns detected
          </div>
          <ul className="space-y-1.5">
            {result.patterns_detected.map((p, i) => (
              <li
                key={i}
                className="rounded-lg border border-red-500/30 bg-red-500/10 p-2.5 text-[11px]"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-accent-red">
                    🔍 {p.type.replace(/_/g, ' ')}
                  </span>
                  <span className="font-mono text-slate-300">
                    {(p.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                {p.description && (
                  <div className="mt-1 text-slate-300">{p.description}</div>
                )}
                {p.source && (
                  <div className="mt-0.5 text-slate-500">
                    Source: <span className="font-mono">{p.source}</span>
                    {p.targets != null && ` → ${p.targets} target${p.targets === 1 ? '' : 's'}`}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.override && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-[11px] text-amber-100">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-accent-yellow" />
          <div>
            <div className="font-medium">
              Override from INSPECT → {result.recommendation}
            </div>
            <div className="mt-0.5 text-slate-300">
              GNN topology analysis takes precedence over the per-flow score.
            </div>
          </div>
        </div>
      )}

      {result.explanation && (
        <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-2.5 text-[11px] text-slate-200">
          {result.explanation}
        </div>
      )}
    </div>
  )
}
