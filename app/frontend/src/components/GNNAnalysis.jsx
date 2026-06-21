import { Brain, Network, AlertTriangle, ShieldOff, Eye, Shield } from 'lucide-react'

// Render the GNN's view of a single flow: the topology-derived risk
// scores, the patterns it spotted, and (when applicable) the
// override it pushed onto the ensemble's INSPECT decision.
//
// Expects the ``gnn`` payload shape returned by POST /predict when
// ``gnn_analyzed: true``:
//   { flow_threat_score, network_threat_score, total_nodes,
//     total_edges, patterns_detected: [{type, source, confidence}],
//     recommendation, override_reason }
//
// If ``ensembleDecision`` (the pre-override decision) differs from
// ``gnn.recommendation`` we show an "Override from INSPECT" badge.

const RECOMMENDATION_STYLES = {
  ALLOW: {
    icon: Shield,
    classes: 'border-emerald-500/30 bg-emerald-500/10 text-accent-green',
    label: 'Allow',
  },
  INSPECT: {
    icon: Eye,
    classes: 'border-amber-500/30 bg-amber-500/10 text-accent-yellow',
    label: 'Inspect',
  },
  BLOCK: {
    icon: ShieldOff,
    classes: 'border-red-500/30 bg-red-500/10 text-accent-red',
    label: 'Block',
  },
}

function ScoreRow({ label, score, tone }) {
  const pct = Math.round(Math.min(1, Math.max(0, score)) * 100)
  const color =
    tone === 'flow'
      ? score >= 0.7
        ? '#ef4444'
        : score >= 0.3
        ? '#f59e0b'
        : '#10b981'
      : '#8b5cf6'
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-slate-200">{score.toFixed(3)}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

export default function GNNAnalysis({ gnn, ensembleDecision }) {
  if (!gnn) return null
  const rec = RECOMMENDATION_STYLES[gnn.recommendation] || RECOMMENDATION_STYLES.INSPECT
  const RecIcon = rec.icon
  const overrode = ensembleDecision === 'INSPECT' && gnn.recommendation !== 'INSPECT'

  return (
    <div className="rounded-xl border border-violet-500/30 bg-gradient-to-br from-violet-500/10 to-violet-500/5 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-accent-purple" />
          <div className="text-sm font-medium text-slate-100">GNN Deep Analysis</div>
        </div>
        <span
          className={`pill border text-[10px] font-medium ${rec.classes}`}
          title="GNN recommendation"
        >
          <RecIcon className="mr-1 inline h-3 w-3" />
          {rec.label}
        </span>
      </div>

      <div className="mb-4 grid grid-cols-3 gap-2 text-[11px]">
        <div className="rounded-lg border border-white/10 bg-white/5 p-2">
          <div className="text-slate-500">Nodes</div>
          <div className="font-mono text-sm text-slate-100">{gnn.total_nodes}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/5 p-2">
          <div className="text-slate-500">Edges</div>
          <div className="font-mono text-sm text-slate-100">{gnn.total_edges}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/5 p-2">
          <div className="text-slate-500">Window</div>
          <div className="font-mono text-sm text-slate-100">60 s</div>
        </div>
      </div>

      <div className="space-y-3">
        <ScoreRow label="Flow threat score" score={gnn.flow_threat_score} tone="flow" />
        <ScoreRow label="Network threat score" score={gnn.network_threat_score} tone="net" />
      </div>

      {gnn.patterns_detected?.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs text-slate-300">
            <Network className="h-3.5 w-3.5 text-accent-purple" />
            Patterns detected
          </div>
          <ul className="space-y-1.5">
            {gnn.patterns_detected.map((p, i) => (
              <li
                key={i}
                className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-[11px]"
              >
                <div>
                  <span className="font-medium text-accent-purple">{p.type}</span>
                  {p.source && (
                    <span className="ml-2 font-mono text-slate-400">{p.source}</span>
                  )}
                </div>
                <span className="font-mono text-slate-300">
                  {(p.confidence * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {overrode && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 text-[11px] text-amber-100">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent-yellow" />
          <div>
            <div className="font-medium">
              Override from INSPECT → {gnn.recommendation}
            </div>
            {gnn.override_reason && (
              <div className="mt-0.5 text-slate-300">{gnn.override_reason}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
