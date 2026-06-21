import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  CheckCircle2,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  ShieldCheck,
  Terminal,
  FileDown,
  KeyRound,
  X,
  ArrowRight,
  AlertTriangle,
  Globe,
  Home,
} from 'lucide-react'
import StatCard from '../components/StatCard.jsx'
import DecisionBadge from '../components/DecisionBadge.jsx'
import { usePolling } from '../hooks/usePolling.js'
import { api } from '../lib/api.js'

function fmtTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function fmtDuration(seconds) {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}m ${s}s`
}

function isInternalIp(ip) {
  if (!ip) return false
  return (
    ip.startsWith('10.') ||
    ip.startsWith('192.168.') ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(ip)
  )
}

// Pull the few feature columns we actually want to surface in the
// "Traffic Characteristics" panel. CICFlowMeter column names.
function summarizeFeatures(features) {
  const f = features || {}
  const durMicros = f['Flow Duration']
  const fwd = (f['Total Fwd Packets'] ?? 0) + (f['Total Backward Packets'] ?? 0)
  const bytes =
    (f['Total Length of Fwd Packets'] ?? 0) + (f['Total Length of Bwd Packets'] ?? 0)
  const pktRate = f['Flow Packets/s']
  return {
    durationSec: durMicros ? durMicros / 1_000_000 : null,
    packets: Math.round(fwd) || null,
    bytes: Math.round(bytes) || null,
    pktRate: pktRate ?? null,
  }
}

// Hand-rolled rule-of-thumb mapping: given an attack type, surface 1–2 known
// patterns it tends to match. Keeps the UI useful even when SHAP is offline.
function similarAttacks(attackType) {
  const key = (attackType || '').toLowerCase()
  if (key.includes('ddos') || key.includes('dos')) {
    return [
      { name: 'DDoS signature', similarity: 0.87 },
      { name: 'SYN flood pattern', similarity: 0.72 },
    ]
  }
  if (key.includes('brute') || key.includes('patator') || key.includes('password')) {
    return [
      { name: 'SSH brute-force', similarity: 0.83 },
      { name: 'Credential stuffing', similarity: 0.68 },
    ]
  }
  if (key.includes('scan') || key.includes('portscan')) {
    return [
      { name: 'Nmap port scan', similarity: 0.91 },
      { name: 'Service enumeration', similarity: 0.62 },
    ]
  }
  if (key.includes('infiltration') || key.includes('botnet') || key.includes('c&c')) {
    return [
      { name: 'C2 beacon pattern', similarity: 0.79 },
      { name: 'Lateral movement', similarity: 0.55 },
    ]
  }
  if (key.includes('web') || key.includes('xss') || key.includes('sqli') || key.includes('injection')) {
    return [
      { name: 'SQL injection signature', similarity: 0.74 },
      { name: 'Web application exploit', similarity: 0.66 },
    ]
  }
  if (key === 'benign' || key === 'unknown') return []
  return [{ name: `Generic ${attackType}`, similarity: 0.55 }]
}

function ShapBar({ feature, shap }) {
  const pct = Math.min(100, Math.abs(shap) * 200)
  const isUp = shap >= 0
  const color = isUp ? '#ef4444' : '#10b981'
  return (
    <div className="flex items-center gap-3 text-[11px]">
      <div className="w-44 truncate text-slate-300" title={feature}>
        {feature}
      </div>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="w-14 text-right font-mono" style={{ color }}>
        {isUp ? '+' : '−'}
        {Math.abs(shap).toFixed(3)}
      </div>
    </div>
  )
}

function EndpointCard({ title, ip, port, icon: Icon, accent }) {
  const internal = isInternalIp(ip)
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-wide text-slate-400">
        <Icon className="h-3.5 w-3.5" /> {title}
      </div>
      <div className="space-y-0.5 text-xs">
        <div>
          <span className="text-slate-500">IP: </span>
          <span className="font-mono text-slate-100">{ip || '—'}</span>
        </div>
        <div>
          <span className="text-slate-500">Port: </span>
          <span className="font-mono text-slate-100">{port ?? '—'}</span>
        </div>
        <div className="text-[10px] text-slate-400">
          {internal ? (
            <span className="inline-flex items-center gap-1">
              <Home className="h-3 w-3" /> Internal
            </span>
          ) : (
            <span className="inline-flex items-center gap-1">
              <Globe className="h-3 w-3" /> External {accent ? `(${accent})` : ''}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function CaptureCard({ capture, onVerify, verifying }) {
  // capture.is_real_attack is server-truth: true = real attack,
  // false = false positive, null/undefined = pending. Always compare
  // with === so a stale null never falls into the "false positive"
  // bucket via a truthy check.
  const isVerified = capture.verified === true
  const isRealAttack = capture.is_real_attack === true
  const isFalsePositive = capture.verified === true && capture.is_real_attack === false

  const [open, setOpen] = useState(!isVerified)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [shap, setShap] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)

  useEffect(() => {
    if (!open || detail) return
    setDetailLoading(true)
    api
      .honeypotCapture(capture.id)
      .then((d) => {
        setDetail(d)
        // Kick off SHAP if we have features.
        if (d.flow_features && Object.keys(d.flow_features).length > 0) {
          setShapLoading(true)
          api
            .explain(d.flow_features, {
              src_ip: d.src_ip,
              dst_port: d.dst_port,
              attack_type: d.attack_type,
            })
            .then((r) => setShap(r))
            .catch((e) => setShap({ error: e.message }))
            .finally(() => setShapLoading(false))
        }
      })
      .catch((e) => setDetail({ error: e.message }))
      .finally(() => setDetailLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const summary = summarizeFeatures(detail?.flow_features)
  const sims = similarAttacks(capture.attack_type)
  const score = capture.risk_score ?? detail?.risk_score
  // Render the prompt OR the verified result — but always from the
  // explicit === true / === false split, never a truthy fallthrough.
  let decisionPrompt
  if (!isVerified) {
    decisionPrompt = (
      <div className="rounded-xl border border-white/15 bg-bg-panel/60 p-3 text-[11px]">
        <div className="mb-2 flex items-center gap-1.5 text-slate-200">
          <AlertTriangle className="h-3.5 w-3.5 text-accent-yellow" /> Is this a real attack?
        </div>
        <ul className="mb-3 space-y-0.5 text-slate-400">
          <li>
            • {isInternalIp(capture.src_ip) ? 'Internal source — possibly legitimate' : 'External source'}
          </li>
          {summary.pktRate != null && (
            <li>
              • Packet rate {summary.pktRate.toFixed(0)}/s{' '}
              {summary.pktRate > 1000 ? '— abnormal' : '— within normal range'}
            </li>
          )}
          {sims[0] && (
            <li>
              • Matches {sims[0].name} ({(sims[0].similarity * 100).toFixed(0)}% similar)
            </li>
          )}
          <li>• Targeting {capture.dst_ip || 'service'} on port {capture.dst_port ?? '—'}</li>
        </ul>
        <div className="flex gap-2">
          <button
            type="button"
            disabled={verifying}
            onClick={() => onVerify(capture.id, true)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs font-medium text-accent-green transition hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Check className="h-3.5 w-3.5" /> YES, Real Attack
          </button>
          <button
            type="button"
            disabled={verifying}
            onClick={() => onVerify(capture.id, false)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs font-medium text-accent-red transition hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <X className="h-3.5 w-3.5" /> NO, False Positive
          </button>
        </div>
      </div>
    )
  } else if (isRealAttack) {
    decisionPrompt = (
      <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-[11px] text-accent-green">
        ✅ Verified as Real Attack — queued for next retrain.
      </div>
    )
  } else if (isFalsePositive) {
    decisionPrompt = (
      <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-[11px] text-accent-red">
        ❌ Marked as False Positive — used as negative example.
      </div>
    )
  } else {
    decisionPrompt = (
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-[11px] text-accent-yellow">
        Awaiting server confirmation…
      </div>
    )
  }

  return (
    <div className="glass overflow-hidden">
      {/* HEADER (always visible, clickable to expand/collapse) ----------- */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-b border-white/5 px-5 py-3 text-left transition hover:bg-white/5"
      >
        <div className="flex items-center gap-3">
          {open ? (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-slate-400" />
          )}
          <div>
            <div className="flex items-center gap-2 text-sm font-medium">
              <span className="font-mono text-slate-500">#{capture.id.slice(0, 6)}</span>
              <span className="text-slate-200">{capture.attack_type || 'Unknown'}</span>
            </div>
            <div className="font-mono text-[11px] text-slate-400">
              {capture.src_ip || '—'} → {capture.dst_ip || '—'}
              {capture.dst_port ? `:${capture.dst_port}` : ''} ·{' '}
              {fmtTime(capture.timestamp)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-slate-300">
            score {score != null ? score.toFixed(2) : '—'}
          </span>
          <DecisionBadge decision={capture.decision} />
          {isRealAttack ? (
            <span className="pill border border-emerald-500/30 bg-emerald-500/10 text-[10px] font-medium text-accent-green">
              Verified ✓
            </span>
          ) : isFalsePositive ? (
            <span className="pill border border-red-500/30 bg-red-500/10 text-[10px] font-medium text-accent-red">
              False Positive ✗
            </span>
          ) : (
            <span className="pill border border-amber-500/30 bg-amber-500/10 text-[10px] font-medium text-accent-yellow">
              Pending
            </span>
          )}
        </div>
      </button>

      {/* BODY -------------------------------------------------------- */}
      {open && (
        <div className="space-y-4 p-5">
          {detailLoading && (
            <div className="text-xs text-slate-500">Loading capture detail…</div>
          )}
          {detail?.error && (
            <div className="text-xs text-accent-red">Error: {detail.error}</div>
          )}

          {/* Source / Destination cards */}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <EndpointCard
              title="Source"
              icon={Globe}
              ip={capture.src_ip}
              port={detail?.session?.src_port}
            />
            <EndpointCard
              title="Destination"
              icon={Home}
              ip={capture.dst_ip}
              port={capture.dst_port}
              accent="your service"
            />
          </div>

          {/* Traffic characteristics */}
          <div>
            <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-200">
              <Activity className="h-3.5 w-3.5" /> Traffic characteristics
            </div>
            <ul className="space-y-0.5 text-[11px] text-slate-300">
              <li>
                <span className="text-slate-500">├ Protocol: </span>
                {detail?.protocol ?? capture.protocol ?? 'TCP'}
              </li>
              <li>
                <span className="text-slate-500">├ Duration: </span>
                {summary.durationSec != null
                  ? `${summary.durationSec.toFixed(2)} s${
                      summary.durationSec < 1 ? ' (very short)' : ''
                    }`
                  : fmtDuration(capture.session_duration_seconds)}
              </li>
              <li>
                <span className="text-slate-500">├ Packets: </span>
                {summary.packets != null
                  ? `${summary.packets.toLocaleString()}${
                      summary.packets > 500 ? ' (very high)' : ''
                    }`
                  : '—'}
              </li>
              <li>
                <span className="text-slate-500">├ Bytes: </span>
                {summary.bytes != null ? summary.bytes.toLocaleString() : '—'}
              </li>
              <li className="flex items-center gap-2">
                <span className="text-slate-500">└ Packet Rate: </span>
                <span className={summary.pktRate > 1000 ? 'text-accent-red' : 'text-slate-300'}>
                  {summary.pktRate != null ? `${summary.pktRate.toFixed(0)}/s` : '—'}
                </span>
                {summary.pktRate != null &&
                  (summary.pktRate > 1000 ? (
                    <span className="text-[10px] font-medium text-accent-red">⚠ ABNORMAL</span>
                  ) : (
                    <span className="text-[10px] text-accent-green">✓ normal</span>
                  ))}
              </li>
            </ul>
          </div>

          {/* WHY IT WAS BLOCKED (SHAP) */}
          <div>
            <div className="mb-2 text-xs font-medium text-slate-200">
              Why it was blocked (SHAP)
            </div>
            {shapLoading && (
              <div className="text-[11px] text-slate-500">Computing SHAP…</div>
            )}
            {shap?.error && (
              <div className="text-[11px] text-accent-red">SHAP error: {shap.error}</div>
            )}
            {shap?.top_features?.length > 0 && (
              <div className="space-y-1.5">
                {shap.top_features.slice(0, 5).map((tf) => (
                  <ShapBar
                    key={tf.feature}
                    feature={tf.feature}
                    shap={
                      tf.direction === '+'
                        ? Math.abs(tf.shap_value)
                        : -Math.abs(tf.shap_value)
                    }
                  />
                ))}
              </div>
            )}
            {!shap && !shapLoading && detail && (
              <div className="text-[11px] text-slate-500">
                No stored features for this capture — SHAP not available.
              </div>
            )}
            {shap?.explanation && (
              <div className="mt-2 rounded-lg border border-white/10 bg-white/5 p-2 text-[11px] text-slate-300">
                {shap.explanation}
              </div>
            )}
          </div>

          {/* Similar known attacks */}
          {sims.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-medium text-slate-200">
                Similar known attacks
              </div>
              <ul className="space-y-1 text-[11px]">
                {sims.map((s, i) => (
                  <li
                    key={i}
                    className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5"
                  >
                    <span className="text-slate-300">
                      {i === 0 ? '├' : i === sims.length - 1 ? '└' : '├'} {s.name}
                    </span>
                    <span className="font-mono text-slate-400">
                      {(s.similarity * 100).toFixed(0)}% similar
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Session commands / files / credentials */}
          {detail?.session && (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div>
                <div className="mb-1 flex items-center gap-1.5 text-[11px] text-slate-300">
                  <Terminal className="h-3 w-3" /> Commands
                </div>
                {detail.session.commands?.length ? (
                  <ul className="space-y-0.5 rounded-lg border border-white/10 bg-white/5 p-2 font-mono text-[10px] text-slate-200">
                    {detail.session.commands.slice(0, 4).map((c, i) => (
                      <li key={i}>$ {c}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-[10px] text-slate-500">None.</div>
                )}
              </div>
              <div>
                <div className="mb-1 flex items-center gap-1.5 text-[11px] text-slate-300">
                  <FileDown className="h-3 w-3" /> Files
                </div>
                {detail.session.files_downloaded?.length ? (
                  <ul className="space-y-0.5 font-mono text-[10px] text-slate-300">
                    {detail.session.files_downloaded.slice(0, 4).map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-[10px] text-slate-500">None.</div>
                )}
              </div>
              <div>
                <div className="mb-1 flex items-center gap-1.5 text-[11px] text-slate-300">
                  <KeyRound className="h-3 w-3" /> Credentials
                </div>
                {detail.session.credentials_tried?.length ? (
                  <ul className="space-y-0.5 font-mono text-[10px] text-slate-300">
                    {detail.session.credentials_tried.slice(0, 4).map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-[10px] text-slate-500">None.</div>
                )}
              </div>
            </div>
          )}

          {/* Decision prompt */}
          {decisionPrompt}
        </div>
      )}
    </div>
  )
}

export default function Honeypot() {
  const { data: status, refresh: refreshStatus } = usePolling(
    api.honeypotStatus, 3000,
  )
  const { data: capturesData, refresh: refreshCaptures } = usePolling(
    () => api.honeypotCaptures(200), 3000,
  )
  const { data: queueData, refresh: refreshQueue } = usePolling(
    api.honeypotTrainingQueue, 3000,
  )

  const [verifyingId, setVerifyingId] = useState(null)
  const [verifyError, setVerifyError] = useState(null)
  // Optimistic overrides: id -> is_real_attack (true/false). Merged
  // onto polled captures so the UI reflects the click immediately,
  // before the 3 s polling tick refreshes server state.
  const [pendingVerify, setPendingVerify] = useState({})

  const rawCaptures = capturesData?.captures ?? []
  const captures = rawCaptures.map((c) =>
    Object.prototype.hasOwnProperty.call(pendingVerify, c.id)
      ? { ...c, verified: true, is_real_attack: pendingVerify[c.id] }
      : c,
  )
  const fb = status?.feedback ?? {
    pending_review: 0,
    verified_attacks: 0,
    false_positives: 0,
    training_queue: 0,
    total_captured: 0,
  }
  const queueSize = queueData?.count ?? fb.training_queue ?? 0

  const verify = async (id, isRealAttack) => {
    // Optimistic update — paint the verified state instantly.
    setPendingVerify((m) => ({ ...m, [id]: isRealAttack === true }))
    setVerifyingId(id)
    setVerifyError(null)
    try {
      const result = await api.honeypotVerify(id, isRealAttack === true)
      console.debug('[verify]', id, '→', result)
      // Force-refresh the three pollers right now so the queue banner,
      // captures list, and feedback counts reflect the new state
      // without waiting up to 3 s for the next scheduled tick.
      await Promise.allSettled([
        refreshStatus(),
        refreshCaptures(),
        refreshQueue(),
      ])
    } catch (e) {
      setVerifyError(e.message)
      console.error('[verify failed]', e)
      // Roll back the optimistic update on failure.
      setPendingVerify((m) => {
        const next = { ...m }
        delete next[id]
        return next
      })
    } finally {
      setVerifyingId(null)
    }
  }

  return (
    <div className="space-y-6">
      {/* Training-queue banner */}
      <div className="glass flex items-center justify-between gap-4 px-5 py-3">
        <div className="flex items-center gap-3 text-sm">
          <CheckCircle2 className="h-5 w-5 text-accent-green" />
          <div>
            <div className="font-medium text-slate-100">
              Training queue:{' '}
              <span className="text-accent-green">{queueSize.toLocaleString()}</span>{' '}
              capture{queueSize === 1 ? '' : 's'} ready
            </div>
            <div className="text-xs text-slate-400">
              Confirmed real attacks waiting for the next model retrain.
            </div>
          </div>
        </div>
        <Link
          to="/threats"
          className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-gradient-to-r from-accent-purple/30 to-accent-blue/30 px-3 py-1.5 text-xs font-medium text-slate-100 transition hover:brightness-110"
        >
          Open Threat Analytics
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={ShieldCheck}
          label="Honeypot"
          value={status?.online ? 'Online' : 'Offline'}
          tone={status?.online ? 'green' : 'red'}
          hint={status?.backend?.backend ? `backend: ${status.backend.backend}` : '—'}
        />
        <StatCard
          icon={Activity}
          label="Captured Sessions"
          value={(status?.captured_sessions ?? 0).toLocaleString()}
          tone="purple"
          hint={
            status?.unique_sources
              ? `${status.unique_sources} unique sources`
              : 'no captures yet'
          }
        />
        <StatCard
          icon={Clock}
          label="Last Capture"
          value={status?.last_capture ? fmtTime(status.last_capture) : '—'}
          tone="blue"
          hint={status?.last_retrained_at ? `model: ${fmtTime(status.last_retrained_at)}` : ''}
        />
        <StatCard
          icon={CheckCircle2}
          label="Verified Attacks"
          value={fb.verified_attacks.toLocaleString()}
          tone="yellow"
          hint={`${fb.pending_review} pending review`}
        />
      </div>

      <div>
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-slate-100">
              🍯 Honeypot Captures — Verify Blocked Traffic
            </div>
            <div className="text-xs text-slate-400">
              {captures.length} capture{captures.length === 1 ? '' : 's'} ·{' '}
              {fb.pending_review} pending · expand a card for full evidence
            </div>
          </div>
          {verifyError && (
            <span className="text-xs text-accent-red">Verify error: {verifyError}</span>
          )}
        </div>

        {captures.length === 0 ? (
          <div className="glass p-8 text-center text-sm text-slate-500">
            No captures yet — run the demo simulator to populate.
          </div>
        ) : (
          <div className="space-y-3">
            {captures.map((c) => (
              <CaptureCard
                key={c.id}
                capture={c}
                onVerify={verify}
                verifying={verifyingId === c.id}
              />
            ))}
          </div>
        )}
      </div>

      <div className="glass p-5">
        <div className="mb-3 text-sm font-medium">Feedback loop</div>
        <ul className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs text-slate-300 md:grid-cols-5">
          <li className="flex items-center justify-between">
            <span>Pending review</span>
            <span className="font-mono text-accent-yellow">{fb.pending_review}</span>
          </li>
          <li className="flex items-center justify-between">
            <span>Verified attacks</span>
            <span className="font-mono text-accent-green">{fb.verified_attacks}</span>
          </li>
          <li className="flex items-center justify-between">
            <span>False positives</span>
            <span className="font-mono text-slate-400">{fb.false_positives ?? 0}</span>
          </li>
          <li className="flex items-center justify-between">
            <span>Training queue</span>
            <span className="font-mono text-accent-blue">{queueSize}</span>
          </li>
          <li className="flex items-center justify-between">
            <span>Last retrained</span>
            <span className="font-mono text-slate-400">
              {fmtTime(status?.last_retrained_at)}
            </span>
          </li>
        </ul>
      </div>
    </div>
  )
}
