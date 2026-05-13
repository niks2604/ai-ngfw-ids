import { useEffect, useState } from 'react'
import { Activity, CheckCircle2, Clock, ShieldCheck, Terminal, FileDown, KeyRound } from 'lucide-react'
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

export default function Honeypot() {
  const { data: status } = usePolling(api.honeypotStatus, 3000)
  const { data: capturesData } = usePolling(() => api.honeypotCaptures(200), 3000)

  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [verifying, setVerifying] = useState(false)

  const captures = capturesData?.captures ?? []
  const fb = status?.feedback ?? { pending_review: 0, verified_attacks: 0, total_captured: 0 }

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    setDetailLoading(true)
    api
      .honeypotCapture(selectedId)
      .then((r) => setDetail(r))
      .catch((e) => setDetail({ error: e.message }))
      .finally(() => setDetailLoading(false))
  }, [selectedId])

  const verify = async (id) => {
    setVerifying(true)
    try {
      await api.honeypotVerify(id)
      const fresh = await api.honeypotCapture(id)
      setDetail(fresh)
    } catch (e) {
      setDetail({ ...(detail || {}), error: e.message })
    } finally {
      setVerifying(false)
    }
  }

  return (
    <div className="space-y-6">
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

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="glass overflow-hidden xl:col-span-2">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
            <div>
              <div className="text-sm font-medium">Captured Attacks</div>
              <div className="text-xs text-slate-400">
                Click a row to inspect the session
              </div>
            </div>
            <span className="pill border border-white/10 bg-white/5 text-slate-300">
              {captures.length} shown
            </span>
          </div>
          <div className="max-h-[70vh] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-bg-panel/90 backdrop-blur">
                <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                  <th className="px-5 py-2 font-medium">Time</th>
                  <th className="px-5 py-2 font-medium">Attacker IP</th>
                  <th className="px-5 py-2 font-medium">Attack Type</th>
                  <th className="px-5 py-2 font-medium">Duration</th>
                  <th className="px-5 py-2 font-medium">Commands</th>
                  <th className="px-5 py-2 font-medium">Decision</th>
                  <th className="px-5 py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {captures.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-5 py-8 text-center text-sm text-slate-500">
                      No captures yet — run the demo simulator to populate.
                    </td>
                  </tr>
                )}
                {captures.map((c) => (
                  <tr
                    key={c.id}
                    onClick={() => setSelectedId(c.id)}
                    className={`cursor-pointer border-t border-white/5 hover:bg-white/5 ${
                      selectedId === c.id ? 'bg-white/5' : ''
                    }`}
                  >
                    <td className="px-5 py-2 font-mono text-xs text-slate-400">
                      {fmtTime(c.timestamp)}
                    </td>
                    <td className="px-5 py-2 font-mono text-xs">{c.src_ip || '—'}</td>
                    <td className="px-5 py-2 text-xs">{c.attack_type}</td>
                    <td className="px-5 py-2 font-mono text-xs">
                      {fmtDuration(c.session_duration_seconds)}
                    </td>
                    <td className="px-5 py-2 font-mono text-xs">{c.commands_count}</td>
                    <td className="px-5 py-2">
                      <DecisionBadge decision={c.decision} />
                    </td>
                    <td className="px-5 py-2 text-xs">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedId(c.id)
                        }}
                        className="pill border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-4 xl:col-span-1">
          <div className="glass p-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-medium">Attack Details</div>
              {detail?.verified && (
                <span className="pill border border-emerald-500/30 bg-emerald-500/10 text-accent-green">
                  Verified
                </span>
              )}
            </div>
            {!selectedId && (
              <div className="text-xs text-slate-500">
                Select a capture from the table to see commands, files, and credentials.
              </div>
            )}
            {selectedId && detailLoading && (
              <div className="text-xs text-slate-500">Loading session…</div>
            )}
            {detail?.error && (
              <div className="text-xs text-accent-red">Error: {detail.error}</div>
            )}
            {detail && !detail.error && (
              <div className="space-y-4 text-xs">
                <div className="flex items-center gap-2">
                  <DecisionBadge decision={detail.decision} />
                  <span className="font-mono text-slate-300">
                    score {detail.risk_score?.toFixed(3)}
                  </span>
                </div>
                <div className="font-mono text-[11px] text-slate-400">
                  {detail.src_ip || '—'} → {detail.dst_ip || '—'}
                  {detail.dst_port ? `:${detail.dst_port}` : ''}
                </div>
                <div>
                  <div className="mb-1 flex items-center gap-1.5 text-slate-300">
                    <Terminal className="h-3.5 w-3.5" /> Commands
                  </div>
                  {detail.session?.commands?.length ? (
                    <ul className="space-y-1 rounded-xl border border-white/10 bg-white/5 p-3 font-mono text-[11px] text-slate-200">
                      {detail.session.commands.map((c, i) => (
                        <li key={i}>$ {c}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-slate-500">None captured.</div>
                  )}
                </div>
                <div>
                  <div className="mb-1 flex items-center gap-1.5 text-slate-300">
                    <FileDown className="h-3.5 w-3.5" /> Files downloaded
                  </div>
                  {detail.session?.files_downloaded?.length ? (
                    <ul className="space-y-0.5 font-mono text-[11px] text-slate-300">
                      {detail.session.files_downloaded.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-slate-500">None.</div>
                  )}
                </div>
                <div>
                  <div className="mb-1 flex items-center gap-1.5 text-slate-300">
                    <KeyRound className="h-3.5 w-3.5" /> Credentials tried
                  </div>
                  {detail.session?.credentials_tried?.length ? (
                    <ul className="space-y-0.5 font-mono text-[11px] text-slate-300">
                      {detail.session.credentials_tried.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-slate-500">None.</div>
                  )}
                </div>
                {detail.session?.log?.length > 0 && (
                  <details className="rounded-xl border border-white/10 bg-white/5 p-3">
                    <summary className="cursor-pointer text-slate-300">Full session log</summary>
                    <pre className="mt-2 whitespace-pre-wrap text-[11px] text-slate-400">
                      {detail.session.log.join('\n')}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>

          <div className="glass p-5">
            <div className="mb-3 text-sm font-medium">Feedback Loop</div>
            <ul className="space-y-1.5 text-xs text-slate-300">
              <li className="flex items-center justify-between">
                <span>Pending review</span>
                <span className="font-mono text-accent-yellow">{fb.pending_review}</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Verified attacks</span>
                <span className="font-mono text-accent-green">{fb.verified_attacks}</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Model last retrained</span>
                <span className="font-mono text-slate-400">
                  {fmtTime(status?.last_retrained_at)}
                </span>
              </li>
            </ul>
            <button
              type="button"
              disabled={!selectedId || detail?.verified || verifying}
              onClick={() => selectedId && verify(selectedId)}
              className="mt-4 w-full rounded-xl border border-white/10 bg-gradient-to-r from-accent-blue/30 to-accent-purple/30 px-3 py-2 text-xs font-medium text-slate-100 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {verifying
                ? 'Verifying…'
                : detail?.verified
                ? 'Already verified'
                : selectedId
                ? 'Verify & Add to Training'
                : 'Select a capture to verify'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
