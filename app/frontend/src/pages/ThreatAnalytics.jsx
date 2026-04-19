import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ShieldAlert } from 'lucide-react'
import { usePolling } from '../hooks/usePolling.js'
import { api } from '../lib/api.js'

function Heatmap({ matrix }) {
  // matrix: {hour, day, count}[]  — day 0..6, hour 0..23
  const cell = (d, h) => matrix?.find((m) => m.day === d && m.hour === h)?.count ?? 0
  const max = Math.max(1, ...(matrix || []).map((m) => m.count))
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-[2px]">
        <tbody>
          {days.map((label, d) => (
            <tr key={d}>
              <td className="pr-2 text-[10px] text-slate-500">{label}</td>
              {Array.from({ length: 24 }).map((_, h) => {
                const c = cell(d, h)
                const alpha = 0.08 + 0.92 * (c / max)
                return (
                  <td
                    key={h}
                    title={`${label} ${h}:00 · ${c}`}
                    className="h-4 w-4 rounded-[3px]"
                    style={{ background: `rgba(239, 68, 68, ${alpha})` }}
                  />
                )
              })}
            </tr>
          ))}
          <tr>
            <td />
            {Array.from({ length: 24 }).map((_, h) => (
              <td key={h} className="text-center text-[9px] text-slate-600">
                {h % 3 === 0 ? h : ''}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  )
}

export default function ThreatAnalytics() {
  const { data } = usePolling(api.demoStats, 3000)

  const attackTypes = Object.entries(data?.attack_types || {}).map(([name, count]) => ({
    name,
    count,
  }))
  const topIps = (data?.top_blocked_ips || []).slice(0, 10)
  const heatmap = data?.heatmap || []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="glass p-5 xl:col-span-2">
          <div className="mb-4 flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-accent-red" />
            <div>
              <div className="text-sm font-medium">Attack type distribution</div>
              <div className="text-xs text-slate-400">From blocked + inspect decisions</div>
            </div>
          </div>
          <div className="h-64">
            {attackTypes.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                No attacks scored yet.
              </div>
            ) : (
              <ResponsiveContainer>
                <BarChart data={attackTypes}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                    contentStyle={{
                      background: 'rgba(17,24,39,0.95)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="count" fill="#ef4444" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="glass p-5">
          <div className="mb-4 text-sm font-medium">Top blocked IPs</div>
          {topIps.length === 0 ? (
            <div className="text-xs text-slate-500">No blocked traffic yet.</div>
          ) : (
            <ul className="space-y-2 text-sm">
              {topIps.map(([ip, n]) => (
                <li key={ip} className="flex items-center justify-between">
                  <span className="font-mono text-slate-300">{ip}</span>
                  <span className="font-mono text-xs text-accent-red">{n}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="glass p-5">
        <div className="mb-4 text-sm font-medium">Threat timeline heatmap</div>
        <div className="text-xs text-slate-400">
          Blocked-flow count by day-of-week × hour-of-day
        </div>
        <div className="mt-4">
          <Heatmap matrix={heatmap} />
        </div>
      </div>
    </div>
  )
}
