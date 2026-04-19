import {
  Check,
  Eye,
  Ban,
  Activity,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import StatCard from '../components/StatCard.jsx'
import DemoControls from '../components/DemoControls.jsx'
import { usePolling } from '../hooks/usePolling.js'
import { api } from '../lib/api.js'

const DONUT_COLORS = { ALLOW: '#10b981', INSPECT: '#f59e0b', BLOCK: '#ef4444', QUARANTINE: '#b91c1c' }

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="mb-1 text-slate-400">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-slate-300">{p.name}</span>
          <span className="ml-auto font-mono text-slate-100">{p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function Overview() {
  const { data: stats } = usePolling(api.demoStats, 2000)

  const counts = stats?.counts || { ALLOW: 0, INSPECT: 0, BLOCK: 0 }
  const total = (counts.ALLOW || 0) + (counts.INSPECT || 0) + (counts.BLOCK || 0)
  const flowsPerSec = stats?.flows_per_sec ?? 0
  const timeline = stats?.timeline ?? [] // [{ts, ALLOW, INSPECT, BLOCK}, ...]
  const donut = Object.entries(counts).map(([name, value]) => ({ name, value }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Check}
          label="Allowed"
          value={(counts.ALLOW || 0).toLocaleString()}
          tone="green"
          hint={total ? `${((100 * (counts.ALLOW || 0)) / total).toFixed(1)}% of traffic` : '—'}
        />
        <StatCard
          icon={Eye}
          label="Inspecting"
          value={(counts.INSPECT || 0).toLocaleString()}
          tone="yellow"
          hint={total ? `${((100 * (counts.INSPECT || 0)) / total).toFixed(1)}% of traffic` : '—'}
        />
        <StatCard
          icon={Ban}
          label="Blocked"
          value={(counts.BLOCK || 0).toLocaleString()}
          tone="red"
          hint={total ? `${((100 * (counts.BLOCK || 0)) / total).toFixed(1)}% of traffic` : '—'}
        />
        <StatCard
          icon={Activity}
          label="Flows / sec"
          value={flowsPerSec.toFixed(1)}
          tone="blue"
          hint={total ? `${total.toLocaleString()} total scored` : 'start the demo →'}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="glass col-span-1 p-5 xl:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <div className="text-sm font-medium">Threats over time</div>
              <div className="text-xs text-slate-400">Rolling window of scored flows</div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer>
              <AreaChart data={timeline}>
                <defs>
                  <linearGradient id="gAllow" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gInspect" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gBlock" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="ts" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area dataKey="ALLOW" stroke="#10b981" fill="url(#gAllow)" stackId="1" />
                <Area dataKey="INSPECT" stroke="#f59e0b" fill="url(#gInspect)" stackId="1" />
                <Area dataKey="BLOCK" stroke="#ef4444" fill="url(#gBlock)" stackId="1" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass p-5">
          <div className="mb-4">
            <div className="text-sm font-medium">Traffic distribution</div>
            <div className="text-xs text-slate-400">Decision mix</div>
          </div>
          <div className="h-64">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={donut}
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={3}
                  dataKey="value"
                  stroke="none"
                >
                  {donut.map((d) => (
                    <Cell key={d.name} fill={DONUT_COLORS[d.name] || '#64748b'} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-1">
          <DemoControls />
        </div>
      </div>
    </div>
  )
}
