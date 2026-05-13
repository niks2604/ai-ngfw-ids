import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Activity, Brain, ShieldAlert, Shield } from 'lucide-react'

const HoneypotIcon = (props) => (
  <span {...props} role="img" aria-label="honeypot">🍯</span>
)

const items = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/flows', icon: Activity, label: 'Live Flows' },
  { to: '/explain', icon: Brain, label: 'Explainability' },
  { to: '/threats', icon: ShieldAlert, label: 'Threat Analytics' },
  { to: '/honeypot', icon: HoneypotIcon, label: 'Honeypot' },
]

export default function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col gap-6 p-5 md:flex">
      <div className="flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple shadow-lg">
          <Shield className="h-5 w-5 text-white" />
        </div>
        <div>
          <div className="text-sm font-semibold">AI-NGFW</div>
          <div className="text-xs text-slate-400">Control Plane</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {items.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `nav-link ${isActive ? 'nav-link-active' : ''}`
            }
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-3 text-xs text-slate-400">
        <div className="font-medium text-slate-200">Ensemble</div>
        <div>RF · XGBoost · IsolationForest</div>
        <div className="mt-1 text-[11px]">Thresholds 0.30 / 0.70</div>
      </div>
    </aside>
  )
}
