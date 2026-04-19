export default function StatCard({ icon: Icon, label, value, tone = 'blue', hint }) {
  const tones = {
    green: 'from-emerald-500/20 to-emerald-500/5 text-accent-green',
    yellow: 'from-amber-500/20 to-amber-500/5 text-accent-yellow',
    red: 'from-red-500/20 to-red-500/5 text-accent-red',
    blue: 'from-sky-500/20 to-sky-500/5 text-accent-blue',
    purple: 'from-violet-500/20 to-violet-500/5 text-accent-purple',
  }
  return (
    <div className="glass relative overflow-hidden p-5">
      <div
        className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${tones[tone]}`}
      />
      <div className="relative flex items-start justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
          <div className={`stat-number mt-2 ${tones[tone].split(' ').pop()}`}>
            {value}
          </div>
          {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
        </div>
        {Icon && <Icon className={`h-5 w-5 ${tones[tone].split(' ').pop()}`} />}
      </div>
    </div>
  )
}
