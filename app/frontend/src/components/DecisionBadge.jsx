import { Check, Eye, Ban } from 'lucide-react'
import { decisionColor } from '../lib/api.js'

const ICONS = {
  ALLOW: Check,
  INSPECT: Eye,
  BLOCK: Ban,
  QUARANTINE: Ban,
}

export default function DecisionBadge({ decision }) {
  const Icon = ICONS[decision] || Eye
  return (
    <span className={`pill border ${decisionColor(decision)}`}>
      <Icon className="h-3 w-3" />
      {decision}
    </span>
  )
}
