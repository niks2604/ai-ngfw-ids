// Thin fetch wrappers for the AI-NGFW API. The Vite dev server proxies /api
// to http://localhost:8000 (see vite.config.js). In production, build output
// is served from the same origin as the API.

const BASE = '/api'

async function json(path, init) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}

export const api = {
  health: () => json('/health'),
  predict: (features, context) =>
    json('/predict', { method: 'POST', body: JSON.stringify({ features, context }) }),
  predictBatch: (flows) =>
    json('/predict/batch', { method: 'POST', body: JSON.stringify({ flows }) }),
  explain: (features, context) =>
    json('/predict/explain', { method: 'POST', body: JSON.stringify({ features, context }) }),

  // Demo controls — implemented in Step 9.
  demoStart: (scenario, speed) =>
    json('/demo/start', { method: 'POST', body: JSON.stringify({ scenario, speed }) }),
  demoStop: () => json('/demo/stop', { method: 'POST' }),
  demoStatus: () => json('/demo/status'),
  demoRecent: (limit = 100) => json(`/demo/recent?limit=${limit}`),
  demoStats: () => json('/demo/stats'),

  honeypotStatus: () => json('/honeypot/status'),
  honeypotCaptures: (limit = 200) => json(`/honeypot/captures?limit=${limit}`),
  honeypotCapture: (id) => json(`/honeypot/capture/${id}`),
  // Pass {is_real_attack: bool}; defaults to true for backwards compat.
  honeypotVerify: (id, isRealAttack = true) =>
    json(`/honeypot/verify/${id}`, {
      method: 'POST',
      body: JSON.stringify({ is_real_attack: isRealAttack }),
    }),
  honeypotTrainingQueue: () => json('/honeypot/training-queue'),

  // Model performance + retraining (feedback loop).
  modelMetrics: () => json('/model/metrics'),
  modelRetrain: () => json('/model/retrain', { method: 'POST' }),
  modelCrossDataset: () => json('/model/cross_dataset'),
  modelIterations: () => json('/model/iterations'),

  // Topology-aware GNN endpoints.
  predictGnn: (features, context) =>
    json('/predict/gnn', { method: 'POST', body: JSON.stringify({ features, context }) }),
  networkGraph: () => json('/network/graph'),
  networkThreats: () => json('/network/threats'),
}

export function decisionColor(decision) {
  switch (decision) {
    case 'ALLOW':
      return 'text-accent-green bg-emerald-500/10 border-emerald-500/30'
    case 'INSPECT':
      return 'text-accent-yellow bg-amber-500/10 border-amber-500/30'
    case 'BLOCK':
    case 'QUARANTINE':
      return 'text-accent-red bg-red-500/10 border-red-500/30'
    default:
      return 'text-slate-300 bg-slate-500/10 border-slate-500/30'
  }
}

export function trustColor(level) {
  switch (level) {
    case 'TRUSTED':
      return 'text-accent-green'
    case 'LOW_RISK':
      return 'text-emerald-300'
    case 'ELEVATED':
      return 'text-accent-yellow'
    case 'HIGH_RISK':
      return 'text-orange-400'
    case 'CRITICAL':
      return 'text-accent-red'
    default:
      return 'text-slate-300'
  }
}
