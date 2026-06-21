"""
AI-NGFW/IDS Inference Service (FastAPI)

Endpoints
---------
GET  /health            -> service + model status
POST /predict           -> single flow decision
POST /predict/batch     -> batch flow decisions
POST /predict/explain   -> decision + SHAP explanation

POST /demo/start        -> start the traffic simulator (body: scenario, speed)
POST /demo/stop         -> stop the traffic simulator
GET  /demo/status       -> simulator status
GET  /demo/recent       -> recent decisions (ring buffer)
GET  /demo/stats        -> aggregated decision stats / timeline / heatmap

Decision thresholds (on ensemble risk score, 0.0-1.0):
    score <  0.3  -> ALLOW
    0.3  <= score < 0.7  -> INSPECT
    score >= 0.7 -> BLOCK
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

from app.api.model_metrics import store as metrics_store
from app.honeypot.honeypot_manager import manager as honeypot_manager
from app.models.ensemble import EnsembleDetector
from app.models.gnn import GNNDetector
from app.simulator.traffic_simulator import TrafficSimulator
from app.zero_trust.zero_trust import FlowContext, ZeroTrustEngine


MODELS_PATH = os.environ.get(
    "NGFW_MODELS_PATH",
    os.path.expanduser("~/sem6el/trained_models/"),
)

# Decision thresholds (override ensemble.py defaults per project spec).
THRESHOLD_INSPECT = 0.3
THRESHOLD_BLOCK = 0.7


def score_to_decision(score: float) -> str:
    if score >= THRESHOLD_BLOCK:
        return "BLOCK"
    if score >= THRESHOLD_INSPECT:
        return "INSPECT"
    return "ALLOW"


class ServiceState:
    ensemble: EnsembleDetector | None = None
    feature_columns: list[str] | None = None
    label_classes: list[str] | None = None
    explainer: Any = None  # populated lazily on first /predict/explain call
    zero_trust: ZeroTrustEngine | None = None
    simulator: TrafficSimulator | None = None
    gnn: GNNDetector | None = None
    loaded_at: float | None = None


state = ServiceState()


def _flows_to_frame(flows: list[dict[str, float]]) -> pd.DataFrame:
    """Align incoming flow dicts to the training feature schema.

    Missing features default to 0.0; extras are dropped. NaN/inf are scrubbed
    to match training-time preprocessing.
    """
    if state.feature_columns is None:
        raise RuntimeError("feature columns not loaded")
    df = pd.DataFrame(flows)
    for col in state.feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[state.feature_columns].astype(float)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensemble = EnsembleDetector(models_path=MODELS_PATH)
    ensemble.load_models()
    state.ensemble = ensemble
    state.feature_columns = joblib.load(
        os.path.join(MODELS_PATH, "feature_columns.joblib")
    )
    try:
        le = joblib.load(os.path.join(MODELS_PATH, "label_encoder.joblib"))
        state.label_classes = list(le.classes_)
    except FileNotFoundError:
        state.label_classes = None
    state.zero_trust = ZeroTrustEngine()
    state.simulator = TrafficSimulator(predict_fn=_sim_predict)
    state.gnn = GNNDetector()
    state.gnn.try_load()  # best-effort; falls back to topology heuristics
    state.loaded_at = time.time()
    # Surface the most recent model artefact mtime as "last retrained".
    try:
        rf_path = os.path.join(MODELS_PATH, "random_forest.joblib")
        mtime = os.path.getmtime(rf_path)
        from datetime import datetime, timezone
        honeypot_manager.last_retrained_at = (
            datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        )
    except OSError:
        pass
    yield
    if state.simulator and state.simulator.running:
        state.simulator.stop()
    state.ensemble = None


app = FastAPI(
    title="AI-NGFW/IDS Inference Service",
    description="Ensemble (RF + XGBoost + IsolationForest) network threat detector.",
    version="0.1.0",
    lifespan=lifespan,
)

# Dev-friendly CORS — Vite serves the dashboard on :5173 during development;
# production deployments should tighten this to the actual origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---------------------------------------------------------------


class FlowContextIn(BaseModel):
    """Optional context passed to the Zero Trust layer."""
    src_ip: str | None = None
    dst_port: int | None = None
    is_internal_src: bool = False
    is_authenticated: bool = False
    prior_violations: int = 0
    session_risk: float = 0.0
    asset_sensitivity: str = "normal"
    # Hint surfaced to the honeypot capture path; not part of Zero Trust input.
    attack_type: str | None = None


class FlowIn(BaseModel):
    """Single flow. Feature keys should match CICIDS2017 column names.

    Unknown keys are dropped; missing expected features default to 0.
    """

    features: dict[str, float] = Field(
        ..., description="Feature name -> value. Unknown keys ignored."
    )
    context: FlowContextIn | None = None


class BatchFlow(BaseModel):
    features: dict[str, float]
    context: FlowContextIn | None = None


class BatchIn(BaseModel):
    flows: list[dict[str, float] | BatchFlow] = Field(..., min_length=1, max_length=10_000)


class ModelScores(BaseModel):
    isolation_forest: float
    random_forest: float
    xgboost: float


class ZeroTrustOut(BaseModel):
    trust_level: str
    primary_action: str
    actions: list[str]            # flat list: [primary, *secondary] for UI
    recommendations: list[str]
    principles_applied: list[str]
    effective_risk: float


class EnsembleScores(BaseModel):
    """Spec-aligned aliases for the three ensemble member scores."""
    rf_score: float
    xgb_score: float
    if_score: float


class GNNPattern(BaseModel):
    type: str                     # POTENTIAL_SCAN | POTENTIAL_DDOS | POTENTIAL_C2
    source: str | None = None
    confidence: float


class GNNAnalysisOut(BaseModel):
    flow_threat_score: float       # endpoint risk for this flow
    network_threat_score: float    # whole-window graph score
    total_nodes: int
    total_edges: int
    patterns_detected: list[GNNPattern]
    recommendation: str            # ALLOW | INSPECT | BLOCK
    override_reason: str | None = None


class PredictionOut(BaseModel):
    decision: str
    risk_score: float
    score: float                  # alias of risk_score for frontend convenience
    ensemble_score: float         # spec alias of risk_score
    model_scores: ModelScores
    ensemble: EnsembleScores      # spec alias of model_scores with rf/xgb/if names
    thresholds: dict[str, float]
    zero_trust: ZeroTrustOut
    gnn_analyzed: bool = False
    gnn: GNNAnalysisOut | None = None
    honeypot_captured: bool = False
    explanation: dict[str, Any] | None = None


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    count: int
    inference_ms: float


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "+" pushes toward attack, "-" toward benign


class ExplainOut(PredictionOut):
    top_features: list[FeatureContribution]
    explanation: str


class GNNContribution(BaseModel):
    """Per-flow GNN view: combined score, endpoint risk, detected pattern."""
    combined_score: float
    ensemble_score: float
    gnn_endpoint_risk: float
    gnn_graph_score: float
    gnn_node_score_src: float
    gnn_node_score_dst: float
    detected_pattern: str | None
    source: str  # "gnn" if model loaded, "heuristic" otherwise
    applied: bool  # True only when the ensemble decision was INSPECT


class GNNPredictionOut(PredictionOut):
    gnn: GNNContribution


class GraphNodeOut(BaseModel):
    ip: str
    score: float
    in_degree: int
    out_degree: int


class GraphEdgeOut(BaseModel):
    src: str
    dst: str


class NetworkGraphOut(BaseModel):
    window_seconds: float
    flow_count: int
    node_count: int
    edge_count: int
    graph_score: float
    source: str
    model_loaded: bool
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    topology_hints: dict[str, list[dict[str, Any]]]


class NetworkThreatOut(BaseModel):
    pattern: str               # "ddos" | "scan" | "c2"
    ip: str
    severity: float
    rationale: str
    details: dict[str, Any]


class NetworkThreatsOut(BaseModel):
    window_seconds: float
    flow_count: int
    graph_score: float
    source: str
    threats: list[NetworkThreatOut]


class HealthOut(BaseModel):
    status: str
    models_loaded: bool
    models: list[str]
    n_features: int | None
    label_classes: list[str] | None
    uptime_seconds: float | None


# --- Helpers ---------------------------------------------------------------


def _run_ensemble(
    df: pd.DataFrame,
    contexts: list[FlowContextIn | None] | None = None,
    feature_dicts: list[dict[str, float]] | None = None,
    capture_on_block: bool = True,
) -> list[dict]:
    if state.ensemble is None or state.zero_trust is None:
        raise HTTPException(503, "Ensemble not loaded")
    results = state.ensemble.analyze(df.values)
    contexts = contexts or [None] * len(results)
    feature_dicts = feature_dicts or [
        {col: float(df.iloc[i][col]) for col in df.columns}
        for i in range(len(results))
    ]

    for r, ctx_in, feats in zip(results, contexts, feature_dicts):
        r["decision"] = score_to_decision(r["risk_score"])
        r["score"] = r["risk_score"]  # frontend alias
        r["ensemble_score"] = r["risk_score"]  # spec alias
        # rf/xgb/if shorthand alongside the original model_scores keys.
        ms = r["model_scores"]
        r["ensemble"] = {
            "rf_score": float(ms.get("random_forest", 0.0)),
            "xgb_score": float(ms.get("xgboost", 0.0)),
            "if_score": float(ms.get("isolation_forest", 0.0)),
        }
        r["thresholds"] = {
            "inspect": THRESHOLD_INSPECT,
            "block": THRESHOLD_BLOCK,
        }
        r["gnn_analyzed"] = False
        r["gnn"] = None
        r["honeypot_captured"] = False
        r["explanation"] = None

        if ctx_in:
            ctx_payload = ctx_in.model_dump()
            ctx_payload.pop("attack_type", None)
            fc = FlowContext(**ctx_payload)
            ctx_dict = ctx_in.model_dump()
        else:
            fc = FlowContext()
            ctx_dict = {}
        zt = state.zero_trust.evaluate(
            risk_score=r["risk_score"],
            model_scores=r["model_scores"],
            context=fc,
        )
        r["zero_trust"] = {
            "trust_level": zt.trust_level.value,
            "primary_action": zt.primary_action.value,
            "actions": [zt.primary_action.value, *(a.value for a in zt.secondary_actions)],
            "recommendations": zt.recommendations,
            "principles_applied": zt.principles_applied,
            "effective_risk": zt.effective_risk,
        }

        # Feed the flow into the rolling GNN window even on ALLOW —
        # the graph needs every flow to build accurate topology.
        if state.gnn is not None:
            state.gnn.add_flow(feats, ctx_dict)

        if capture_on_block and r["decision"] == "BLOCK":
            _capture_to_honeypot(r, ctx_in, feats)
            r["honeypot_captured"] = True
    return results


# --- GNN override for INSPECT decisions ------------------------------------


def _apply_gnn_override(
    result: dict,
    features: dict[str, float],
    ctx_in: FlowContextIn | None,
) -> dict:
    """Layer the GNN's topology view on top of an ensemble INSPECT.

    Behaviour
    ---------
    - For ALLOW / BLOCK: returns the result unchanged (no GNN call).
    - For INSPECT: runs the GNN on the rolling 60-s window. If the
      endpoint risk crosses the BLOCK / ALLOW thresholds we override
      the decision and record an ``override_reason`` so the UI can
      explain why.

    The honeypot is hit *after* the override so an INSPECT->BLOCK
    upgrade still captures the session.
    """
    if state.gnn is None:
        return result
    if result["decision"] != "INSPECT":
        return result

    ctx_dict = ctx_in.model_dump() if ctx_in else {}
    contribution = state.gnn.score_flow(
        features=features,
        context=ctx_dict,
        ensemble_score=float(result["risk_score"]),
    )
    analysis = state.gnn.analyze_window()

    flow_threat = float(contribution["gnn_endpoint_risk"])
    network_threat = float(contribution["gnn_graph_score"])

    # Compose patterns_detected from the topology hints that mention
    # this flow's src/dst. Confidence == severity from the hint.
    patterns: list[dict] = []
    src = ctx_dict.get("src_ip") or features.get("src_ip")
    dst = ctx_dict.get("dst_ip") or features.get("dst_ip")
    pattern_type_map = {
        "ddos": "POTENTIAL_DDOS",
        "scan": "POTENTIAL_SCAN",
        "c2":   "POTENTIAL_C2",
    }
    for category, label in pattern_type_map.items():
        for h in analysis.topology_hints.get(category, []):
            if h.get("ip") in (src, dst):
                patterns.append({
                    "type": label,
                    "source": h.get("ip"),
                    "confidence": float(h.get("severity", 0.5)),
                })

    # Override rule from the spec.
    new_decision = result["decision"]
    reason: str | None = None
    if flow_threat > THRESHOLD_BLOCK:
        new_decision = "BLOCK"
        if patterns:
            reason = (
                f"GNN detected {patterns[0]['type'].replace('POTENTIAL_', '').lower()} "
                f"pattern on {patterns[0]['source']}"
            )
        else:
            reason = (
                f"GNN endpoint risk {flow_threat:.2f} exceeded BLOCK threshold "
                f"{THRESHOLD_BLOCK}"
            )
    elif flow_threat < THRESHOLD_INSPECT:
        new_decision = "ALLOW"
        reason = (
            f"GNN endpoint risk {flow_threat:.2f} below INSPECT threshold "
            f"({THRESHOLD_INSPECT}); flow has no suspicious topology"
        )

    result["gnn_analyzed"] = True
    result["gnn"] = {
        "flow_threat_score": flow_threat,
        "network_threat_score": network_threat,
        "total_nodes": int(analysis.node_count),
        "total_edges": int(analysis.edge_count),
        "patterns_detected": patterns,
        "recommendation": new_decision,
        "override_reason": reason,
    }

    if new_decision != result["decision"]:
        result["decision"] = new_decision
        # Risk score is left as the ensemble's — the override is a
        # *policy* decision, not a rescoring. The UI shows both.
        if new_decision == "BLOCK" and not result["honeypot_captured"]:
            _capture_to_honeypot(result, ctx_in, features)
            result["honeypot_captured"] = True
    return result


def _capture_to_honeypot(
    result: dict,
    ctx_in: FlowContextIn | None,
    features: dict[str, float] | None = None,
) -> None:
    """Forward a BLOCK decision into the honeypot store. Best-effort: any
    failure is swallowed so the predict path never fails because of capture."""
    try:
        ctx = ctx_in.model_dump() if ctx_in else {}
        honeypot_manager.capture(
            src_ip=ctx.get("src_ip"),
            dst_port=ctx.get("dst_port"),
            decision=result["decision"],
            risk_score=float(result["risk_score"]),
            attack_type=ctx.get("attack_type"),
            model_scores={k: float(v) for k, v in result["model_scores"].items()},
            zero_trust=result["zero_trust"],
            flow_features={k: float(v) for k, v in (features or {}).items()},
        )
    except Exception:
        pass


# --- Routes ----------------------------------------------------------------


@app.get("/health", response_model=HealthOut)
def health():
    loaded = state.ensemble is not None and state.ensemble.is_loaded
    return HealthOut(
        status="ok" if loaded else "loading",
        models_loaded=loaded,
        models=list(state.ensemble.models.keys()) if loaded else [],
        n_features=len(state.feature_columns) if state.feature_columns else None,
        label_classes=state.label_classes,
        uptime_seconds=(time.time() - state.loaded_at) if state.loaded_at else None,
    )


@app.post("/predict", response_model=PredictionOut)
def predict(flow: FlowIn):
    """Single-flow decision with GNN override for INSPECT band.

    Flow:
        1. Ensemble scores the flow -> initial decision.
        2. If INSPECT, the GNN's topology view can override to
           ALLOW or BLOCK based on the rolling 60-s graph.
        3. BLOCK decisions (including overrides) hit the honeypot.
    """
    df = _flows_to_frame([flow.features])
    result = _run_ensemble(df, contexts=[flow.context])[0]
    _apply_gnn_override(result, flow.features, flow.context)
    return PredictionOut(**result)


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn):
    feature_list: list[dict[str, float]] = []
    contexts: list[FlowContextIn | None] = []
    for item in batch.flows:
        if isinstance(item, BatchFlow):
            feature_list.append(item.features)
            contexts.append(item.context)
        else:
            feature_list.append(item)
            contexts.append(None)

    df = _flows_to_frame(feature_list)
    t0 = time.perf_counter()
    results = _run_ensemble(df, contexts=contexts)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return BatchOut(
        predictions=[PredictionOut(**r) for r in results],
        count=len(results),
        inference_ms=round(dt_ms, 3),
    )


@app.post("/predict/explain", response_model=ExplainOut)
def predict_explain(flow: FlowIn):
    from app.explainability.shap_explainer import get_explainer

    df = _flows_to_frame([flow.features])
    result = _run_ensemble(df, contexts=[flow.context])[0]

    explainer = get_explainer(state.ensemble, state.feature_columns)
    explanation = explainer.explain(df, result["decision"], top_n=5)

    # The parent PredictionOut has an optional ``explanation: dict|None``
    # placeholder; pop it so ExplainOut's ``explanation: str`` isn't
    # passed twice via **result.
    result.pop("explanation", None)

    return ExplainOut(
        **result,
        top_features=[FeatureContribution(**f) for f in explanation["top_features"]],
        explanation=explanation["text"],
    )


# --- GNN endpoints ---------------------------------------------------------


@app.post("/predict/gnn", response_model=GNNPredictionOut)
def predict_gnn(flow: FlowIn):
    """Predict with graph context layered on top of the ensemble.

    The GNN's score is only blended into the final decision when the
    ensemble returns INSPECT — the band where additional context can
    change the call. For ALLOW / BLOCK the ensemble decision stands
    unchanged, but the GNN's view is still returned so the dashboard
    can show the topology context.
    """
    if state.gnn is None:
        raise HTTPException(503, "GNN not initialised")

    df = _flows_to_frame([flow.features])
    result = _run_ensemble(df, contexts=[flow.context])[0]

    ctx_dict = flow.context.model_dump() if flow.context else {}
    contribution = state.gnn.score_flow(
        features=flow.features,
        context=ctx_dict,
        ensemble_score=float(result["risk_score"]),
    )

    applied = result["decision"] == "INSPECT"
    if applied:
        new_score = contribution["combined_score"]
        result["risk_score"] = new_score
        result["score"] = new_score
        result["decision"] = score_to_decision(new_score)
        # Honeypot capture if the GNN escalated us to BLOCK.
        if result["decision"] == "BLOCK":
            _capture_to_honeypot(result, flow.context, flow.features)

    gnn_payload = {**contribution, "applied": applied}
    # Drop the parent's optional ``gnn`` placeholder so we can pass
    # the GNNContribution explicitly without a duplicate kwarg.
    result.pop("gnn", None)
    return GNNPredictionOut(**result, gnn=GNNContribution(**gnn_payload))


@app.get("/network/graph", response_model=NetworkGraphOut)
def network_graph(max_nodes: int = 150, max_edges: int = 400):
    """Current 60-s flow graph + per-node risk + topology hints."""
    if state.gnn is None:
        raise HTTPException(503, "GNN not initialised")
    analysis = state.gnn.analyze_window()
    snap = state.gnn.builder.build_snapshot()

    # Cap the returned graph so the dashboard stays responsive on
    # large windows — show the highest-risk nodes first.
    ranked_ips = sorted(
        snap.node_ids,
        key=lambda ip: analysis.node_scores.get(ip, 0.0),
        reverse=True,
    )[: max(1, min(max_nodes, 500))]
    keep_set = set(ranked_ips)
    keep_idx = {snap.node_index[ip] for ip in ranked_ips}

    in_deg = [0] * len(snap.node_ids)
    out_deg = [0] * len(snap.node_ids)
    edges: list[GraphEdgeOut] = []
    for i in range(snap.edge_index.shape[1]):
        u = int(snap.edge_index[0, i])
        v = int(snap.edge_index[1, i])
        out_deg[u] += 1
        in_deg[v] += 1
        if u in keep_idx and v in keep_idx and len(edges) < max_edges:
            edges.append(GraphEdgeOut(
                src=snap.node_ids[u],
                dst=snap.node_ids[v],
            ))

    nodes = [
        GraphNodeOut(
            ip=ip,
            score=float(analysis.node_scores.get(ip, 0.0)),
            in_degree=in_deg[snap.node_index[ip]],
            out_degree=out_deg[snap.node_index[ip]],
        )
        for ip in ranked_ips
    ]

    return NetworkGraphOut(
        window_seconds=analysis.window_seconds,
        flow_count=analysis.flow_count,
        node_count=analysis.node_count,
        edge_count=analysis.edge_count,
        graph_score=analysis.graph_score,
        source=analysis.source,
        model_loaded=analysis.model_loaded,
        nodes=nodes,
        edges=edges,
        topology_hints=analysis.topology_hints,
    )


@app.get("/network/threats", response_model=NetworkThreatsOut)
def network_threats():
    """Topology-derived threats (DDoS victims, scanners, C2 beacons)."""
    if state.gnn is None:
        raise HTTPException(503, "GNN not initialised")
    analysis = state.gnn.analyze_window()

    threats: list[NetworkThreatOut] = []
    for pattern in ("ddos", "scan", "c2"):
        for item in analysis.topology_hints.get(pattern, []):
            ip = item.get("ip")
            threats.append(NetworkThreatOut(
                pattern=pattern,
                ip=str(ip),
                severity=float(item.get("severity", 0.5)),
                rationale=str(item.get("rationale", "")),
                details={k: v for k, v in item.items()
                         if k not in ("ip", "severity", "rationale")},
            ))
    # Highest severity first so the dashboard's top item is the worst.
    threats.sort(key=lambda t: t.severity, reverse=True)

    return NetworkThreatsOut(
        window_seconds=analysis.window_seconds,
        flow_count=analysis.flow_count,
        graph_score=analysis.graph_score,
        source=analysis.source,
        threats=threats,
    )


# --- Demo simulator endpoints ---------------------------------------------


def _sim_predict(features: dict, context: dict | None) -> dict:
    """Internal predict used by the simulator — reuses _run_ensemble."""
    df = _flows_to_frame([features])
    ctx_in = FlowContextIn(**context) if context else None
    return _run_ensemble(df, contexts=[ctx_in])[0]


class DemoStartIn(BaseModel):
    scenario: str = Field(..., description="normal_traffic | ddos_attack | port_scan | brute_force | mixed_realistic")
    speed: float = Field(10.0, ge=1.0, le=100.0, description="flows per second")


@app.post("/demo/start")
def demo_start(body: DemoStartIn):
    if state.simulator is None:
        raise HTTPException(503, "simulator not initialised")
    try:
        state.simulator.start(body.scenario, body.speed)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return state.simulator.status()


@app.post("/demo/stop")
def demo_stop():
    if state.simulator is None:
        raise HTTPException(503, "simulator not initialised")
    state.simulator.stop()
    return state.simulator.status()


@app.get("/demo/status")
def demo_status():
    if state.simulator is None:
        return {"available": False, "running": False}
    return state.simulator.status()


@app.get("/demo/recent")
def demo_recent(limit: int = 100):
    if state.simulator is None:
        return {"flows": []}
    limit = max(1, min(limit, 500))
    return {"flows": state.simulator.store.recent(limit=limit)}


@app.get("/demo/stats")
def demo_stats():
    if state.simulator is None:
        return {"counts": {}, "flows_per_sec": 0.0, "timeline": [],
                "attack_types": {}, "top_blocked_ips": [], "heatmap": []}
    return state.simulator.store.stats()


# --- Honeypot endpoints ----------------------------------------------------


@app.get("/honeypot/status")
def honeypot_status():
    return honeypot_manager.stats()


@app.get("/honeypot/captures")
def honeypot_captures(limit: int = 200):
    limit = max(1, min(limit, 1000))
    evts = honeypot_manager.events()[-limit:][::-1]
    return {
        "count": len(evts),
        "captures": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "src_ip": e.src_ip,
                "dst_ip": e.dst_ip,
                "dst_port": e.dst_port,
                "attack_type": e.attack_type or "Unknown",
                "decision": e.decision,
                "risk_score": e.risk_score,
                "session_duration_seconds": e.session.get("session_duration_seconds"),
                "commands_count": len(e.session.get("commands", [])),
                "verified": e.verified,
                "is_real_attack": e.is_real_attack,
                "in_training_queue": e.in_training_queue,
            }
            for e in evts
        ],
    }


@app.get("/honeypot/capture/{event_id}")
def honeypot_capture_detail(event_id: str):
    event = honeypot_manager.get(event_id)
    if event is None:
        raise HTTPException(404, f"capture {event_id} not found")
    return {
        "id": event.id,
        "timestamp": event.timestamp,
        "src_ip": event.src_ip,
        "dst_ip": event.dst_ip,
        "dst_port": event.dst_port,
        "protocol": event.protocol,
        "decision": event.decision,
        "risk_score": event.risk_score,
        "attack_type": event.attack_type or "Unknown",
        "model_scores": event.model_scores,
        "zero_trust": event.zero_trust,
        "session": event.session,
        "verified": event.verified,
        "is_real_attack": event.is_real_attack,
        "in_training_queue": event.in_training_queue,
        "flow_features": event.flow_features,
    }


class VerifyIn(BaseModel):
    """Verification body. ``is_real_attack`` defaults to ``True`` so
    the old POST /honeypot/verify/{id} (no body) keeps working."""
    is_real_attack: bool = True


@app.post("/honeypot/verify/{event_id}")
def honeypot_verify(event_id: str, body: VerifyIn | None = None):
    """Mark a capture as verified. ``is_real_attack=true`` enqueues
    it for the next retrain; ``false`` marks it as a false positive."""
    is_real = True if body is None else bool(body.is_real_attack)
    event = honeypot_manager.verify_capture(event_id, is_real_attack=is_real)
    if event is None:
        raise HTTPException(404, f"capture {event_id} not found")
    metrics_store.increment_verified(is_real_attack=is_real)
    queue_size = len(honeypot_manager.get_training_queue())
    metrics_store.set_training_queue_size(queue_size)
    return {
        "id": event.id,
        "verified": event.verified,
        "is_real_attack": event.is_real_attack,
        "in_training_queue": event.in_training_queue,
        "feedback": honeypot_manager.feedback_counts(),
        "training_queue_size": queue_size,
    }


@app.get("/honeypot/training-queue")
def honeypot_training_queue(limit: int = 1000):
    """Captures the analyst confirmed as real attacks, awaiting retrain."""
    queue = honeypot_manager.get_training_queue()[-limit:][::-1]
    return {
        "count": len(honeypot_manager.get_training_queue()),
        "captures": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "src_ip": e.src_ip,
                "dst_port": e.dst_port,
                "attack_type": e.attack_type or "Unknown",
                "risk_score": e.risk_score,
            }
            for e in queue
        ],
    }


# --- Model performance + retraining ----------------------------------------


class RetrainOut(BaseModel):
    previous_version: str
    current_version: str
    previous_accuracy: float
    current_accuracy: float
    samples_added: int
    improvement: float
    seconds: float
    note: str | None = None


@app.get("/model/metrics")
def model_metrics():
    snap = metrics_store.snapshot()
    # Always reconcile training queue against the source of truth on
    # the honeypot, so a restart that lost the metrics counter still
    # surfaces the right number.
    queue_size = len(honeypot_manager.get_training_queue())
    snap["training_queue_size"] = queue_size
    snap["verified_captures"] = honeypot_manager.get_verified_count()
    return snap


@app.get("/model/cross_dataset")
def model_cross_dataset():
    """Three-stage honest cross-dataset comparison.

    Returns the in-distribution (CICIDS-2017), zero-adaptation
    cross-dataset (CICIDS-2018), and post-honeypot adapted numbers
    in one payload, sourced from the JSONs produced by
    ``training/evaluate_combined_system.py`` and
    ``training/demo_feedback_loop.py``.
    """
    import json as _json
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    combined_path = os.path.join(
        project_root, "training", "combined_system_results.json"
    )
    feedback_path = os.path.join(
        project_root, "training", "feedback_loop_results.json"
    )

    if not os.path.exists(combined_path):
        raise HTTPException(
            503,
            f"{combined_path} not found — run "
            f"`python training/evaluate_combined_system.py` first.",
        )
    payload = _json.load(open(combined_path))
    honest = payload.get("honest_three_stage", {})

    # Backfill the 'adapted_2018' block from the feedback-loop JSON
    # if the combined eval ran before the feedback loop did.
    if not honest.get("adapted_2018") and os.path.exists(feedback_path):
        fb = _json.load(open(feedback_path))
        before = fb.get("before", {})
        after = fb.get("after", {})
        if after:
            honest["adapted_2018"] = {
                "ensemble_accuracy": after.get("acc"),
                "ensemble_f1": after.get("f1"),
                "ensemble_fpr": after.get("fpr"),
                "samples_added": fb.get("samples_added_to_train"),
                "improvement_pp": (
                    (after.get("acc", 0) - before.get("acc", 0)) * 100
                ),
            }
    return {
        "generated_at": payload.get("generated_at"),
        "thresholds": payload.get("thresholds"),
        "training_2017": honest.get("training_2017"),
        "unseen_2018": honest.get("unseen_2018"),
        "adapted_2018": honest.get("adapted_2018"),
    }



@app.get("/model/iterations")
def model_iterations():
    """Multi-iteration honeypot feedback-loop curve.

    Returns the baseline + per-iteration metrics produced by
    ``training/demo_feedback_loop.py``. Lets the dashboard plot the
    learning curve over multiple rounds of analyst feedback.
    """
    import json as _json
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(
        project_root, "training", "multi_iteration_results.json"
    )
    if not os.path.exists(path):
        raise HTTPException(
            503,
            f"{path} not found — run "
            f"`python training/demo_feedback_loop.py` first.",
        )
    payload = _json.load(open(path))
    return payload


@app.post("/model/retrain", response_model=RetrainOut)
def model_retrain():
    """Retrain the Random Forest on the original training set + the
    verified honeypot captures.

    Demo-grade: we subsample the original CICIDS-2017 training set to
    keep wall-clock under ~20 s, then mix in the verified captures
    using the same feature schema. On finish we evaluate against the
    CICIDS-2017 test split, persist the new model, and reset the
    training queue.
    """
    if state.ensemble is None or state.feature_columns is None:
        raise HTTPException(503, "Ensemble not loaded")

    queue = honeypot_manager.get_training_queue()
    if len(queue) < 1:
        raise HTTPException(
            400,
            "Training queue is empty — verify some captures first.",
        )

    t0 = time.perf_counter()
    try:
        new_acc, samples_added = _retrain_rf_on_queue(queue)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Training data unavailable: {e}")

    consumed = honeypot_manager.clear_training_queue()
    record = metrics_store.record_retrain(
        new_accuracy=new_acc,
        samples_added=samples_added,
        note=f"Retrained RF on {samples_added} verified captures",
    )
    metrics_store.set_training_queue_size(0)
    seconds = time.perf_counter() - t0
    snap = metrics_store.snapshot()
    return RetrainOut(
        previous_version=record["previous_version"],
        current_version=record["current_version"],
        previous_accuracy=snap["baseline_accuracy"],
        current_accuracy=record["current_accuracy"],
        samples_added=samples_added,
        improvement=record["improvement"],
        seconds=round(seconds, 2),
        note=f"Consumed {len(consumed)} captures",
    )


def _retrain_rf_on_queue(queue: list) -> tuple[float, int]:
    """Sample-efficient RF retrain: original-train subsample + queue.

    Returns (test_accuracy_pct, n_queue_samples_used).

    Keeps things demo-fast (a few seconds) by capping the training
    sample size — see ``MAX_TRAIN_SAMPLES``. The retrained classifier
    replaces the live RF in the ensemble in-place; the joblib file on
    disk is also updated so reloads pick it up.
    """
    import os as _os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    MAX_TRAIN_SAMPLES = 80_000   # ~10s training on CPU

    # balanced_splits matches the 51-feature schema used by the
    # scaler + RF + 2018 alignment path. train_test_splits is the
    # legacy 50-feature top_features subset and won't concat.
    splits_path = _os.path.expanduser(
        "~/sem6el/data/processed/balanced_splits.joblib"
    )
    if not _os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)
    splits = joblib.load(splits_path)
    X_train = splits.get("X_train_scaled", splits["X_train"])
    y_train = splits["y_train_binary"]
    X_test = splits.get("X_test_scaled", splits["X_test"])
    y_test = splits["y_test_binary"]

    rng = np.random.default_rng(42)
    if len(X_train) > MAX_TRAIN_SAMPLES:
        idx = rng.choice(len(X_train), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Build feature vectors for the queued captures. Each capture
    # stores its full feature dict in ``flow_features`` when available;
    # otherwise we synthesise a neutral row from the model scores.
    queue_feats: list[list[float]] = []
    queue_labels: list[int] = []
    cols = list(state.feature_columns)
    for e in queue:
        feats = dict(e.flow_features or {})
        row = [float(feats.get(c, 0.0)) for c in cols]
        queue_feats.append(row)
        # Use the analyst's verdict for the label: real attack → 1,
        # false positive → 0. None (un-set) defaults to 0 since the
        # queue should never hold un-verified captures, but be safe.
        queue_labels.append(1 if e.is_real_attack is True else 0)
    if not queue_feats:
        raise HTTPException(400, "Queued captures have no feature vectors.")
    X_queue = np.array(queue_feats, dtype=float)
    y_queue = np.array(queue_labels, dtype=int)

    # Apply the saved scaler before training so the queue mixes in the
    # same numeric space as the cached X_train_scaled splits.
    scaler = state.ensemble.scaler
    X_queue_scaled = scaler.transform(X_queue)
    # X_train above is already pre-scaled in the splits file
    # (see training/data_preprocessing.py); keep that contract.

    X_full = np.vstack([X_train, X_queue_scaled])
    y_full = np.concatenate([y_train, y_queue])

    rf = RandomForestClassifier(
        n_estimators=80, max_depth=18, n_jobs=-1,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_full, y_full)

    # Evaluate against the held-out test split.
    pred = rf.predict(X_test)
    acc = float(accuracy_score(y_test, pred)) * 100.0

    # Swap into the live ensemble and persist to disk so the next
    # service boot reloads the improved model.
    state.ensemble.models["random_forest"] = rf
    out_path = _os.path.join(MODELS_PATH, "random_forest.joblib")
    try:
        joblib.dump(rf, out_path)
    except OSError:
        pass  # keep the in-memory update even if disk is read-only

    return acc, len(queue_feats)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
