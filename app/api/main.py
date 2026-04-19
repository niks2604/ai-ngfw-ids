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

from app.models.ensemble import EnsembleDetector
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
    state.loaded_at = time.time()
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


class PredictionOut(BaseModel):
    decision: str
    risk_score: float
    score: float                  # alias of risk_score for frontend convenience
    model_scores: ModelScores
    thresholds: dict[str, float]
    zero_trust: ZeroTrustOut


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
) -> list[dict]:
    if state.ensemble is None or state.zero_trust is None:
        raise HTTPException(503, "Ensemble not loaded")
    results = state.ensemble.analyze(df.values)
    contexts = contexts or [None] * len(results)

    for r, ctx_in in zip(results, contexts):
        r["decision"] = score_to_decision(r["risk_score"])
        r["score"] = r["risk_score"]  # frontend alias
        r["thresholds"] = {
            "inspect": THRESHOLD_INSPECT,
            "block": THRESHOLD_BLOCK,
        }

        fc = FlowContext(**ctx_in.model_dump()) if ctx_in else FlowContext()
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
    return results


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
    df = _flows_to_frame([flow.features])
    result = _run_ensemble(df, contexts=[flow.context])[0]
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

    return ExplainOut(
        **result,
        top_features=[FeatureContribution(**f) for f in explanation["top_features"]],
        explanation=explanation["text"],
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
