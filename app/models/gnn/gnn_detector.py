"""
High-level GNN inference wrapper.

:class:`GNNDetector` is the entry point used by the API. It owns the
rolling :class:`FlowGraphBuilder`, optionally loads a trained GAT
model, and exposes three operations:

- :meth:`add_flow`            — ingest a flow into the rolling window
                                 (called from every ``/predict*`` path)
- :meth:`analyze_window`      — score the current 60-s window and
                                 return per-IP probabilities + a
                                 graph-level probability + topology
                                 hints. Backs ``GET /network/graph``
                                 and ``GET /network/threats``.
- :meth:`score_flow`          — return an enhanced risk for a single
                                 flow by reading the GNN's score for
                                 its endpoints. Called by the API only
                                 when the ensemble returns INSPECT.

The model file is optional: if the weights are not present the
detector still emits topology-hint-based scores so the endpoints
never 500 in development.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.gnn.graph_builder import (
    EDGE_FEATURE_DIM,
    FlowGraphBuilder,
    GraphSnapshot,
    NODE_FEATURE_DIM,
)


DEFAULT_MODEL_PATH = os.environ.get(
    "NGFW_GNN_MODEL_PATH",
    os.path.expanduser("~/sem6el/trained_models/gnn_gat.pt"),
)


@dataclass
class GNNAnalysis:
    graph_score: float                          # whole-window risk in [0, 1]
    node_scores: dict[str, float]               # ip -> risk in [0, 1]
    topology_hints: dict[str, list[dict]]       # ddos / scan / c2
    flow_count: int
    node_count: int
    edge_count: int
    window_seconds: float
    model_loaded: bool
    source: str                                 # "gnn" | "heuristic"


class GNNDetector:
    def __init__(
        self,
        model_path: str | None = None,
        window_seconds: float = 60.0,
        ensemble_weight: float = 0.6,
        gnn_weight: float = 0.4,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.builder = FlowGraphBuilder(window_seconds=window_seconds)
        self.model: Any = None
        self._model_lock = threading.Lock()
        self.model_loaded = False
        # How much weight the GNN gets in the INSPECT-band combined
        # score. The ensemble retains the majority weight because it
        # has per-flow features the GNN doesn't see.
        self.ensemble_weight = ensemble_weight
        self.gnn_weight = gnn_weight
        # Cache the most recent analysis so /network/graph and
        # /network/threats can share work; expires after 2 s.
        self._cache: GNNAnalysis | None = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 2.0

    # --- model lifecycle ---------------------------------------------

    def try_load(self) -> bool:
        """Load the trained GAT weights if available. Safe to call
        repeatedly — only the first successful call mutates state."""
        if self.model_loaded:
            return True
        with self._model_lock:
            if self.model_loaded:
                return True
            if not os.path.exists(self.model_path):
                return False
            try:
                from app.models.gnn.gnn_model import load_model
                self.model = load_model(self.model_path)
                self.model_loaded = True
                return True
            except Exception as e:  # noqa: BLE001 — never crash on optional model
                print(f"[GNN] could not load model at {self.model_path}: {e}")
                return False

    # --- ingestion ---------------------------------------------------

    def add_flow(
        self,
        features: dict[str, float],
        context: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> None:
        """Push a flow into the rolling window.

        Both ``features`` and ``context`` may carry the IPs (the API
        path strips them from features but the simulator keeps them in
        context). We accept either to keep the call sites simple.
        """
        src = None
        dst = None
        if context:
            src = context.get("src_ip")
            dst = context.get("dst_ip")
        src = src or features.get("src_ip")
        dst = dst or features.get("dst_ip")
        # Mirror the simulator: synthesise a destination IP when only
        # a port is known so the graph stays connected. Without this,
        # API single-flow calls produce empty graphs.
        if src and not dst:
            dst = "internal"
        if not (src and dst):
            return
        self.builder.add_flow(src, dst, features, ts=ts)
        self._invalidate_cache()

    # --- analysis ----------------------------------------------------

    def analyze_window(self, force: bool = False) -> GNNAnalysis:
        """Return scores + topology hints for the rolling window."""
        if not force and self._cache is not None:
            if time.time() - self._cache_ts < self._cache_ttl:
                return self._cache

        snap = self.builder.build_snapshot()
        analysis = self._analyze_snapshot(snap)
        self._cache = analysis
        self._cache_ts = time.time()
        return analysis

    def analyze_flows(self, flows: list[dict[str, Any]]) -> GNNAnalysis:
        """One-shot analysis of an explicit flow batch.

        Used by ``POST /predict/gnn`` when callers want to score a
        self-contained window rather than the rolling buffer.
        """
        snap = self.builder.build_from_flows(flows)
        return self._analyze_snapshot(snap)

    def _analyze_snapshot(self, snap: GraphSnapshot) -> GNNAnalysis:
        N = len(snap.node_ids)
        if N == 0:
            return GNNAnalysis(
                graph_score=0.0,
                node_scores={},
                topology_hints=snap.topology_hints,
                flow_count=0,
                node_count=0,
                edge_count=0,
                window_seconds=snap.window_seconds,
                model_loaded=self.model_loaded,
                source="heuristic",
            )

        node_scores: dict[str, float]
        graph_score: float
        source: str

        if self.try_load():
            try:
                node_p, graph_p = self._run_model(snap)
                node_scores = {
                    ip: float(node_p[i]) for i, ip in enumerate(snap.node_ids)
                }
                graph_score = float(graph_p)
                source = "gnn"
            except Exception as e:  # noqa: BLE001
                print(f"[GNN] inference failed, falling back to heuristics: {e}")
                node_scores, graph_score = self._heuristic_scores(snap)
                source = "heuristic"
        else:
            node_scores, graph_score = self._heuristic_scores(snap)
            source = "heuristic"

        return GNNAnalysis(
            graph_score=graph_score,
            node_scores=node_scores,
            topology_hints=snap.topology_hints,
            flow_count=snap.flow_count,
            node_count=N,
            edge_count=int(snap.edge_index.shape[1]),
            window_seconds=snap.window_seconds,
            model_loaded=self.model_loaded,
            source=source,
        )

    def _run_model(self, snap: GraphSnapshot) -> tuple[np.ndarray, float]:
        data = FlowGraphBuilder.snapshot_to_pyg(snap)
        return self.model.predict(data)

    @staticmethod
    def _heuristic_scores(
        snap: GraphSnapshot,
    ) -> tuple[dict[str, float], float]:
        """Topology-hint-derived fallback scores.

        Used until the model is trained and on hosts without
        torch-geometric. Returns the same shape as the model so the
        rest of the pipeline doesn't branch.
        """
        node_scores: dict[str, float] = {ip: 0.0 for ip in snap.node_ids}
        for category in ("ddos", "scan", "c2"):
            for item in snap.topology_hints.get(category, []):
                ip = item.get("ip")
                sev = float(item.get("severity", 0.5))
                if ip in node_scores:
                    node_scores[ip] = max(node_scores[ip], sev)
        graph_score = max(node_scores.values()) if node_scores else 0.0
        return node_scores, graph_score

    # --- ensemble integration ----------------------------------------

    def score_flow(
        self,
        features: dict[str, float],
        context: dict[str, Any] | None,
        ensemble_score: float,
    ) -> dict[str, Any]:
        """Combine the GNN's view with an ensemble score for one flow.

        Only meaningful when the ensemble returned INSPECT — for ALLOW
        we trust the ensemble and skip the GNN, for BLOCK there's
        nothing to gain from a second opinion that could only
        downgrade. The API enforces that gate; this method is
        side-effect-free so callers may still invoke it on any flow.
        """
        # Make sure this flow is in the window before we score it,
        # otherwise the source IP won't appear as a node.
        self.add_flow(features, context)

        analysis = self.analyze_window()
        src = (context or {}).get("src_ip") or features.get("src_ip")
        dst = (context or {}).get("dst_ip") or features.get("dst_ip")

        node_risk_src = analysis.node_scores.get(src, 0.0) if src else 0.0
        node_risk_dst = analysis.node_scores.get(dst, 0.0) if dst else 0.0
        # The endpoint score is the worst of the two — a flow is risky
        # if either its source or destination is in a suspicious
        # topology.
        endpoint_risk = max(node_risk_src, node_risk_dst, analysis.graph_score * 0.5)

        combined = (
            self.ensemble_weight * ensemble_score
            + self.gnn_weight * endpoint_risk
        )
        combined = float(min(1.0, max(0.0, combined)))

        # Which topology bucket (if any) flagged the endpoint? This
        # gets surfaced in the decision rationale.
        pattern = None
        for category in ("ddos", "scan", "c2"):
            ips = {h["ip"] for h in analysis.topology_hints.get(category, [])}
            if src in ips or dst in ips:
                pattern = category
                break

        return {
            "combined_score": combined,
            "ensemble_score": float(ensemble_score),
            "gnn_endpoint_risk": float(endpoint_risk),
            "gnn_graph_score": float(analysis.graph_score),
            "gnn_node_score_src": float(node_risk_src),
            "gnn_node_score_dst": float(node_risk_dst),
            "detected_pattern": pattern,
            "source": analysis.source,
        }

    # --- cache helpers -----------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cache = None
