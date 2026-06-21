"""
Diagnose the GNN's calibration on CICIDS-2018.

Why this exists
---------------
The combined-system evaluation showed the cascade overriding 99.5% of
INSPECT flows to ALLOW — i.e. the GNN's per-flow scores are almost
all below 0.5, so the cascade's "GNN > 0.5 → BLOCK" rule never fires.

This script reproduces that scoring path on a small subsample of
CICIDS-2018 and answers four questions, in order:

1. **Score distribution** — min / max / mean / median / std + a
   10-bucket histogram of per-flow GNN scores.
2. **Threshold sweep** — for each candidate threshold in
   [0.1, 0.5], compute accuracy / precision / recall / F1 / FPR.
3. **Graph health** — for a handful of sample windows, print node
   count, edge count, average degree. Small graphs degrade attention
   so we want to make sure the windowing is actually building
   something the GAT can use.
4. **Cascade with the best threshold** — re-run Strategy B from the
   combined eval using the threshold from step 2 in place of 0.5,
   and report whether that fixes the override imbalance.

Outputs are written to ``training/gnn_threshold_debug.json``.
"""

from __future__ import annotations

# Same OpenMP guards as evaluate_combined_system.py — without these,
# the joint xgboost+torch initialisation deadlocks the per-window
# scoring loop on macOS.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.ensemble import EnsembleDetector           # noqa: E402
from app.models.gnn.graph_builder import FlowGraphBuilder   # noqa: E402
from app.models.gnn.gnn_model import load_model             # noqa: E402
from training.evaluate_cicids2018 import (                  # noqa: E402
    MODELS_PATH,
    _load_2018,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

GNN_MODEL_PATH = os.path.join(MODELS_PATH, "gnn_gat.pt")
RESULTS_PATH = _PROJECT_ROOT / "training" / "gnn_threshold_debug.json"

# 20k is plenty for distribution shape and stable threshold tuning.
N_FLOWS = 20_000
WINDOW_FLOWS = 200
T_INSPECT = 0.3
T_BLOCK = 0.7
SEED = 42

WIDTH = 64


def banner(s):
    print("\n" + "═" * WIDTH)
    print(s.center(WIDTH))
    print("═" * WIDTH)


def section(s):
    print("\n" + s)
    print("─" * WIDTH)


# ---------------------------------------------------------------------
# IP synthesis (same as evaluate_combined_system)
# ---------------------------------------------------------------------


def _synth_ips(row_idx, is_attack, rng):
    if is_attack:
        s = rng.randrange(50); d = rng.randrange(3)
        return (f"185.220.{s // 250 + 1}.{s % 250 + 1}",
                f"10.0.{d // 250 + 1}.{d % 250 + 1}")
    s = rng.randrange(200); d = rng.randrange(200)
    return (f"203.0.{s // 250 + 1}.{s % 250 + 1}",
            f"10.1.{d // 250 + 1}.{d % 250 + 1}")


# ---------------------------------------------------------------------
# Score flows + capture per-window diagnostics
# ---------------------------------------------------------------------


def score_with_diagnostics(X, y, feature_cols, gnn_model):
    import torch as _torch
    _torch.set_num_threads(1)

    rng = random.Random(SEED)
    gnn_scores = np.zeros(len(X), dtype=np.float32)
    # Capture per-window stats so we can verify graph health.
    window_stats: list[dict] = []
    # Per-window graph-level score (the head we did NOT use for
    # per-flow assignment) — useful to compare against per-node.
    graph_scores: list[float] = []
    cols = list(feature_cols)
    n = len(X)
    t1 = time.perf_counter()
    for start in range(0, n, WINDOW_FLOWS):
        end = min(start + WINDOW_FLOWS, n)
        builder = FlowGraphBuilder(window_seconds=1e9)
        row_src: list[str] = []
        row_dst: list[str] = []
        t_base = time.time()
        for j in range(start, end):
            is_attack = bool(y[j] == 1)
            src, dst = _synth_ips(j, is_attack, rng)
            row_src.append(src); row_dst.append(dst)
            feats = {c: float(X[j, k]) for k, c in enumerate(cols)}
            builder.add_flow(src, dst, feats, ts=t_base + (j - start) * 0.1)
        snap = builder.build_snapshot()
        if len(snap.node_ids) == 0:
            continue
        data = FlowGraphBuilder.snapshot_to_pyg(snap)
        node_p, graph_p = gnn_model.predict(data)
        graph_scores.append(float(graph_p))

        n_nodes = len(snap.node_ids)
        n_edges = int(snap.edge_index.shape[1])
        avg_deg = (n_edges * 2.0) / max(1, n_nodes)
        window_stats.append({
            "window_start": start,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_degree": avg_deg,
            "graph_score": float(graph_p),
            "node_max": float(np.max(node_p)) if len(node_p) else 0.0,
            "node_mean": float(np.mean(node_p)) if len(node_p) else 0.0,
        })

        graph_half = float(graph_p) * 0.5
        for offset, j in enumerate(range(start, end)):
            src_idx = snap.node_index.get(row_src[offset], -1)
            dst_idx = snap.node_index.get(row_dst[offset], -1)
            src_s = float(node_p[src_idx]) if src_idx >= 0 else 0.0
            dst_s = float(node_p[dst_idx]) if dst_idx >= 0 else 0.0
            gnn_scores[j] = max(src_s, dst_s, graph_half)
    print(f"  scored in {time.perf_counter() - t1:.1f}s "
          f"({len(window_stats)} windows)")
    return gnn_scores, window_stats, np.array(graph_scores, dtype=float)


# ---------------------------------------------------------------------
# 1) Score distribution + histogram
# ---------------------------------------------------------------------


def print_distribution(label: str, scores: np.ndarray) -> dict:
    section(f"{label} SCORE DISTRIBUTION")
    if len(scores) == 0:
        print("  (no scores)")
        return {}
    info = {
        "label": label,
        "n": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "fraction_below_0.5": float((scores < 0.5).mean()),
        "fraction_below_0.3": float((scores < 0.3).mean()),
    }
    print(f"  ├── Min:    {info['min']:.3f}")
    print(f"  ├── Max:    {info['max']:.3f}")
    print(f"  ├── Mean:   {info['mean']:.3f}")
    print(f"  ├── Median: {info['median']:.3f}")
    print(f"  ├── Std:    {info['std']:.3f}")
    print(f"  ├── Fraction < 0.5: "
          f"{info['fraction_below_0.5'] * 100:.1f}%")
    print(f"  └── Histogram:")
    buckets = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=buckets)
    n_total = sum(counts)
    bar_max = max(counts.max(), 1)
    hist: list[dict] = []
    for i, c in enumerate(counts):
        lo, hi = buckets[i], buckets[i + 1]
        bar = "█" * int(40 * c / bar_max)
        pct = c / n_total * 100 if n_total else 0
        print(f"      {lo:.1f}-{hi:.1f}: {bar:<40} {c:>6,} ({pct:.1f}%)")
        hist.append({"low": float(lo), "high": float(hi),
                     "count": int(c), "pct": float(pct)})
    info["histogram"] = hist
    return info


# ---------------------------------------------------------------------
# 2) Threshold sweep
# ---------------------------------------------------------------------


def threshold_sweep(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, list[dict]]:
    section("THRESHOLD SWEEP (GNN-only decisions)")
    print(f"  Threshold │ Accuracy │ Precision │ Recall │ F1     │ FPR")
    print(f"  ──────────┼──────────┼───────────┼────────┼────────┼────────")
    sweep_rows: list[dict] = []
    best_f1, best_t = -1.0, 0.5
    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pred = (scores >= t).astype(int)
        acc = float(accuracy_score(y_true, pred))
        prec = float(precision_score(y_true, pred, zero_division=0))
        rec = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        marker = ""
        if f1 > best_f1:
            best_f1 = f1; best_t = t
        sweep_rows.append({"threshold": t, "accuracy": acc,
                           "precision": prec, "recall": rec,
                           "f1": f1, "fpr": fpr})
        print(f"  {t:<10.2f}│ {acc * 100:>7.1f}% │ {prec * 100:>8.1f}% │ "
              f"{rec * 100:>5.1f}% │ {f1:.3f}  │ {fpr * 100:>5.1f}%")
    for r in sweep_rows:
        if r["threshold"] == best_t:
            r["best_f1"] = True
    print(f"\n  Best F1 at threshold = {best_t:.2f} (F1 = {best_f1:.3f})")
    return best_t, sweep_rows


# ---------------------------------------------------------------------
# 3) Graph health
# ---------------------------------------------------------------------


def graph_health(window_stats: list[dict]) -> dict:
    section("GRAPH HEALTH ACROSS WINDOWS")
    if not window_stats:
        print("  (no windows)")
        return {}
    nodes = np.array([w["n_nodes"] for w in window_stats])
    edges = np.array([w["n_edges"] for w in window_stats])
    deg = np.array([w["avg_degree"] for w in window_stats])
    print(f"  Windows:      {len(window_stats)}")
    print(f"  Nodes/window: min={nodes.min():>4}  "
          f"median={int(np.median(nodes)):>4}  max={nodes.max():>4}  "
          f"mean={nodes.mean():.1f}")
    print(f"  Edges/window: min={edges.min():>4}  "
          f"median={int(np.median(edges)):>4}  max={edges.max():>4}  "
          f"mean={edges.mean():.1f}")
    print(f"  Avg degree:   min={deg.min():.2f}  "
          f"median={float(np.median(deg)):.2f}  max={deg.max():.2f}  "
          f"mean={deg.mean():.2f}")

    # The training pipeline used windows of 200 flows producing ~50-150
    # nodes; anything significantly different here means the eval is
    # off-distribution from training.
    too_sparse = int((deg < 1.5).sum())
    too_dense = int((deg > 6.0).sum())
    print(f"  Windows with avg_degree < 1.5: {too_sparse} (too sparse — "
          f"attention can't propagate)")
    print(f"  Windows with avg_degree > 6.0: {too_dense} (very dense — "
          f"may saturate the head)")

    print("\n  Sample (first 5 windows):")
    print(f"  {'start':<6}{'nodes':>7}{'edges':>7}{'avg_deg':>10}"
          f"{'graph_p':>10}{'node_max':>10}")
    for w in window_stats[:5]:
        print(f"  {w['window_start']:<6}{w['n_nodes']:>7}{w['n_edges']:>7}"
              f"{w['avg_degree']:>10.2f}{w['graph_score']:>10.3f}"
              f"{w['node_max']:>10.3f}")
    return {
        "n_windows": int(len(window_stats)),
        "nodes": {"min": int(nodes.min()), "max": int(nodes.max()),
                  "mean": float(nodes.mean()),
                  "median": float(np.median(nodes))},
        "edges": {"min": int(edges.min()), "max": int(edges.max()),
                  "mean": float(edges.mean()),
                  "median": float(np.median(edges))},
        "avg_degree": {"min": float(deg.min()), "max": float(deg.max()),
                       "mean": float(deg.mean()),
                       "median": float(np.median(deg))},
        "too_sparse_count": too_sparse,
        "too_dense_count": too_dense,
        "sample": window_stats[:5],
    }


# ---------------------------------------------------------------------
# 4) Cascade with calibrated threshold
# ---------------------------------------------------------------------


def cascade(ens, gnn, gnn_threshold) -> dict:
    pred = np.zeros_like(ens, dtype=int)
    allow_mask = ens < T_INSPECT
    block_mask = ens >= T_BLOCK
    inspect_mask = ~allow_mask & ~block_mask
    pred[block_mask] = 1
    pred[allow_mask] = 0
    pred[inspect_mask] = (gnn[inspect_mask] >= gnn_threshold).astype(int)
    gnn_to_block = int((inspect_mask & (gnn >= gnn_threshold)).sum())
    gnn_to_allow = int((inspect_mask & (gnn < gnn_threshold)).sum())
    return {
        "predictions": pred,
        "allow_count": int(allow_mask.sum()),
        "inspect_count": int(inspect_mask.sum()),
        "block_count": int(block_mask.sum()),
        "gnn_to_allow_in_inspect": gnn_to_allow,
        "gnn_to_block_in_inspect": gnn_to_block,
    }


def cascade_metrics(c: dict, y: np.ndarray) -> dict:
    pred = c["predictions"]
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {"accuracy": acc, "f1": f1, "fpr": fpr,
            "tp": int(((pred == 1) & (y == 1)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "tn": tn, "fp": fp,
            "allow_count": c["allow_count"],
            "inspect_count": c["inspect_count"],
            "block_count": c["block_count"],
            "gnn_to_allow_in_inspect": c["gnn_to_allow_in_inspect"],
            "gnn_to_block_in_inspect": c["gnn_to_block_in_inspect"]}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    banner("GNN THRESHOLD DEBUG (CICIDS-2018)")

    # Load
    print("Loading models + data…")
    feature_cols = joblib.load(
        os.path.join(MODELS_PATH, "feature_columns.joblib")
    )
    ensemble = EnsembleDetector(models_path=MODELS_PATH)
    ensemble.load_models()
    gnn = load_model(GNN_MODEL_PATH)

    X_full, y_full, _ = _load_2018(
        os.path.expanduser("~/sem6el/data/cicids2018"),
        list(feature_cols),
    )
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_full))[:N_FLOWS]
    X = X_full[perm].astype(float)
    y = y_full[perm].astype(int)
    print(f"  {len(y):,} flows "
          f"({int((y == 0).sum()):,} benign / {int((y == 1).sum()):,} attack)")

    # Score
    print("\nScoring with ensemble…")
    t0 = time.perf_counter()
    ens_scores = ensemble.predict_ensemble(X)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("\nScoring with GNN (per-flow node-level)…")
    gnn_scores, window_stats, window_graph_scores = score_with_diagnostics(
        X, y, feature_cols, gnn,
    )

    # 1) Distribution — both per-flow and per-window
    flow_dist = print_distribution("PER-FLOW GNN", gnn_scores)
    graph_dist = print_distribution("PER-WINDOW GRAPH HEAD", window_graph_scores)

    # 2) Threshold sweep on per-flow scores (the cascade rule uses these)
    best_t, sweep_rows = threshold_sweep(gnn_scores, y)

    # 3) Graph health
    graph_info = graph_health(window_stats)

    # 4) Cascade with calibrated threshold
    section("CASCADE RECOMPUTATION (BEST GNN THRESHOLD)")
    casc_default = cascade(ens_scores, gnn_scores, 0.5)
    casc_tuned = cascade(ens_scores, gnn_scores, best_t)
    m_def = cascade_metrics(casc_default, y)
    m_tuned = cascade_metrics(casc_tuned, y)

    rows = [
        ("Cascade @ 0.50 (default)", m_def),
        (f"Cascade @ {best_t:.2f} (tuned)", m_tuned),
    ]
    print(f"  {'Variant':<28}│ Accuracy │ F1     │ FPR    │ GNN→BLOCK")
    print(f"  ────────────────────────────┼──────────┼────────┼────────┼──────────")
    for name, m in rows:
        print(f"  {name:<28}│ {m['accuracy'] * 100:>7.1f}% │ {m['f1']:.3f}  │ "
              f"{m['fpr'] * 100:>5.1f}%  │ {m['gnn_to_block_in_inspect']:>6,} "
              f"({m['gnn_to_block_in_inspect'] / max(1, m['inspect_count']) * 100:.1f}%)")

    # --- Diagnosis -------------------------------------------------
    section("DIAGNOSIS + SUGGESTED FIX")
    frac_below_default = flow_dist.get("fraction_below_0.5", 1.0)
    suggestions: list[str] = []
    if frac_below_default >= 0.9:
        suggestions.append(
            f"Per-flow GNN scores collapse below 0.5 "
            f"({frac_below_default * 100:.1f}% under threshold). "
            f"Lower the cascade's GNN threshold to {best_t:.2f}, where "
            f"F1 is best on this subsample."
        )
    if graph_dist and graph_dist.get("mean", 0) > 0.7:
        suggestions.append(
            f"The graph-level head is over-confident on attack "
            f"(mean {graph_dist['mean']:.2f}) — the node head pulls "
            f"per-flow scores down. If you want graph-level signal "
            f"to drive the cascade, switch the per-flow assignment "
            f"in evaluate_combined_system to use graph_p directly."
        )
    if graph_info.get("avg_degree", {}).get("mean", 99) < 2.0:
        suggestions.append(
            f"Average node degree is low ({graph_info['avg_degree']['mean']:.2f}). "
            f"GAT attention needs neighbour signal — increase "
            f"WINDOW_FLOWS or remove the parallel-edge collapse in "
            f"FlowGraphBuilder if the model is starved."
        )
    if not suggestions:
        suggestions.append(
            "No single failure mode dominates. The 99.5% override likely "
            "comes from the joint domain + threshold mismatch — re-train "
            "the GNN with a 2017+2018 mixture before relying on it for "
            "cross-dataset deployments."
        )
    for s in suggestions:
        print(f"  • {s}")

    # --- Persist ---------------------------------------------------
    out = {
        "n_flows": int(len(y)),
        "n_benign": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
        "per_flow_distribution": flow_dist,
        "per_window_graph_score_distribution": graph_dist,
        "threshold_sweep": sweep_rows,
        "best_threshold_f1": float(best_t),
        "graph_health": graph_info,
        "cascade_default_threshold": m_def,
        "cascade_tuned_threshold": m_tuned,
        "suggestions": suggestions,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=lambda v:
                  float(v) if hasattr(v, "item") else str(v))
    print(f"\nSaved diagnostics to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
