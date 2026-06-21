"""
End-to-end evaluation of the trained GAT detector.

Sections
--------
1. **Load test** — load the .pt checkpoint, count params, print arch.
2. **Build test graph** — construct a graph from real CICIDS-2017 rows
   and report node / edge / degree stats.
3. **Inference test** — run a single forward pass, measure latency.
4. **Pattern detection** — synthesise the three topologies the GNN
   was trained to spot (DDoS, port scan, benign) and check whether
   the model + topology hints flag them correctly.
5. **Accuracy on CICIDS-2017** — sample labelled windows from the
   processed parquet (reusing :func:`training.train_gnn.build_windows`)
   and compute accuracy / precision / recall / F1 / ROC-AUC.
6. **Cross-dataset on CICIDS-2018** — same evaluation on the unseen
   2018 CSV mirror loaded by :mod:`training.evaluate_cicids2018`.

Results are written to ``training/gnn_evaluation_results.json``.

Run
---
    venv/bin/python training/evaluate_gnn.py [--cicids2017-windows N]
                                             [--cicids2018-windows N]
                                             [--window-flows N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from app.models.gnn.graph_builder import (  # noqa: E402
    FlowGraphBuilder,
)
from app.models.gnn.gnn_model import load_model  # noqa: E402
from training.train_gnn import build_windows  # noqa: E402
from training.evaluate_cicids2018 import (  # noqa: E402
    DATA_DIR as CICIDS2018_DIR,
    _normalise_columns,
    _label_to_binary,
)


# ---------------------------------------------------------------------
# Paths + defaults
# ---------------------------------------------------------------------

MODELS_PATH = os.path.expanduser("~/sem6el/trained_models")
GNN_MODEL_PATH = os.path.join(MODELS_PATH, "gnn_gat.pt")
LABEL_ENCODER_PATH = os.path.join(MODELS_PATH, "label_encoder.joblib")
PROCESSED_PARQUET = os.path.expanduser(
    "~/sem6el/data/processed/processed_data.parquet"
)
RESULTS_PATH = _PROJECT_ROOT / "training" / "gnn_evaluation_results.json"

WIDTH = 64
DECISION_THRESHOLD = 0.5     # graph-level prob >= → predicted attack


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def banner(title: str) -> None:
    print("\n" + "═" * WIDTH)
    print(title.center(WIDTH))
    print("═" * WIDTH)


def section(title: str) -> None:
    print("\n" + title)
    print("─" * WIDTH)


def kv(label: str, value, ok: bool | None = None) -> None:
    """Aligned key/value line, optionally with a status marker."""
    suffix = ""
    if ok is True:
        suffix = " ✅"
    elif ok is False:
        suffix = " ❌"
    print(f"  {label:<22} {value}{suffix}")


# ---------------------------------------------------------------------
# 1. Load + architecture
# ---------------------------------------------------------------------


def section_load_test() -> tuple[torch.nn.Module, dict]:
    section("MODEL INFO")
    if not os.path.exists(GNN_MODEL_PATH):
        kv("Model file", GNN_MODEL_PATH, ok=False)
        raise FileNotFoundError(
            f"GNN model not found at {GNN_MODEL_PATH}. "
            f"Train it first: python training/train_gnn.py"
        )
    model = load_model(GNN_MODEL_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    arch_lines = [
        line.strip() for line in repr(model).splitlines()
        if "GATConv" in line or "Linear" in line
    ]
    # Build a short architecture description from the live module.
    gat_layers = [m for m in model.modules() if "GATConv" in type(m).__name__]
    heads = getattr(gat_layers[0], "heads", "?") if gat_layers else "?"
    hidden = getattr(gat_layers[0], "out_channels", "?") if gat_layers else "?"
    description = f"GAT ({len(gat_layers)} GATConv layers, {hidden} hidden, {heads} heads)"

    kv("Model file", GNN_MODEL_PATH)
    kv("Architecture", description)
    kv("Parameters", f"{n_params:,}")
    kv("Status", "Loaded successfully", ok=True)

    return model, {
        "model_file": GNN_MODEL_PATH,
        "architecture": description,
        "n_parameters": int(n_params),
        "module_summary": arch_lines,
    }


# ---------------------------------------------------------------------
# 2. Build test graph from real CICIDS-2017 data
# ---------------------------------------------------------------------


def _load_cicids2017(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Load a random slice of the processed parquet."""
    if not os.path.exists(PROCESSED_PARQUET):
        raise FileNotFoundError(PROCESSED_PARQUET)
    df = pd.read_parquet(PROCESSED_PARQUET)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_rows, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _synth_ip(label_id: int, role: str, i: int) -> str:
    """Match the synthesis used in :mod:`training.train_gnn` so the
    graphs the model sees at eval look like the ones it trained on."""
    prefix = {
        0: "203.0",       # Benign
        3: "185.220",     # DDoS
        4: "45.155",      # DoS
        6: "92.118",      # Portscan
        2: "103.27",      # Bruteforce
        1: "172.105",     # Botnet
        7: "172.16",      # WebAttacks
        5: "198.51",      # Infiltration
    }.get(label_id, "10.0")
    if role == "dst":
        return f"10.0.{(i % 250) + 1}.{(i // 250) + 1}"
    return f"{prefix}.{(i % 250) + 1}.{(i // 250) + 1}"


def section_build_test_graph(rng: random.Random, n_rows: int = 1000) -> tuple[FlowGraphBuilder, dict]:
    section("GRAPH CONSTRUCTION TEST")
    df = _load_cicids2017(n_rows=n_rows)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "attack_label"
    ]

    builder = FlowGraphBuilder(window_seconds=1e9)
    t0 = time.time()
    for j, row in df.iterrows():
        label = int(row.get("attack_label", 0))
        src = _synth_ip(label, "src", rng.randrange(200))
        dst = _synth_ip(label, "dst", rng.randrange(200))
        feats = {c: float(row[c]) for c in numeric_cols}
        builder.add_flow(src, dst, feats, ts=t0 + (j / n_rows) * 60.0)

    snap = builder.build_snapshot(now=t0 + 120.0)
    avg_deg = (snap.edge_index.shape[1] * 2.0) / max(1, len(snap.node_ids))

    kv("Test flows", f"{n_rows:,}")
    kv("Nodes created", f"{len(snap.node_ids):,}")
    kv("Edges created", f"{snap.edge_index.shape[1]:,}")
    kv("Avg degree", f"{avg_deg:.1f}")
    kv("Status", "Graph built successfully", ok=True)

    info = {
        "test_flows": n_rows,
        "nodes": int(len(snap.node_ids)),
        "edges": int(snap.edge_index.shape[1]),
        "avg_degree": float(avg_deg),
    }
    return builder, info


# ---------------------------------------------------------------------
# 3. Inference test
# ---------------------------------------------------------------------


def section_inference_test(model: torch.nn.Module, builder: FlowGraphBuilder) -> dict:
    section("INFERENCE TEST")
    snap = builder.build_snapshot()
    data = FlowGraphBuilder.snapshot_to_pyg(snap)

    # Warm-up forward (PyG can lazy-compile on first call).
    with torch.no_grad():
        model(data.x, data.edge_index, edge_attr=data.edge_attr)

    t0 = time.perf_counter()
    node_p, graph_p = model.predict(data)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    threat_label = "Attack" if graph_p >= DECISION_THRESHOLD else "Normal"
    suspicious = int((node_p >= DECISION_THRESHOLD).sum())

    kv("Graph threat score", f"{graph_p:.3f} ({threat_label})")
    kv("Suspicious nodes", f"{suspicious}/{len(node_p)}")
    kv("Inference time", f"{dt_ms:.0f} ms")
    kv("Status", "Inference working", ok=True)

    return {
        "graph_threat_score": float(graph_p),
        "suspicious_nodes": suspicious,
        "total_nodes": int(len(node_p)),
        "inference_ms": float(dt_ms),
    }


# ---------------------------------------------------------------------
# 4. Pattern detection on synthetic topologies
# ---------------------------------------------------------------------


def _ddos_flows(n_sources: int = 50, victim: str = "10.0.0.5") -> FlowGraphBuilder:
    b = FlowGraphBuilder(window_seconds=1e9)
    t = time.time()
    for i in range(n_sources):
        for k in range(4):           # multiple flows per source -> heavy in-degree
            b.add_flow(
                f"185.220.{i // 250 + 1}.{i % 250 + 1}",
                victim,
                {"fwd_bytes": 50, "bwd_bytes": 0, "duration": 1e5,
                 "fwd_packets": 1, "bwd_packets": 0, "byte_rate": 5e2,
                 "packet_rate": 10.0, "dst_port": 80, "protocol": 6},
                ts=t + i * 0.1 + k * 0.01,
            )
    return b


def _scan_flows(n_targets: int = 30, attacker: str = "92.118.30.1") -> FlowGraphBuilder:
    b = FlowGraphBuilder(window_seconds=1e9)
    rng = random.Random(11)
    t = time.time()
    # Jittered timestamps — uniform spacing would trip the beacon /
    # C2 heuristic instead of the scan one. Real port scanners are
    # bursty, not metronomic.
    for i in range(n_targets):
        b.add_flow(
            attacker,
            f"10.0.0.{i + 10}",
            {"fwd_bytes": 40, "bwd_bytes": 0, "duration": 5e4,
             "fwd_packets": 1, "bwd_packets": 0, "byte_rate": 80,
             "packet_rate": 20.0, "dst_port": (i + 20) % 65535, "protocol": 6},
            ts=t + rng.uniform(0.0, 5.0),
        )
    return b


def _normal_flows(n_flows: int = 60) -> FlowGraphBuilder:
    """Random benign-looking traffic — many src/dst pairs, varied
    byte sizes, no high in-degree, no high out-degree."""
    b = FlowGraphBuilder(window_seconds=1e9)
    rng = random.Random(7)
    t = time.time()
    for i in range(n_flows):
        src = f"203.0.{rng.randint(100, 200)}.{rng.randint(1, 254)}"
        dst = f"10.1.{rng.randint(0, 50)}.{rng.randint(1, 254)}"
        size = rng.randint(2_000, 50_000)
        b.add_flow(
            src, dst,
            {"fwd_bytes": size, "bwd_bytes": size * rng.randint(2, 8),
             "duration": rng.randint(int(1e5), int(5e6)),
             "fwd_packets": rng.randint(2, 30),
             "bwd_packets": rng.randint(2, 50),
             "byte_rate": rng.randint(1_000, 50_000),
             "packet_rate": rng.uniform(1.0, 50.0),
             "dst_port": rng.choice([80, 443, 53, 22, 25, 110, 143, 587]),
             "protocol": 6},
            ts=t + i * 0.5,
        )
    return b


def _score_pattern(model, builder: FlowGraphBuilder) -> dict:
    snap = builder.build_snapshot()
    data = FlowGraphBuilder.snapshot_to_pyg(snap)
    node_p, graph_p = model.predict(data)

    # Pattern selection: pick the highest-severity topology hint.
    type_label_map = {
        "ddos": "POTENTIAL_DDOS",
        "scan": "POTENTIAL_SCAN",
        "c2": "POTENTIAL_C2",
    }
    best_pattern = None
    best_severity = 0.0
    best_source = None
    for cat in ("ddos", "scan", "c2"):
        for h in snap.topology_hints.get(cat, []):
            sev = float(h.get("severity", 0.5))
            if sev > best_severity:
                best_severity = sev
                best_pattern = type_label_map[cat]
                best_source = h.get("ip")
    # Confidence blends the model's max node probability with the
    # topology hint severity — both signals agreeing means high
    # confidence.
    if best_pattern is not None:
        confidence = float(max(best_severity, float(np.max(node_p))))
    else:
        confidence = 0.0

    return {
        "graph_threat_score": float(graph_p),
        "max_node_score": float(np.max(node_p)) if len(node_p) else 0.0,
        "suspicious_nodes": int((node_p >= DECISION_THRESHOLD).sum()),
        "pattern": best_pattern,
        "pattern_source": best_source,
        "confidence": confidence,
        "topology_hints": {
            cat: [h.get("ip") for h in snap.topology_hints.get(cat, [])]
            for cat in ("ddos", "scan", "c2")
        },
    }


def section_pattern_detection(model) -> dict:
    section("PATTERN DETECTION TEST")

    tests = [
        ("DDoS Pattern (50 sources → 1 target)", "POTENTIAL_DDOS", _ddos_flows(), True),
        ("Port Scan Pattern (1 source → 30 targets)", "POTENTIAL_SCAN", _scan_flows(), True),
        ("Normal Traffic (random connections)", None, _normal_flows(), False),
    ]
    results = []
    for i, (label, expected, builder, expect_high) in enumerate(tests, 1):
        r = _score_pattern(model, builder)
        verdict_marker = "HIGH" if r["graph_threat_score"] >= 0.6 else "LOW"
        # ✅ if (expect_high and HIGH) or (not expect_high and LOW), else ⚠
        correct = (expect_high and verdict_marker == "HIGH") or \
                  (not expect_high and verdict_marker == "LOW")
        ok_mark = "✅" if correct else "⚠️"
        print(f"\n  Test {i}: {label}")
        print(f"    Graph threat score:     {r['graph_threat_score']:.2f} {ok_mark} {verdict_marker}")
        print(f"    Pattern detected:       {r['pattern'] or 'None'}")
        if r["pattern"]:
            print(f"    Confidence:             {r['confidence']*100:.0f}%")
            print(f"    Pattern source:         {r['pattern_source']}")
        else:
            print(f"    Confidence:             N/A")
        results.append({
            "name": label,
            "expected_pattern": expected,
            "expected_high_threat": expect_high,
            **r,
            "verdict": verdict_marker,
            "correct": correct,
        })
    return {"tests": results}


# ---------------------------------------------------------------------
# 5. Accuracy on CICIDS-2017
# ---------------------------------------------------------------------


def _eval_windows(model, windows, label: str) -> dict:
    """Run inference on a list of (snap, node_y, graph_y) tuples and
    compute the standard metric block."""
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    t0 = time.perf_counter()
    for snap, _node_y, graph_y in windows:
        data = FlowGraphBuilder.snapshot_to_pyg(snap)
        _node_p, graph_p = model.predict(data)
        y_true.append(int(graph_y))
        y_score.append(float(graph_p))
        y_pred.append(1 if graph_p >= DECISION_THRESHOLD else 0)
    dt = time.perf_counter() - t0
    if not y_true:
        return {}
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        auc = None
    return {
        "label": label,
        "n_windows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "inference_seconds_total": float(dt),
        "ms_per_window": float(dt / len(y_true) * 1000.0),
    }


def _print_metrics_block(m: dict) -> None:
    if not m:
        print("  (no windows evaluated)")
        return
    print(f"  Windows evaluated:  {m['n_windows']:,}")
    print()
    print(f"  Graph-Level (Is network under attack?):")
    print(f"    Accuracy:     {m['accuracy']*100:.1f}%")
    print(f"    Precision:    {m['precision']*100:.1f}%")
    print(f"    Recall:       {m['recall']*100:.1f}%")
    print(f"    F1-Score:     {m['f1_score']*100:.1f}%")
    print(f"    ROC-AUC:      {m['roc_auc']:.3f}" if m["roc_auc"] is not None else "    ROC-AUC:      n/a")
    cm = np.array(m["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    print()
    print(f"  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Normal  Attack")
    print(f"    Actual Normal   {tn:>6}  {fp:>6}")
    print(f"           Attack   {fn:>6}  {tp:>6}")
    print(f"\n  Latency:        {m['ms_per_window']:.1f} ms / window")


def section_accuracy_cicids2017(
    model,
    n_windows: int,
    window_flows: int,
) -> dict:
    section("ACCURACY ON CICIDS-2017 TEST SET")

    if not os.path.exists(PROCESSED_PARQUET):
        print(f"  (skipped: {PROCESSED_PARQUET} not found)")
        return {}

    df = pd.read_parquet(PROCESSED_PARQUET)
    # Hold out a deterministic eval slice so this isn't measuring
    # against rows that may have appeared in training windows.
    eval_df = df.sample(frac=0.2, random_state=99).reset_index(drop=True)
    if os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
        classes = list(le.classes_)
    else:
        classes = ["Benign"]

    windows = build_windows(
        eval_df,
        classes=classes,
        window_flows=window_flows,
        n_windows=n_windows,
        attack_prob=0.5,
        seed=2024,
    )
    metrics = _eval_windows(model, windows, label="CICIDS-2017")
    _print_metrics_block(metrics)
    return metrics


# ---------------------------------------------------------------------
# 6. Cross-dataset on CICIDS-2018
# ---------------------------------------------------------------------


def _load_cicids2018_df() -> pd.DataFrame:
    """Pull the cleaned 2018 CSV, normalise column names, encode the
    integer Label down to {0=Benign, 1=Attack}, and return a frame
    shaped like the CICIDS-2017 ``attack_label``-bearing one."""
    files = sorted(
        p for p in os.listdir(CICIDS2018_DIR)
        if p.endswith(".csv") or p.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"No CICIDS-2018 data under {CICIDS2018_DIR}"
        )
    frames = []
    for f in files:
        path = os.path.join(CICIDS2018_DIR, f)
        df = pd.read_csv(path, low_memory=False) if f.endswith(".csv") \
            else pd.read_parquet(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = _normalise_columns(df)
        label_col = next(
            (c for c in df.columns if c.lower() == "label"), None
        )
        if label_col is None:
            continue
        y = _label_to_binary(df[label_col])
        df = df.drop(columns=[label_col])
        df["attack_label"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def section_cross_dataset_2018(
    model,
    n_windows: int,
    window_flows: int,
) -> dict:
    section("CROSS-DATASET: CICIDS-2018")

    try:
        df = _load_cicids2018_df()
    except FileNotFoundError as e:
        print(f"  (skipped: {e})")
        return {}

    # The 2018 schema only carries Benign vs. Attack at this point —
    # build_windows just needs a `classes` list where the chosen
    # benign label (0) maps to the entry named "Benign".
    classes = ["Benign", "Attack"]
    # build_windows samples by attack_label.unique() so we need both
    # labels present in eval_df.
    eval_df = df.sample(
        n=min(len(df), 200_000), random_state=99
    ).reset_index(drop=True)

    windows = build_windows(
        eval_df,
        classes=classes,
        window_flows=window_flows,
        n_windows=n_windows,
        attack_prob=0.5,
        seed=2024,
    )
    metrics = _eval_windows(model, windows, label="CICIDS-2018")
    if metrics:
        print(f"  Windows evaluated:  {metrics['n_windows']:,}")
        print(f"  Accuracy:           {metrics['accuracy']*100:.1f}%")
        print(f"  F1-Score:           {metrics['f1_score']*100:.1f}%")
        if metrics["roc_auc"] is not None:
            print(f"  ROC-AUC:            {metrics['roc_auc']:.3f}")
        print(f"  Note:               Expected drop due to domain shift")
    return metrics


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    banner("GNN EVALUATION REPORT")

    model, model_info = section_load_test()
    builder, graph_info = section_build_test_graph(rng, n_rows=args.graph_test_flows)
    inference_info = section_inference_test(model, builder)
    pattern_info = section_pattern_detection(model)
    cicids2017_metrics = section_accuracy_cicids2017(
        model,
        n_windows=args.cicids2017_windows,
        window_flows=args.window_flows,
    )
    cicids2018_metrics = section_cross_dataset_2018(
        model,
        n_windows=args.cicids2018_windows,
        window_flows=args.window_flows,
    )

    banner("✅ GNN EVALUATION COMPLETE")

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_info,
        "graph_construction": graph_info,
        "inference": inference_info,
        "pattern_detection": pattern_info,
        "accuracy_cicids2017": cicids2017_metrics,
        "accuracy_cicids2018": cicids2018_metrics,
        "config": {
            "decision_threshold": DECISION_THRESHOLD,
            "graph_test_flows": args.graph_test_flows,
            "cicids2017_windows": args.cicids2017_windows,
            "cicids2018_windows": args.cicids2018_windows,
            "window_flows": args.window_flows,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved evaluation results to {RESULTS_PATH}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end evaluation of the trained GAT detector."
    )
    p.add_argument("--graph-test-flows", type=int, default=1000,
                   help="rows of CICIDS-2017 used in the graph-construction test")
    p.add_argument("--cicids2017-windows", type=int, default=500,
                   help="number of 60-s windows to score for accuracy")
    p.add_argument("--cicids2018-windows", type=int, default=200,
                   help="number of 60-s windows for cross-dataset eval")
    p.add_argument("--window-flows", type=int, default=200,
                   help="flows per window")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
