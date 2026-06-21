"""
Train the GAT-based flow GNN on CICIDS2017.

How the dataset is shaped
-------------------------
CICIDS2017's processed parquet drops source / destination IPs in
favour of per-flow statistics, so we can't construct *the* graph the
original capture had. Instead, we synthesise one that preserves the
attack topologies the model needs to learn:

- ``Benign`` rows get random IP pairs from a /16 pool.
- ``DDoS`` / ``DoS`` rows are aimed at a small set of victim IPs from
  a large pool of attacker IPs (high in-degree on the victim).
- ``Portscan`` rows are emitted from a single attacker to many
  destinations (high out-degree on the attacker, ports vary per row).
- ``Botnet`` rows are emitted from a small bot pool to a small C2
  pool (star topology + repeated edges).
- ``Bruteforce`` / ``WebAttacks`` / ``Infiltration`` rows are aimed at
  fixed targets with low-volume attackers.

This is **not** a perfect reconstruction of the original captures —
it's a deliberate synthesis that gives the GNN labelled graphs whose
topology matches each attack class. The node-level label is "is this
endpoint involved in an attack", and the graph-level label is "does
this window contain *any* attack flow". This matches how the runtime
detector is queried.

Outputs
-------
- ``~/sem6el/trained_models/gnn_gat.pt``      (state dict)
- ``~/sem6el/trained_models/gnn_metrics.json`` (val accuracy / AUC)

Run
---
    python training/train_gnn.py [--limit N] [--epochs E] [--window-flows W]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running as a script from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.gnn.graph_builder import (  # noqa: E402
    EDGE_FEATURE_DIM,
    FlowGraphBuilder,
    NODE_FEATURE_DIM,
)


PROCESSED_PATH = os.path.expanduser(
    "~/sem6el/data/processed/processed_data.parquet"
)
LABEL_ENCODER_PATH = os.path.expanduser(
    "~/sem6el/trained_models/label_encoder.joblib"
)
MODEL_OUT = os.path.expanduser("~/sem6el/trained_models/gnn_gat.pt")
METRICS_OUT = os.path.expanduser("~/sem6el/trained_models/gnn_metrics.json")


# IP pool sizes per class. Tuned so the resulting graphs *look like*
# their attack: DDoS has many sources to one sink, scan has one source
# to many sinks, etc.
TOPOLOGY: dict[str, dict] = {
    "Benign":      {"src_pool": 200, "dst_pool": 200, "dst_concentration": 1.0},
    "DDoS":        {"src_pool": 200, "dst_pool": 3,   "dst_concentration": 0.05},
    "DoS":         {"src_pool": 100, "dst_pool": 3,   "dst_concentration": 0.05},
    "Portscan":    {"src_pool": 2,   "dst_pool": 150, "dst_concentration": 1.0},
    "Bruteforce":  {"src_pool": 5,   "dst_pool": 3,   "dst_concentration": 0.2},
    "WebAttacks":  {"src_pool": 8,   "dst_pool": 4,   "dst_concentration": 0.2},
    "Botnet":      {"src_pool": 10,  "dst_pool": 4,   "dst_concentration": 0.3},
    "Infiltration":{"src_pool": 3,   "dst_pool": 5,   "dst_concentration": 0.3},
}


def _pick_attack_name(label_id: int, classes: list[str]) -> str:
    """Map encoded ``attack_label`` -> the topology key. Defaults to
    ``Benign`` for unknown labels."""
    if 0 <= label_id < len(classes):
        name = classes[label_id]
        if name in TOPOLOGY:
            return name
    return "Benign"


def _ip_for(class_name: str, role: str, index: int) -> str:
    """Stable IP synthesis per (class, role, index). The /16 prefix
    encodes the class so the model can't simply memorise IPs."""
    prefix = {
        "Benign":      "203.0",
        "DDoS":        "185.220",
        "DoS":         "45.155",
        "Portscan":    "92.118",
        "Bruteforce":  "103.27",
        "WebAttacks":  "172.16",
        "Botnet":      "172.105",
        "Infiltration":"198.51",
    }.get(class_name, "10.0")
    if role == "dst":
        # Destinations clustered into /24
        return f"10.0.{(index % 250) + 1}.{(index // 250) + 1}"
    return f"{prefix}.{(index % 250) + 1}.{(index // 250) + 1}"


# ---------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------


def build_windows(
    df: pd.DataFrame,
    classes: list[str],
    window_flows: int = 200,
    n_windows: int = 4000,
    attack_prob: float = 0.5,
    seed: int = 42,
) -> list[tuple]:
    """Return a list of (snapshot, node_labels, graph_label) tuples.

    Each window has ``window_flows`` rows drawn either from a single
    attack class (with prob ``attack_prob``) or from Benign rows.
    Attack windows also include a small benign tail so the model
    learns to score only the *attacker / victim* IPs as attack, not
    the whole graph.
    """
    rng = random.Random(seed)

    # The processed parquet keeps the original string label column
    # alongside the encoded ``attack_label`` — restrict to numeric
    # columns so the feature dict is float-castable.
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "attack_label"
    ]

    by_label: dict[int, np.ndarray] = {
        lbl: df.index[df["attack_label"] == lbl].to_numpy()
        for lbl in df["attack_label"].unique()
    }
    benign_lbl = classes.index("Benign") if "Benign" in classes else 0
    benign_idx = by_label.get(benign_lbl, np.array([], dtype=np.int64))
    attack_labels = [lbl for lbl in by_label if lbl != benign_lbl and len(by_label[lbl]) > 0]
    if not attack_labels:
        raise RuntimeError("dataset has no attack rows")

    out: list[tuple] = []
    for w in range(n_windows):
        is_attack = rng.random() < attack_prob and len(attack_labels) > 0
        if is_attack:
            attack_lbl = rng.choice(attack_labels)
            attack_name = _pick_attack_name(attack_lbl, classes)
            n_attack = int(window_flows * rng.uniform(0.4, 0.9))
            n_benign = window_flows - n_attack
            attack_rows = rng.choices(by_label[attack_lbl].tolist(), k=n_attack)
            benign_rows = (
                rng.choices(benign_idx.tolist(), k=n_benign)
                if len(benign_idx) > 0 else []
            )
            row_ids = attack_rows + benign_rows
            rng.shuffle(row_ids)
        else:
            attack_name = "Benign"
            row_ids = rng.choices(
                by_label.get(benign_lbl, []).tolist() or sum(
                    (by_label[l].tolist() for l in by_label), []
                ),
                k=window_flows,
            )

        builder = FlowGraphBuilder(window_seconds=1e9)  # no expiry during build
        attacker_ips: set[str] = set()
        victim_ips: set[str] = set()
        t = time.time()
        for j, ridx in enumerate(row_ids):
            row = df.iloc[ridx]
            row_label = int(row["attack_label"])
            row_is_attack = row_label != benign_lbl
            if row_is_attack:
                class_name = _pick_attack_name(row_label, classes)
            else:
                class_name = "Benign"
            topo = TOPOLOGY.get(class_name, TOPOLOGY["Benign"])
            src_i = rng.randrange(topo["src_pool"])
            # Concentrate destinations for DDoS/DoS — most rows hit
            # the same victim, a few sprinkle elsewhere.
            if rng.random() > topo["dst_concentration"]:
                dst_i = rng.randrange(max(1, topo["dst_pool"]))
            else:
                dst_i = rng.randrange(topo["dst_pool"])
            src_ip = _ip_for(class_name, "src", src_i)
            dst_ip = _ip_for(class_name, "dst", dst_i)
            if row_is_attack:
                attacker_ips.add(src_ip)
                victim_ips.add(dst_ip)

            # Spread timestamps across the synthesised 60s window so
            # the beacon score has signal for periodic attacks.
            ts = t + (j / max(window_flows, 1)) * 60.0
            feats = {col: float(row[col]) for col in numeric_cols}
            builder.add_flow(src_ip, dst_ip, feats, ts=ts)

        snap = builder.build_snapshot(now=t + 120.0)
        if len(snap.node_ids) == 0:
            continue

        # Node labels: 1 if this IP played the attacker/victim role
        # in any attack flow this window, else 0.
        attack_set = attacker_ips | victim_ips
        node_labels = np.array(
            [1.0 if ip in attack_set else 0.0 for ip in snap.node_ids],
            dtype=np.float32,
        )
        graph_label = 1.0 if is_attack else 0.0
        out.append((snap, node_labels, graph_label))
    return out


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from app.models.gnn.gnn_model import GATDetectorModel, save_model

    print(f"Loading processed CICIDS2017 from {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    classes: list[str]
    if os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
        classes = list(le.classes_)
    else:
        classes = ["Benign"]
    print(f"  classes: {classes}")

    print(
        f"Building {args.n_windows} windows of {args.window_flows} flows each "
        f"(attack_prob={args.attack_prob})"
    )
    windows = build_windows(
        df,
        classes=classes,
        window_flows=args.window_flows,
        n_windows=args.n_windows,
        attack_prob=args.attack_prob,
        seed=args.seed,
    )
    print(f"  built {len(windows)} non-empty windows")

    # Train/val split at the window level so attack samples aren't
    # split between the two.
    rng = random.Random(args.seed)
    rng.shuffle(windows)
    split = int(len(windows) * 0.85)
    train_ws, val_ws = windows[:split], windows[split:]

    def to_data(item):
        snap, node_y, graph_y = item
        d = FlowGraphBuilder.snapshot_to_pyg(snap)
        d.y = torch.tensor([graph_y], dtype=torch.float32)
        d.node_y = torch.from_numpy(node_y)
        return d

    train_data = [to_data(w) for w in train_ws]
    val_data = [to_data(w) for w in val_ws]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = GATDetectorModel(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
    ).build().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            node_logits, graph_logits = model(
                batch.x, batch.edge_index,
                edge_attr=batch.edge_attr, batch=batch.batch,
            )
            graph_y = batch.y.float().to(device)
            node_y = batch.node_y.float().to(device)
            loss = (
                F.binary_cross_entropy_with_logits(graph_logits, graph_y)
                + F.binary_cross_entropy_with_logits(node_logits, node_y)
            )
            loss.backward()
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        # Validation
        model.eval()
        graph_correct = 0
        graph_total = 0
        node_correct = 0
        node_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_logits, graph_logits = model(
                    batch.x, batch.edge_index,
                    edge_attr=batch.edge_attr, batch=batch.batch,
                )
                graph_pred = (torch.sigmoid(graph_logits) > 0.5).long()
                graph_correct += int((graph_pred == batch.y.long().to(device)).sum())
                graph_total += int(batch.y.numel())
                node_pred = (torch.sigmoid(node_logits) > 0.5).long()
                node_correct += int(
                    (node_pred == batch.node_y.long().to(device)).sum()
                )
                node_total += int(batch.node_y.numel())
        graph_acc = graph_correct / max(1, graph_total)
        node_acc = node_correct / max(1, node_total)
        print(
            f"epoch {epoch:3d}  loss={train_loss:.4f}  "
            f"graph_acc={graph_acc:.4f}  node_acc={node_acc:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_graph_acc": graph_acc,
            "val_node_acc": node_acc,
        })

        if graph_acc > best_val:
            best_val = graph_acc
            save_model(model, MODEL_OUT)
            print(f"  ↳ saved {MODEL_OUT} (best graph_acc={best_val:.4f})")

    metrics = {
        "best_val_graph_acc": best_val,
        "final_graph_acc": history[-1]["val_graph_acc"] if history else 0.0,
        "final_node_acc": history[-1]["val_node_acc"] if history else 0.0,
        "history": history,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "window_flows": args.window_flows,
            "n_windows": args.n_windows,
            "attack_prob": args.attack_prob,
        },
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")
    print(f"Best graph_acc: {best_val:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the AI-NGFW GAT detector.")
    p.add_argument("--limit", type=int, default=None,
                   help="cap CICIDS rows used for window sampling")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--window-flows", type=int, default=200,
                   help="flows per synthesised window")
    p.add_argument("--n-windows", type=int, default=4000)
    p.add_argument("--attack-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
