"""
Traffic simulator for the demo dashboard.

Loads rows from the processed CICIDS2017 parquet, replays them through the
in-process ensemble at a configurable speed (1x–100x), and stores the recent
decisions in a ring buffer so the React dashboard can poll `/demo/recent`
and `/demo/stats` without writing to disk on the hot path.

Scenarios filter on `attack_label` to deterministically produce the right
kind of traffic for a demo:

    normal_traffic   -> Benign only
    ddos_attack      -> DDoS + DoS (heavy bias toward DDoS)
    port_scan        -> Portscan
    brute_force      -> Bruteforce
    mixed_realistic  -> empirical mix weighted toward benign, with all attacks

Speed semantics: `speed=10` means "10 flows per second" (not 10× wall-clock).
This gives the operator direct control over dashboard throughput.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import pandas as pd


# --- Config ---------------------------------------------------------------

PROCESSED_PATH = os.path.expanduser("~/sem6el/data/processed/processed_data.parquet")
FEATURE_COLS_PATH = os.path.expanduser("~/sem6el/trained_models/feature_columns.joblib")
LABEL_ENCODER_PATH = os.path.expanduser("~/sem6el/trained_models/label_encoder.joblib")

RING_BUFFER_SIZE = 500          # /demo/recent pulls from this
TIMELINE_BUCKETS = 30           # rolling windows for /demo/stats timeline
BUCKET_SECONDS = 2              # seconds per timeline bucket

# label_id -> friendly name (matches trained label_encoder classes_).
ATTACK_NAMES = {
    0: "Benign",
    1: "Botnet",
    2: "Bruteforce",
    3: "DDoS",
    4: "DoS",
    5: "Infiltration",
    6: "Portscan",
    7: "WebAttacks",
}


SCENARIOS: dict[str, list[tuple[int, float]]] = {
    # scenario -> [(label_id, weight)]
    "normal_traffic":   [(0, 1.0)],
    "ddos_attack":      [(3, 0.7), (4, 0.3)],
    "port_scan":        [(6, 1.0)],
    "brute_force":      [(2, 1.0)],
    "mixed_realistic":  [(0, 0.6), (3, 0.1), (4, 0.08), (2, 0.06),
                         (6, 0.06), (1, 0.04), (5, 0.03), (7, 0.03)],
}


# --- Ring buffer + aggregation --------------------------------------------


@dataclass
class FlowRecord:
    ts: str
    src_ip: str
    dst_ip: str
    dst_port: int | None
    protocol: int | None
    attack_type: str
    features: dict[str, float]
    context: dict[str, Any]
    decision: str
    score: float
    trust_level: str
    primary_action: str
    model_scores: dict[str, float]


class SimulatorStore:
    """In-memory ring buffer + rolling aggregations. Thread-safe."""

    def __init__(self, maxlen: int = RING_BUFFER_SIZE):
        self._lock = threading.Lock()
        self._buf: deque[FlowRecord] = deque(maxlen=maxlen)
        self._decision_counts: Counter[str] = Counter()
        self._attack_counts: Counter[str] = Counter()
        self._blocked_ips: Counter[str] = Counter()
        self._heatmap: Counter[tuple[int, int]] = Counter()   # (day, hour)
        self._timeline: deque[dict[str, Any]] = deque(maxlen=TIMELINE_BUCKETS)
        self._bucket_start: float | None = None
        self._bucket_counts: Counter[str] = Counter()
        self._total_sent = 0
        self._started_at: float | None = None

    def mark_started(self):
        with self._lock:
            self._started_at = time.time()
            self._bucket_start = self._started_at

    def mark_stopped(self):
        with self._lock:
            # Flush current bucket.
            self._flush_bucket_locked()

    def record(self, rec: FlowRecord):
        with self._lock:
            self._buf.append(rec)
            self._decision_counts[rec.decision] += 1
            self._total_sent += 1

            if rec.decision in ("BLOCK", "QUARANTINE") or rec.attack_type != "Benign":
                if rec.attack_type != "Benign":
                    self._attack_counts[rec.attack_type] += 1
                if rec.decision in ("BLOCK", "QUARANTINE"):
                    self._blocked_ips[rec.src_ip] += 1
                    now = datetime.now(timezone.utc)
                    self._heatmap[(now.weekday(), now.hour)] += 1

            # Timeline bucket rollover.
            now = time.time()
            if self._bucket_start is None:
                self._bucket_start = now
            if now - self._bucket_start >= BUCKET_SECONDS:
                self._flush_bucket_locked()
                self._bucket_start = now
            self._bucket_counts[rec.decision] += 1

    def _flush_bucket_locked(self):
        if not self._bucket_counts:
            return
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._timeline.append({
            "ts": stamp,
            "ALLOW": self._bucket_counts.get("ALLOW", 0),
            "INSPECT": self._bucket_counts.get("INSPECT", 0),
            "BLOCK": self._bucket_counts.get("BLOCK", 0) + self._bucket_counts.get("QUARANTINE", 0),
        })
        self._bucket_counts.clear()

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._buf)[-limit:]
            items.reverse()
            return [self._record_to_dict(r) for r in items]

    @staticmethod
    def _record_to_dict(r: FlowRecord) -> dict:
        return {
            "ts": r.ts,
            "src_ip": r.src_ip,
            "dst_ip": r.dst_ip,
            "dst_port": r.dst_port,
            "protocol": r.protocol,
            "attack_type": r.attack_type,
            "features": r.features,
            "context": r.context,
            "decision": r.decision,
            "score": r.score,
            "trust_level": r.trust_level,
            "primary_action": r.primary_action,
            "model_scores": r.model_scores,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            if self._bucket_counts:
                # Include in-flight bucket so the live chart keeps moving.
                tl = list(self._timeline) + [{
                    "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "ALLOW": self._bucket_counts.get("ALLOW", 0),
                    "INSPECT": self._bucket_counts.get("INSPECT", 0),
                    "BLOCK": self._bucket_counts.get("BLOCK", 0) + self._bucket_counts.get("QUARANTINE", 0),
                }]
            else:
                tl = list(self._timeline)

            elapsed = (time.time() - self._started_at) if self._started_at else 0.0
            fps = self._total_sent / elapsed if elapsed > 0 else 0.0

            return {
                "counts": {
                    "ALLOW": self._decision_counts.get("ALLOW", 0),
                    "INSPECT": self._decision_counts.get("INSPECT", 0),
                    "BLOCK": self._decision_counts.get("BLOCK", 0) + self._decision_counts.get("QUARANTINE", 0),
                },
                "flows_per_sec": round(fps, 2),
                "total_sent": self._total_sent,
                "timeline": tl,
                "attack_types": dict(self._attack_counts.most_common()),
                "top_blocked_ips": self._blocked_ips.most_common(10),
                "heatmap": [
                    {"day": d, "hour": h, "count": c}
                    for (d, h), c in self._heatmap.items()
                ],
            }


# --- Simulator ------------------------------------------------------------


class TrafficSimulator:
    """Replays CICIDS2017 rows through the ensemble.

    `predict_fn(features, context) -> dict` is injected so the simulator can
    re-use the API's `_run_ensemble` path and keep decision logic in one place.
    """

    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.store = SimulatorStore()
        self._df: pd.DataFrame | None = None
        self._feature_columns: list[str] | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._scenario: str | None = None
        self._speed: float = 10.0

    # --- lifecycle ----------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "running": self.running,
            "scenario": self._scenario,
            "speed": self._speed,
            "sent": self.store._total_sent,
            "scenarios": list(SCENARIOS.keys()),
        }

    def start(self, scenario: str, speed: float) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if self.running:
            raise RuntimeError("simulator already running")
        self._ensure_data_loaded()
        self._scenario = scenario
        self._speed = max(1.0, min(float(speed), 100.0))
        self._stop.clear()
        self.store = SimulatorStore()  # fresh stats per run
        self.store.mark_started()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        self.store.mark_stopped()

    # --- data ---------------------------------------------------------

    def _ensure_data_loaded(self) -> None:
        if self._df is not None:
            return
        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(f"processed data not found: {PROCESSED_PATH}")
        df = pd.read_parquet(PROCESSED_PATH)
        self._feature_columns = joblib.load(FEATURE_COLS_PATH)
        # Keep model features + label, plus Destination Port / Protocol for
        # display (they may have been dropped by feature selection).
        keep = set(self._feature_columns) | {"attack_label", "Destination Port", "Protocol"}
        cols = [c for c in df.columns if c in keep]
        self._df = df[cols].copy()

    def _pick_row(self, label_id: int) -> pd.Series:
        subset = self._df[self._df["attack_label"] == label_id]
        return subset.iloc[random.randrange(len(subset))]

    def _sample_label(self) -> int:
        weights = SCENARIOS[self._scenario]
        labels = [w[0] for w in weights]
        probs = [w[1] for w in weights]
        return random.choices(labels, probs, k=1)[0]

    def _row_to_flow(self, row: pd.Series, label_id: int) -> tuple[dict, dict, int | None, int | None]:
        feats = {c: float(row[c]) for c in self._feature_columns if c in row.index}
        dst_port = (
            int(row["Destination Port"])
            if "Destination Port" in row.index else self._synth_dst_port(label_id)
        )
        protocol = int(row["Protocol"]) if "Protocol" in row.index else None

        # Synthetic network context (the CICIDS parquet dropped IPs).
        src_ip = self._synth_src_ip(label_id)
        dst_ip = "10.0.0." + str(random.randint(10, 250))

        ctx = {
            "src_ip": src_ip,
            "dst_port": dst_port,
            "is_internal_src": False,
            "is_authenticated": label_id == 0 and random.random() < 0.7,
            "prior_violations": random.randint(2, 6) if label_id != 0 else 0,
            "session_risk": 0.4 if label_id in (2, 3, 4) else 0.0,
            "asset_sensitivity": "high" if dst_port in (22, 3389) else "normal",
            "attack_type": ATTACK_NAMES[label_id],
        }
        feats_with_ports = {**feats, "src_ip": src_ip, "dst_ip": dst_ip}
        return feats, ctx, dst_port, protocol

    @staticmethod
    def _synth_dst_port(label_id: int) -> int:
        if label_id == 2:    # Bruteforce: SSH / RDP / FTP
            return random.choice([22, 22, 22, 3389, 21])
        if label_id == 6:    # Portscan: random high plus common
            return random.choice([22, 80, 443, 3306, 8080, 21, 23, 25, 445, 1433])
        if label_id == 7:    # WebAttacks
            return random.choice([80, 443, 8080, 8443])
        if label_id in (3, 4):  # DDoS / DoS
            return random.choice([80, 80, 443, 53])
        if label_id == 1:    # Botnet: C2 over common outbound ports
            return random.choice([443, 8443, 6667])
        if label_id == 5:    # Infiltration: often SMB / RDP
            return random.choice([445, 3389, 22])
        # Benign: weighted common ports
        return random.choices(
            [80, 443, 53, 22, 3306, 25, 587, 143, 993, 110],
            weights=[25, 40, 10, 5, 5, 3, 3, 3, 3, 3],
            k=1,
        )[0]

    @staticmethod
    def _synth_src_ip(label_id: int) -> str:
        # Attackers get clustered /24s so "top blocked IPs" looks meaningful.
        if label_id == 3:    # DDoS — large botnet spray
            return f"185.220.{random.randint(100,120)}.{random.randint(1,254)}"
        if label_id == 4:    # DoS
            return f"45.155.{random.randint(10,30)}.{random.randint(1,254)}"
        if label_id == 6:    # Portscan
            return f"92.118.{random.randint(20,40)}.{random.randint(1,254)}"
        if label_id == 2:    # Bruteforce
            return f"103.27.{random.randint(50,70)}.{random.randint(1,254)}"
        if label_id in (1, 5, 7):
            return f"172.105.{random.randint(200,220)}.{random.randint(1,254)}"
        # Benign — broad residential-looking range
        return f"203.0.{random.randint(100,200)}.{random.randint(1,254)}"

    # --- main loop ----------------------------------------------------

    def _run(self) -> None:
        interval = 1.0 / self._speed
        while not self._stop.is_set():
            start = time.perf_counter()
            try:
                label_id = self._sample_label()
                row = self._pick_row(label_id)
                feats, ctx, dst_port, protocol = self._row_to_flow(row, label_id)

                result = self.predict_fn(feats, ctx)

                rec = FlowRecord(
                    ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    src_ip=ctx["src_ip"],
                    dst_ip="10.0.0." + str(random.randint(10, 250)),
                    dst_port=dst_port,
                    protocol=protocol,
                    attack_type=ATTACK_NAMES[label_id],
                    features=feats,
                    context=ctx,
                    decision=result["decision"],
                    score=float(result["risk_score"]),
                    trust_level=result["zero_trust"]["trust_level"],
                    primary_action=result["zero_trust"]["primary_action"],
                    model_scores={
                        k: float(v) for k, v in result["model_scores"].items()
                    },
                )
                self.store.record(rec)
            except Exception:
                # Simulator must not crash the API; swallow and continue.
                pass

            # Pace loop to target speed (flows per second).
            elapsed = time.perf_counter() - start
            sleep = interval - elapsed
            if sleep > 0:
                self._stop.wait(sleep)
