"""
Flow -> graph conversion for the GNN layer.

Each call to :meth:`FlowGraphBuilder.add_flow` adds an (src_ip, dst_ip,
features, timestamp) tuple to a rolling buffer. :meth:`build_graph`
materialises the flows in the last ``window_seconds`` into a PyTorch
Geometric ``Data`` object:

- One **node per distinct IP** observed in the window.
- One **directed edge per flow** (parallel edges collapse to a single
  edge whose features are the per-flow mean — this keeps adjacency
  sparse without losing magnitude information).

Node features are aggregations that expose the three topologies we
care about:

- **DDoS**           -> very high *in-degree* on a single dst
- **Port / host scan** -> very high *out-degree* on a single src,
                          plus high *unique-dst-port* count
- **C2 beacon (star)** -> one node with many low-volume edges to
                          distinct peers + low byte_rate variance

These hints are also returned in plain-Python form by
:meth:`topology_hints` so the API can surface them even when the
GNN model itself has not been loaded yet.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Edge feature schema. Kept short and dataset-agnostic so the same
# graph builder works for CICIDS2017 training and for live API traffic.
EDGE_FEATURES: list[str] = [
    "duration",
    "fwd_bytes",
    "bwd_bytes",
    "total_bytes",
    "fwd_packets",
    "bwd_packets",
    "byte_rate",
    "packet_rate",
    "dst_port",
    "proto_tcp",
    "proto_udp",
]

# Node feature schema. Recomputed from the edge set on every build.
NODE_FEATURES: list[str] = [
    "in_degree",
    "out_degree",
    "unique_in_peers",
    "unique_out_peers",
    "unique_dst_ports",
    "total_bytes_sent",
    "total_bytes_recv",
    "mean_byte_rate",
    "mean_duration",
    "syn_like_ratio",      # short, low-byte, fan-out -> scan-y
    "beacon_score",        # many small, regularly spaced flows -> C2-y
]

NODE_FEATURE_DIM = len(NODE_FEATURES)
EDGE_FEATURE_DIM = len(EDGE_FEATURES)


@dataclass
class FlowEvent:
    ts: float
    src_ip: str
    dst_ip: str
    features: dict[str, float]


@dataclass
class GraphSnapshot:
    """Plain-Python representation of the current graph.

    Used both as the input to the GNN (after conversion to a PyG
    ``Data`` object) and as the payload for ``GET /network/graph``,
    so the dashboard does not need PyTorch installed to render it.
    """

    node_ids: list[str]
    node_index: dict[str, int]
    node_features: np.ndarray   # (N, NODE_FEATURE_DIM)
    edge_index: np.ndarray      # (2, E)
    edge_features: np.ndarray   # (E, EDGE_FEATURE_DIM)
    window_seconds: float
    flow_count: int
    topology_hints: dict[str, Any] = field(default_factory=dict)


class FlowGraphBuilder:
    """Rolling 60-second flow buffer with thread-safe ingestion."""

    def __init__(self, window_seconds: float = 60.0, max_buffer: int = 5000):
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._buffer: deque[FlowEvent] = deque(maxlen=max_buffer)

    # --- ingestion ----------------------------------------------------

    def add_flow(
        self,
        src_ip: str | None,
        dst_ip: str | None,
        features: dict[str, float],
        ts: float | None = None,
    ) -> None:
        """Record a flow. Missing IPs are dropped — there is no graph
        without endpoints."""
        if not src_ip or not dst_ip:
            return
        evt = FlowEvent(
            ts=float(ts) if ts is not None else time.time(),
            src_ip=str(src_ip),
            dst_ip=str(dst_ip),
            features=dict(features),
        )
        with self._lock:
            self._buffer.append(evt)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def _active_window(self, now: float | None = None) -> list[FlowEvent]:
        now = time.time() if now is None else now
        cutoff = now - self.window_seconds
        with self._lock:
            return [e for e in self._buffer if e.ts >= cutoff]

    # --- normalisation ------------------------------------------------

    @staticmethod
    def _edge_vec_from_features(f: dict[str, float]) -> np.ndarray:
        """Pull edge features out of a flow dict.

        Accepts both CICIDS2017 column names (``"Flow Duration"``) and
        already-normalised keys (``"duration"``) so the same builder
        works for both training and live API traffic.
        """
        def g(*keys: str, default: float = 0.0) -> float:
            for k in keys:
                if k in f and f[k] is not None:
                    try:
                        return float(f[k])
                    except (TypeError, ValueError):
                        continue
            return default

        duration = g("duration", "Flow Duration") / 1e6 if "Flow Duration" in f else g("duration", "Flow Duration")
        fwd_bytes = g("fwd_bytes", "Total Length of Fwd Packets", "Fwd Packets Length Total")
        bwd_bytes = g("bwd_bytes", "Total Length of Bwd Packets", "Bwd Packets Length Total")
        fwd_pkts = g("fwd_packets", "Total Fwd Packets")
        bwd_pkts = g("bwd_packets", "Total Backward Packets")
        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts
        byte_rate = g("byte_rate", "Flow Bytes/s")
        if byte_rate == 0.0 and duration > 0:
            byte_rate = total_bytes / max(duration, 1e-6)
        packet_rate = g("packet_rate", "Flow Packets/s")
        if packet_rate == 0.0 and duration > 0:
            packet_rate = total_pkts / max(duration, 1e-6)

        dst_port = g("dst_port", "Destination Port")
        proto = g("protocol", "Protocol")
        proto_tcp = 1.0 if int(proto) == 6 else 0.0
        proto_udp = 1.0 if int(proto) == 17 else 0.0

        # Log-compress the heavy-tailed magnitudes — DDoS bytes can be
        # 10^7+ while a benign flow is 10^2, and raw values destabilise
        # GAT attention.
        return np.array([
            math.log1p(max(duration, 0.0)),
            math.log1p(max(fwd_bytes, 0.0)),
            math.log1p(max(bwd_bytes, 0.0)),
            math.log1p(max(total_bytes, 0.0)),
            math.log1p(max(fwd_pkts, 0.0)),
            math.log1p(max(bwd_pkts, 0.0)),
            math.log1p(max(byte_rate, 0.0)),
            math.log1p(max(packet_rate, 0.0)),
            float(dst_port) / 65535.0,
            proto_tcp,
            proto_udp,
        ], dtype=np.float32)

    # --- graph construction -------------------------------------------

    def build_snapshot(self, now: float | None = None) -> GraphSnapshot:
        """Build a graph snapshot for the active window."""
        events = self._active_window(now=now)
        return self._snapshot_from_events(events)

    def build_from_flows(self, flows: list[dict[str, Any]]) -> GraphSnapshot:
        """Build a one-shot snapshot from an explicit flow list.

        Used by ``POST /predict/gnn`` when the caller wants to score a
        batch of flows against a self-contained graph rather than the
        rolling buffer. Each flow dict must contain ``src_ip``,
        ``dst_ip`` and a ``features`` map (or be flat — both forms are
        accepted).
        """
        events: list[FlowEvent] = []
        for f in flows:
            feats = f.get("features", f)
            src = f.get("src_ip") or feats.get("src_ip")
            dst = f.get("dst_ip") or feats.get("dst_ip")
            ts = f.get("ts") or time.time()
            if not src or not dst:
                continue
            events.append(FlowEvent(ts=float(ts), src_ip=str(src),
                                    dst_ip=str(dst), features=dict(feats)))
        return self._snapshot_from_events(events)

    def _snapshot_from_events(self, events: list[FlowEvent]) -> GraphSnapshot:
        if not events:
            return GraphSnapshot(
                node_ids=[],
                node_index={},
                node_features=np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_features=np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
                window_seconds=self.window_seconds,
                flow_count=0,
                topology_hints={"ddos": [], "scan": [], "c2": []},
            )

        # 1) Index unique IPs as graph nodes.
        node_index: dict[str, int] = {}
        for e in events:
            for ip in (e.src_ip, e.dst_ip):
                if ip not in node_index:
                    node_index[ip] = len(node_index)
        node_ids = list(node_index.keys())
        N = len(node_ids)

        # 2) Collapse parallel flows (same src->dst) into one mean
        #    edge. Track raw flow contributions so we can compute node
        #    statistics + topology hints without re-walking events.
        pair_feats: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        pair_timestamps: dict[tuple[int, int], list[float]] = defaultdict(list)
        pair_dst_ports: dict[tuple[int, int], list[float]] = defaultdict(list)

        for e in events:
            u = node_index[e.src_ip]
            v = node_index[e.dst_ip]
            vec = self._edge_vec_from_features(e.features)
            pair_feats[(u, v)].append(vec)
            pair_timestamps[(u, v)].append(e.ts)
            pair_dst_ports[(u, v)].append(float(
                e.features.get("dst_port",
                e.features.get("Destination Port", 0.0)) or 0.0
            ))

        edges = list(pair_feats.keys())
        E = len(edges)
        edge_index = np.zeros((2, E), dtype=np.int64)
        edge_features = np.zeros((E, EDGE_FEATURE_DIM), dtype=np.float32)
        for i, (u, v) in enumerate(edges):
            edge_index[0, i] = u
            edge_index[1, i] = v
            edge_features[i] = np.mean(np.stack(pair_feats[(u, v)]), axis=0)

        # 3) Per-node aggregates. Walk the original event list so the
        #    statistics reflect the *true* flow count, not the
        #    collapsed-edge count.
        in_deg = np.zeros(N, dtype=np.float32)
        out_deg = np.zeros(N, dtype=np.float32)
        in_peers: list[set[int]] = [set() for _ in range(N)]
        out_peers: list[set[int]] = [set() for _ in range(N)]
        out_ports: list[set[int]] = [set() for _ in range(N)]
        bytes_sent = np.zeros(N, dtype=np.float64)
        bytes_recv = np.zeros(N, dtype=np.float64)
        byte_rate_sum = np.zeros(N, dtype=np.float64)
        byte_rate_n = np.zeros(N, dtype=np.float64)
        dur_sum = np.zeros(N, dtype=np.float64)
        dur_n = np.zeros(N, dtype=np.float64)
        syn_like = np.zeros(N, dtype=np.float64)
        flow_n_out = np.zeros(N, dtype=np.float64)
        beacon_times: list[list[float]] = [[] for _ in range(N)]

        for e in events:
            u = node_index[e.src_ip]
            v = node_index[e.dst_ip]
            f = e.features
            fwd_bytes = float(f.get("fwd_bytes",
                              f.get("Total Length of Fwd Packets",
                              f.get("Fwd Packets Length Total", 0.0))) or 0.0)
            bwd_bytes = float(f.get("bwd_bytes",
                              f.get("Total Length of Bwd Packets",
                              f.get("Bwd Packets Length Total", 0.0))) or 0.0)
            total_bytes = fwd_bytes + bwd_bytes
            duration = float(f.get("duration", f.get("Flow Duration", 0.0)) or 0.0)
            byte_rate = float(f.get("byte_rate", f.get("Flow Bytes/s", 0.0)) or 0.0)
            dst_port = int(float(f.get("dst_port",
                                f.get("Destination Port", 0)) or 0))

            out_deg[u] += 1.0
            in_deg[v] += 1.0
            out_peers[u].add(v)
            in_peers[v].add(u)
            out_ports[u].add(dst_port)
            bytes_sent[u] += fwd_bytes
            bytes_recv[v] += bwd_bytes
            byte_rate_sum[u] += byte_rate
            byte_rate_n[u] += 1.0
            dur_sum[u] += duration
            dur_n[u] += 1.0
            flow_n_out[u] += 1.0
            beacon_times[u].append(e.ts)
            # SYN-like: short, low-byte flow with no response.
            if total_bytes < 200 and bwd_bytes == 0:
                syn_like[u] += 1.0

        node_features = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
        for i in range(N):
            mean_br = byte_rate_sum[i] / byte_rate_n[i] if byte_rate_n[i] > 0 else 0.0
            mean_dur = dur_sum[i] / dur_n[i] if dur_n[i] > 0 else 0.0
            syn_ratio = syn_like[i] / flow_n_out[i] if flow_n_out[i] > 0 else 0.0
            beacon = _beacon_score(beacon_times[i])
            node_features[i] = np.array([
                math.log1p(in_deg[i]),
                math.log1p(out_deg[i]),
                math.log1p(len(in_peers[i])),
                math.log1p(len(out_peers[i])),
                math.log1p(len(out_ports[i])),
                math.log1p(bytes_sent[i]),
                math.log1p(bytes_recv[i]),
                math.log1p(max(mean_br, 0.0)),
                math.log1p(max(mean_dur, 0.0)),
                syn_ratio,
                beacon,
            ], dtype=np.float32)

        hints = _topology_hints(
            node_ids=node_ids,
            in_deg=in_deg,
            out_deg=out_deg,
            unique_out_peers=[len(s) for s in out_peers],
            unique_out_ports=[len(s) for s in out_ports],
            syn_ratio=[
                (syn_like[i] / flow_n_out[i]) if flow_n_out[i] > 0 else 0.0
                for i in range(N)
            ],
            beacon_scores=[_beacon_score(beacon_times[i]) for i in range(N)],
        )

        return GraphSnapshot(
            node_ids=node_ids,
            node_index=node_index,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            window_seconds=self.window_seconds,
            flow_count=len(events),
            topology_hints=hints,
        )

    # --- PyG conversion ----------------------------------------------

    @staticmethod
    def snapshot_to_pyg(snapshot: GraphSnapshot):
        """Convert a snapshot to a PyTorch Geometric ``Data`` object.

        Imports torch lazily so the rest of the codebase can build and
        introspect graphs even on hosts without torch-geometric
        installed (e.g. the dashboard-only deployment).
        """
        import torch
        from torch_geometric.data import Data

        x = torch.from_numpy(snapshot.node_features)
        edge_index = torch.from_numpy(snapshot.edge_index).long()
        edge_attr = torch.from_numpy(snapshot.edge_features)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------
# Topology hint heuristics
# ---------------------------------------------------------------------
# These are intentionally simple — they exist so the system has *some*
# signal to surface before the GNN is trained, and so the GNN's outputs
# can be sanity-checked against them at inference time. The trained
# model is expected to subsume them with better precision.


def _beacon_score(timestamps: list[float]) -> float:
    """High when flows from this node are evenly spaced (C2 beacons)."""
    if len(timestamps) < 4:
        return 0.0
    ts = sorted(timestamps)
    gaps = np.diff(ts)
    if len(gaps) == 0:
        return 0.0
    mean = float(np.mean(gaps))
    std = float(np.std(gaps))
    if mean <= 0:
        return 0.0
    # Coefficient of variation, inverted and clipped to [0, 1].
    cv = std / mean
    return float(max(0.0, 1.0 - cv))


def _topology_hints(
    node_ids: list[str],
    in_deg: np.ndarray,
    out_deg: np.ndarray,
    unique_out_peers: list[int],
    unique_out_ports: list[int],
    syn_ratio: list[float],
    beacon_scores: list[float],
) -> dict[str, list[dict[str, Any]]]:
    """Return per-pattern node lists used by the dashboard threat view."""
    hints: dict[str, list[dict[str, Any]]] = {"ddos": [], "scan": [], "c2": []}
    if len(node_ids) == 0:
        return hints

    # Thresholds are intentionally lenient — we want recall on the
    # hint path; the model + ensemble combo gates final decisions.
    DDOS_IN_DEG = 20
    SCAN_OUT_DEG = 20
    SCAN_PORT_FANOUT = 10
    C2_PEERS = 5
    C2_BEACON = 0.7

    for i, ip in enumerate(node_ids):
        if in_deg[i] >= DDOS_IN_DEG:
            hints["ddos"].append({
                "ip": ip,
                "in_degree": int(in_deg[i]),
                "severity": float(min(1.0, in_deg[i] / 100.0)),
                "rationale": (
                    f"{int(in_deg[i])} distinct sources targeting this host "
                    f"in the window"
                ),
            })
        if out_deg[i] >= SCAN_OUT_DEG and (
            unique_out_ports[i] >= SCAN_PORT_FANOUT or syn_ratio[i] >= 0.5
        ):
            hints["scan"].append({
                "ip": ip,
                "out_degree": int(out_deg[i]),
                "unique_ports": int(unique_out_ports[i]),
                "syn_ratio": float(syn_ratio[i]),
                "severity": float(min(1.0, out_deg[i] / 100.0)),
                "rationale": (
                    f"fanout to {unique_out_peers[i]} peers across "
                    f"{unique_out_ports[i]} ports, {int(syn_ratio[i]*100)}% "
                    f"unanswered"
                ),
            })
        if (
            unique_out_peers[i] >= C2_PEERS
            and beacon_scores[i] >= C2_BEACON
        ):
            hints["c2"].append({
                "ip": ip,
                "peers": int(unique_out_peers[i]),
                "beacon_score": float(beacon_scores[i]),
                "severity": float(beacon_scores[i]),
                "rationale": (
                    f"{unique_out_peers[i]} peers contacted at regular "
                    f"intervals (beacon score {beacon_scores[i]:.2f})"
                ),
            })

    return hints
