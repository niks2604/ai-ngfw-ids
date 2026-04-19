"""
NetFlow v5 collector — ingestion skeleton.

Scope
-----
Full production ingestion (v9/IPFIX, template caching, flow deduplication,
backpressure, rotation) is future work. This module provides:

  1. A NetFlow v5 packet parser (header + 30-byte records).
  2. A UDP listener that decodes packets and pushes `NetFlowRecord` objects
     onto a queue for downstream ML scoring.
  3. A `feed_to_ensemble` hook so the ingestion pipeline can close the loop
     with the inference service.

NetFlow v5 is chosen because it is the smallest well-defined format for the
paper's dataflow diagram. Real deployments should prefer v9/IPFIX.
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
from dataclasses import dataclass, asdict
from typing import Callable, Iterator


# ---- NetFlow v5 packet format --------------------------------------------
# Header: 24 bytes
# Record: 48 bytes  (NOT 30 — RFC / Cisco spec)
# See https://www.cisco.com/c/en/us/td/docs/net_mgmt/netflow_collection_engine/3-6/user/guide/format.html

_HEADER_FMT = "!HHIIIIBBH"            # 24 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_RECORD_FMT = "!IIIHHIIIIHHBBBBHHBBH"  # 48 bytes
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)


@dataclass
class NetFlowHeader:
    version: int
    count: int
    sys_uptime: int
    unix_secs: int
    unix_nsecs: int
    flow_sequence: int
    engine_type: int
    engine_id: int
    sampling_interval: int


@dataclass
class NetFlowRecord:
    """A single NetFlow v5 flow record (one direction)."""
    src_addr: str
    dst_addr: str
    next_hop: str
    input_iface: int
    output_iface: int
    packets: int
    bytes: int
    first_ms: int         # sys_uptime ms at flow start
    last_ms: int
    src_port: int
    dst_port: int
    tcp_flags: int
    protocol: int
    tos: int
    src_as: int
    dst_as: int
    src_mask: int
    dst_mask: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.last_ms - self.first_ms)

    def to_flow_features(self) -> dict[str, float]:
        """Translate a raw NetFlow record into a minimal CICIDS-style feature dict.

        This is intentionally partial — the ML model expects 51 engineered
        features; NetFlow v5 only supplies ~10. Callers should enrich with
        bidirectional aggregation before sending to `/predict`.
        """
        dur_s = self.duration_ms / 1000.0 or 1e-6
        return {
            "Destination Port": float(self.dst_port),
            "Protocol": float(self.protocol),
            "Flow Duration": float(self.duration_ms * 1000),  # to microseconds
            "Total Fwd Packets": float(self.packets),
            "Fwd Packets Length Total": float(self.bytes),
            "Flow Bytes/s": float(self.bytes) / dur_s,
            "Flow Packets/s": float(self.packets) / dur_s,
            # Flag bits: bit 0=FIN, 1=SYN, 2=RST, 3=PSH, 4=ACK, 5=URG
            "FIN Flag Count": float((self.tcp_flags >> 0) & 1),
            "SYN Flag Count": float((self.tcp_flags >> 1) & 1),
            "RST Flag Count": float((self.tcp_flags >> 2) & 1),
            "PSH Flag Count": float((self.tcp_flags >> 3) & 1),
            "ACK Flag Count": float((self.tcp_flags >> 4) & 1),
            "URG Flag Count": float((self.tcp_flags >> 5) & 1),
        }


# ---- Parsing --------------------------------------------------------------


def _ip(n: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", n))


def parse_packet(data: bytes) -> tuple[NetFlowHeader, list[NetFlowRecord]]:
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"packet too small: {len(data)} bytes")

    (version, count, uptime, secs, nsecs, seq, eng_t, eng_id, sampling) = (
        struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    )
    if version != 5:
        raise ValueError(f"expected NetFlow v5, got v{version}")

    header = NetFlowHeader(
        version=version, count=count, sys_uptime=uptime, unix_secs=secs,
        unix_nsecs=nsecs, flow_sequence=seq, engine_type=eng_t,
        engine_id=eng_id, sampling_interval=sampling,
    )

    records: list[NetFlowRecord] = []
    for i in range(count):
        start = _HEADER_SIZE + i * _RECORD_SIZE
        end = start + _RECORD_SIZE
        if end > len(data):
            break
        fields = struct.unpack(_RECORD_FMT, data[start:end])
        (src, dst, nh, in_i, out_i, pkts, bts,
         first, last, sp, dp, _pad, flags, proto, tos,
         sas, das, smask, dmask, _pad2) = fields
        records.append(NetFlowRecord(
            src_addr=_ip(src), dst_addr=_ip(dst), next_hop=_ip(nh),
            input_iface=in_i, output_iface=out_i,
            packets=pkts, bytes=bts, first_ms=first, last_ms=last,
            src_port=sp, dst_port=dp, tcp_flags=flags, protocol=proto,
            tos=tos, src_as=sas, dst_as=das, src_mask=smask, dst_mask=dmask,
        ))
    return header, records


# ---- UDP collector --------------------------------------------------------


class NetFlowCollector:
    """Threaded UDP listener that parses NetFlow v5 and queues records.

    Usage
    -----
    >>> col = NetFlowCollector(port=2055)
    >>> col.start()
    >>> for rec in col.stream():  # blocks for records
    ...     features = rec.to_flow_features()
    ...     # send `features` to /predict
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 2055,
        queue_max: int = 10_000,
    ):
        self.host = host
        self.port = port
        self.queue: queue.Queue[NetFlowRecord] = queue.Queue(maxsize=queue_max)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = {"packets": 0, "records": 0, "parse_errors": 0, "dropped": 0}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(1.0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
        self._thread = None
        self._sock = None

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            self.stats["packets"] += 1
            try:
                _hdr, records = parse_packet(data)
            except ValueError:
                self.stats["parse_errors"] += 1
                continue
            for rec in records:
                self.stats["records"] += 1
                try:
                    self.queue.put_nowait(rec)
                except queue.Full:
                    self.stats["dropped"] += 1

    def stream(self, timeout: float | None = None) -> Iterator[NetFlowRecord]:
        """Yield records as they arrive. Blocks on the internal queue."""
        while True:
            try:
                yield self.queue.get(timeout=timeout)
            except queue.Empty:
                return


# ---- ML integration hook --------------------------------------------------


def feed_to_ensemble(
    collector: NetFlowCollector,
    predict_fn: Callable[[dict[str, float]], dict],
    on_decision: Callable[[NetFlowRecord, dict], None] | None = None,
) -> None:
    """Blocking loop: read records, enrich, score, dispatch.

    `predict_fn` should accept a feature dict and return the API's prediction
    dict (decision, risk_score, ...). `on_decision` receives (record, result)
    and is where honeypot redirect / logging is wired.

    Full implementation (bidirectional aggregation, 5-tuple keying, flow
    timeouts) is future work — this is a straight 1:1 mapping for now.
    """
    for record in collector.stream():
        features = record.to_flow_features()
        result = predict_fn(features)
        if on_decision:
            on_decision(record, result)


__all__ = [
    "NetFlowHeader",
    "NetFlowRecord",
    "NetFlowCollector",
    "parse_packet",
    "feed_to_ensemble",
]
