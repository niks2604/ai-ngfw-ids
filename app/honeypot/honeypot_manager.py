"""
Honeypot manager — capture skeleton for blocked traffic.

Scope
-----
Full Cowrie / T-Pot integration is future work. This module provides:

  1. A structured event store for traffic the pipeline decides to redirect
     (BLOCK / QUARANTINE / REDIRECT_HONEYPOT from the Zero Trust layer).
  2. A pluggable backend interface so Cowrie, Dionaea, or a custom sink can
     be dropped in later without changing callers.
  3. Aggregation helpers (top source IPs, attack-type breakdown) for the
     paper's analysis section.

Events are persisted as line-delimited JSON under `logs/honeypot/` to keep the
write path crash-safe (no mid-write corruption of a big JSON blob) and easy to
tail with standard tools.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LOG_DIR = Path(
    os.environ.get("NGFW_HONEYPOT_LOG_DIR", os.path.expanduser("~/sem6el/logs/honeypot"))
)


@dataclass
class HoneypotEvent:
    """A single redirected / captured flow."""
    timestamp: str
    src_ip: str | None
    dst_ip: str | None
    dst_port: int | None
    protocol: str | None
    decision: str                          # BLOCK | QUARANTINE | REDIRECT_HONEYPOT
    risk_score: float
    attack_type: str | None = None         # predicted class if available
    model_scores: dict[str, float] = field(default_factory=dict)
    zero_trust: dict[str, Any] = field(default_factory=dict)
    flow_features: dict[str, float] = field(default_factory=dict)
    notes: str | None = None

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class HoneypotBackend:
    """Interface for honeypot backends (Cowrie, Dionaea, custom TCP sink).

    The default implementation is a no-op — it only records the event. A real
    backend should override `redirect()` to hand the flow off to a honeypot
    listener and return connection metadata for the SIEM.
    """

    name: str = "noop"

    def redirect(self, event: HoneypotEvent) -> dict[str, Any]:
        return {"backend": self.name, "redirected": False, "reason": "stub"}

    def status(self) -> dict[str, Any]:
        return {"backend": self.name, "available": True}


class CowriePlaceholder(HoneypotBackend):
    """Marker for future Cowrie SSH/Telnet honeypot integration.

    Intentionally does not attempt a socket connection — this is a scaffold
    so callers can depend on the type today and swap the implementation in
    when Cowrie is deployed.
    """
    name: str = "cowrie"

    def redirect(self, event: HoneypotEvent) -> dict[str, Any]:
        return {
            "backend": self.name,
            "redirected": False,
            "reason": "cowrie integration pending",
            "target": "cowrie://localhost:2222",
        }


class HoneypotManager:
    """Thread-safe manager for redirected traffic.

    Usage
    -----
    >>> mgr = HoneypotManager()
    >>> mgr.capture(src_ip="1.2.3.4", dst_port=22, decision="BLOCK",
    ...             risk_score=0.93, attack_type="Bruteforce")
    """

    def __init__(
        self,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        backend: HoneypotBackend | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend or HoneypotBackend()
        self._lock = threading.Lock()
        self._events: list[HoneypotEvent] = []

    # --- capture -------------------------------------------------------

    def capture(
        self,
        *,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        dst_port: int | None = None,
        protocol: str | None = None,
        decision: str = "BLOCK",
        risk_score: float = 0.0,
        attack_type: str | None = None,
        model_scores: dict[str, float] | None = None,
        zero_trust: dict[str, Any] | None = None,
        flow_features: dict[str, float] | None = None,
        notes: str | None = None,
    ) -> HoneypotEvent:
        event = HoneypotEvent(
            timestamp=HoneypotEvent.now_iso(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            decision=decision,
            risk_score=float(risk_score),
            attack_type=attack_type,
            model_scores=model_scores or {},
            zero_trust=zero_trust or {},
            flow_features=flow_features or {},
            notes=notes,
        )

        # Hand off to backend (no-op by default).
        backend_result = self.backend.redirect(event)
        event.notes = (event.notes or "") + f" | backend={backend_result}"

        with self._lock:
            self._events.append(event)
            self._append_jsonl(event)

        return event

    def _append_jsonl(self, event: HoneypotEvent) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.log_dir / f"events-{day}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    # --- queries / aggregation ----------------------------------------

    def events(self) -> list[HoneypotEvent]:
        with self._lock:
            return list(self._events)

    def load_from_disk(self) -> int:
        """Re-hydrate in-memory events from today's JSONL file."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.log_dir / f"events-{day}.jsonl"
        if not path.exists():
            return 0
        loaded = 0
        with self._lock, path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._events.append(HoneypotEvent(**data))
                loaded += 1
        return loaded

    def top_sources(self, n: int = 10) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for e in self.events():
            if e.src_ip:
                counts[e.src_ip] = counts.get(e.src_ip, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def attack_type_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events():
            key = e.attack_type or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def stats(self) -> dict[str, Any]:
        evts = self.events()
        return {
            "total_captured": len(evts),
            "unique_sources": len({e.src_ip for e in evts if e.src_ip}),
            "top_sources": self.top_sources(5),
            "attack_types": self.attack_type_breakdown(),
            "backend": self.backend.status(),
            "log_dir": str(self.log_dir),
        }


# Module-level singleton for the API to share.
manager = HoneypotManager()
