"""
Model performance tracking for the AI-NGFW retraining loop.

Tracks the moving accuracy curve as the model is retrained on verified
honeypot captures: baseline (the number the model shipped with) vs.
current, plus a version history so the dashboard can render a chart
of how the feedback loop is moving the needle.

Persistence is a single JSON file under :data:`DEFAULT_METRICS_PATH`
(project ``data/model_metrics.json``). The path is overridable via the
``NGFW_MODEL_METRICS_PATH`` env var. Writes are atomic via tmp+rename
so a partial write can't corrupt the file.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = Path(
    os.environ.get(
        "NGFW_MODEL_METRICS_PATH",
        str(_PROJECT_ROOT / "data" / "model_metrics.json"),
    )
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state(baseline: float = 68.9) -> dict[str, Any]:
    return {
        "baseline_accuracy": baseline,
        "current_accuracy": baseline,
        "improvement": 0.0,
        "current_version": "v1.0",
        "verified_captures": 0,
        "false_positive_count": 0,
        "training_queue_size": 0,
        "last_retrained_at": None,
        "history": [
            {"date": "Day 1", "accuracy": baseline, "version": "v1.0"}
        ],
        "versions": [
            {
                "version": "v1.0",
                "date": "Initial",
                "accuracy": baseline,
                "samples": 0,
                "note": "Base model",
            }
        ],
    }


class ModelMetricsStore:
    """Thread-safe accuracy / version history store.

    Read-mostly: the dashboard polls ``snapshot()`` every few seconds.
    Writes happen only on ``record_retrain()`` and verification counters.
    """

    def __init__(self, path: Path | str = DEFAULT_METRICS_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._state = self._load_or_seed()

    # --- persistence --------------------------------------------------

    def _load_or_seed(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    state = json.load(fh)
                # Backfill any new keys that may not exist in older files.
                seed = _default_state(state.get("baseline_accuracy", 68.9))
                for k, v in seed.items():
                    state.setdefault(k, v)
                return state
            except (OSError, json.JSONDecodeError):
                pass
        # Seed file
        state = _default_state()
        self._write(state)
        return state

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(self.path)

    # --- queries ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Dashboard-facing payload — includes derived fields."""
        with self._lock:
            s = dict(self._state)
        return {
            "baseline_accuracy": s["baseline_accuracy"],
            "current_accuracy": s["current_accuracy"],
            "improvement": round(
                s["current_accuracy"] - s["baseline_accuracy"], 2
            ),
            "current_version": s["current_version"],
            "verified_captures": s["verified_captures"],
            "false_positive_count": s.get("false_positive_count", 0),
            "training_queue_size": s["training_queue_size"],
            "last_retrained_at": s["last_retrained_at"],
            "history": s["history"],
            "versions": s["versions"],
        }

    # --- mutators -----------------------------------------------------

    def set_baseline(self, accuracy: float) -> None:
        with self._lock:
            self._state["baseline_accuracy"] = float(accuracy)
            if self._state["current_accuracy"] == 0:
                self._state["current_accuracy"] = float(accuracy)
            self._write(self._state)

    def set_training_queue_size(self, n: int) -> None:
        with self._lock:
            self._state["training_queue_size"] = int(n)
            self._write(self._state)

    def increment_verified(self, is_real_attack: bool = True) -> dict[str, int]:
        with self._lock:
            if is_real_attack:
                self._state["verified_captures"] += 1
            else:
                self._state["false_positive_count"] = (
                    self._state.get("false_positive_count", 0) + 1
                )
            self._write(self._state)
            return {
                "verified_captures": self._state["verified_captures"],
                "false_positive_count": self._state.get("false_positive_count", 0),
            }

    def record_retrain(
        self,
        new_accuracy: float,
        samples_added: int,
        note: str | None = None,
        version: str | None = None,
        date_label: str | None = None,
    ) -> dict[str, Any]:
        """Snapshot a retrain event: rev the version, append history.

        Pass an explicit ``version`` and ``date_label`` to align history
        entries with the multi-iteration feedback-loop demo (e.g.
        ``version='v2.0', date_label='Round 1'``). Without overrides the
        next minor version is auto-assigned and the entry is labelled
        ``Day N``.
        """
        with self._lock:
            prev_version = self._state["current_version"]
            new_version = version or _bump_version(prev_version)
            day_n = len(self._state["history"]) + 1
            label = date_label or f"Day {day_n}"

            self._state["current_accuracy"] = float(new_accuracy)
            self._state["current_version"] = new_version
            self._state["last_retrained_at"] = _now_iso()
            # Verified captures consumed by this retrain → reset counter,
            # the training queue is also cleared by the caller.
            self._state["verified_captures"] = 0
            self._state["training_queue_size"] = 0

            self._state["history"].append({
                "date": label,
                "accuracy": float(new_accuracy),
                "version": new_version,
            })
            self._state["versions"].append({
                "version": new_version,
                "date": label,
                "timestamp": _now_iso(),
                "accuracy": float(new_accuracy),
                "samples": int(samples_added),
                "note": note or "Retrained on verified honeypot captures",
            })
            self._write(self._state)
            return {
                "previous_version": prev_version,
                "current_version": new_version,
                "current_accuracy": float(new_accuracy),
                "samples_added": int(samples_added),
                "improvement": round(
                    new_accuracy - self._state["baseline_accuracy"], 2
                ),
            }

    def reset_history(
        self,
        baseline_accuracy: float,
        baseline_version: str = "v1.0",
        baseline_label: str = "Baseline",
    ) -> dict[str, Any]:
        """Wipe history and re-seed with a fresh baseline entry.

        Used by the multi-iteration demo so re-running the script
        produces a clean version log instead of appending forever.
        """
        with self._lock:
            self._state = {
                "baseline_accuracy": float(baseline_accuracy),
                "current_accuracy": float(baseline_accuracy),
                "improvement": 0.0,
                "current_version": baseline_version,
                "verified_captures": 0,
                "false_positive_count": 0,
                "training_queue_size": 0,
                "last_retrained_at": None,
                "history": [
                    {
                        "date": baseline_label,
                        "accuracy": float(baseline_accuracy),
                        "version": baseline_version,
                    }
                ],
                "versions": [
                    {
                        "version": baseline_version,
                        "date": baseline_label,
                        "timestamp": _now_iso(),
                        "accuracy": float(baseline_accuracy),
                        "samples": 0,
                        "note": "Baseline model — pre-feedback-loop",
                    }
                ],
            }
            self._write(self._state)
            return dict(self._state)


def _bump_version(version: str) -> str:
    """v1.0 -> v1.1, v1.9 -> v1.10, vN.M -> vN.M+1."""
    try:
        major, minor = version.lstrip("v").split(".")
        return f"v{major}.{int(minor) + 1}"
    except ValueError:
        return f"{version}.1"


# Module singleton — the API and the demo script share this.
store = ModelMetricsStore()
