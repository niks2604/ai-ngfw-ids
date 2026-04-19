"""
Zero Trust policy layer.

Wraps the ensemble decision with explicit Zero Trust principles:
  - Never Trust, Always Verify  -> every flow is scored; no static allowlist
  - Assume Breach               -> default posture escalates on anomaly signals
  - Least Privilege             -> lowest-privilege action that still mitigates
  - Continuous Verification     -> re-evaluate via session_risk on repeat sources

This layer is deliberately deterministic and explainable — no ML here, only
policy rules on top of the risk_score + model_scores + optional flow context.
It is designed to be called after the ensemble but before any enforcement
action (block, redirect to honeypot, step-up auth, rate-limit, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustLevel(str, Enum):
    TRUSTED = "TRUSTED"           # score < 0.30 and no red flags
    LOW_RISK = "LOW_RISK"         # 0.30 <= score < 0.50
    ELEVATED = "ELEVATED"         # 0.50 <= score < 0.70
    HIGH_RISK = "HIGH_RISK"       # 0.70 <= score < 0.90
    CRITICAL = "CRITICAL"         # score >= 0.90 or multiple red flags


class Action(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"                   # allow but log with higher verbosity
    STEP_UP_AUTH = "STEP_UP_AUTH"         # require additional verification
    RATE_LIMIT = "RATE_LIMIT"
    INSPECT = "INSPECT"                   # DPI / sandbox
    REDIRECT_HONEYPOT = "REDIRECT_HONEYPOT"
    BLOCK = "BLOCK"
    QUARANTINE = "QUARANTINE"             # block + flag source for review


@dataclass
class FlowContext:
    """Optional context to sharpen Zero Trust evaluation.

    None of these are required; the layer degrades to score-only policy
    when context is absent.
    """
    src_ip: str | None = None
    dst_port: int | None = None
    is_internal_src: bool = False
    is_authenticated: bool = False
    prior_violations: int = 0         # count from session store / SIEM
    session_risk: float = 0.0         # rolling risk from prior flows, 0-1
    asset_sensitivity: str = "normal"  # "low" | "normal" | "high" | "crown_jewel"


@dataclass
class ZeroTrustDecision:
    trust_level: TrustLevel
    primary_action: Action
    secondary_actions: list[Action] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    principles_applied: list[str] = field(default_factory=list)
    effective_risk: float = 0.0       # risk after context adjustments

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_level": self.trust_level.value,
            "primary_action": self.primary_action.value,
            "secondary_actions": [a.value for a in self.secondary_actions],
            "recommendations": self.recommendations,
            "principles_applied": self.principles_applied,
            "effective_risk": round(self.effective_risk, 4),
        }


# Thresholds align with API decision thresholds (0.3 / 0.7) with extra
# granularity inside Zero Trust for finer-grained response.
_TRUST_BANDS = [
    (0.30, TrustLevel.TRUSTED),
    (0.50, TrustLevel.LOW_RISK),
    (0.70, TrustLevel.ELEVATED),
    (0.90, TrustLevel.HIGH_RISK),
    (1.01, TrustLevel.CRITICAL),
]


def _band(score: float) -> TrustLevel:
    for cutoff, level in _TRUST_BANDS:
        if score < cutoff:
            return level
    return TrustLevel.CRITICAL


class ZeroTrustEngine:
    """Stateless policy engine. Safe to share across requests."""

    def evaluate(
        self,
        risk_score: float,
        model_scores: dict[str, float] | None = None,
        context: FlowContext | None = None,
    ) -> ZeroTrustDecision:
        ctx = context or FlowContext()
        model_scores = model_scores or {}

        effective_risk, risk_adjustments = self._adjust_risk(risk_score, model_scores, ctx)
        trust = _band(effective_risk)

        primary, secondary = self._choose_actions(trust, ctx)
        recs = self._recommendations(trust, ctx, model_scores)
        principles = ["Never Trust, Always Verify"]

        if ctx.session_risk > 0 or ctx.prior_violations > 0:
            principles.append("Continuous Verification")
        if trust in (TrustLevel.HIGH_RISK, TrustLevel.CRITICAL):
            principles.append("Assume Breach")
        if primary in (Action.RATE_LIMIT, Action.STEP_UP_AUTH, Action.MONITOR):
            principles.append("Least Privilege")

        return ZeroTrustDecision(
            trust_level=trust,
            primary_action=primary,
            secondary_actions=secondary,
            recommendations=recs + risk_adjustments,
            principles_applied=principles,
            effective_risk=effective_risk,
        )

    # --- adjustments ------------------------------------------------------

    def _adjust_risk(
        self,
        base_risk: float,
        model_scores: dict[str, float],
        ctx: FlowContext,
    ) -> tuple[float, list[str]]:
        risk = base_risk
        notes: list[str] = []

        # Anomaly signal: IsolationForest high even if classifiers disagree.
        iso = model_scores.get("isolation_forest", 0.0)
        if iso >= 0.8 and base_risk < 0.7:
            risk = min(1.0, risk + 0.10)
            notes.append("Anomaly detector flagged unusual flow; raised risk +0.10")

        # Prior violations from this source -> escalate.
        if ctx.prior_violations >= 3:
            risk = min(1.0, risk + 0.15)
            notes.append(f"Source has {ctx.prior_violations} prior violations; raised risk +0.15")

        # Rolling session risk blends in.
        if ctx.session_risk > 0:
            risk = min(1.0, 0.7 * risk + 0.3 * ctx.session_risk)
            notes.append(f"Blended with session risk {ctx.session_risk:.2f}")

        # Crown-jewel assets demand higher scrutiny.
        if ctx.asset_sensitivity == "crown_jewel" and risk < 0.9:
            risk = min(1.0, risk + 0.10)
            notes.append("Destination is crown-jewel asset; raised risk +0.10")

        # Unauthenticated access bumps risk slightly (Never Trust).
        if not ctx.is_authenticated and not ctx.is_internal_src:
            risk = min(1.0, risk + 0.05)
            notes.append("Unauthenticated external source; raised risk +0.05")

        return risk, notes

    # --- action selection -------------------------------------------------

    def _choose_actions(
        self,
        trust: TrustLevel,
        ctx: FlowContext,
    ) -> tuple[Action, list[Action]]:
        if trust is TrustLevel.TRUSTED:
            return Action.ALLOW, [Action.MONITOR] if not ctx.is_authenticated else []

        if trust is TrustLevel.LOW_RISK:
            secondary = [Action.MONITOR]
            if not ctx.is_authenticated:
                secondary.append(Action.STEP_UP_AUTH)
            return Action.ALLOW, secondary

        if trust is TrustLevel.ELEVATED:
            return Action.INSPECT, [Action.RATE_LIMIT, Action.MONITOR]

        if trust is TrustLevel.HIGH_RISK:
            return Action.BLOCK, [Action.REDIRECT_HONEYPOT]

        # CRITICAL
        return Action.QUARANTINE, [Action.BLOCK, Action.REDIRECT_HONEYPOT]

    # --- recommendations --------------------------------------------------

    def _recommendations(
        self,
        trust: TrustLevel,
        ctx: FlowContext,
        model_scores: dict[str, float],
    ) -> list[str]:
        recs: list[str] = []

        if trust is TrustLevel.TRUSTED:
            recs.append("Permit flow; continue passive monitoring.")
            return recs

        if trust is TrustLevel.LOW_RISK:
            recs.append("Permit flow but increase log verbosity for this source.")
            if not ctx.is_authenticated:
                recs.append("Require step-up authentication before sensitive actions.")
            return recs

        if trust is TrustLevel.ELEVATED:
            recs.append("Route flow through deep packet inspection / sandbox.")
            recs.append("Apply rate limiting on source to contain blast radius.")
            if ctx.asset_sensitivity in ("high", "crown_jewel"):
                recs.append("Notify SOC; destination asset is high-value.")
            return recs

        if trust is TrustLevel.HIGH_RISK:
            recs.append("Block flow and redirect source to honeypot for intel capture.")
            recs.append("Open a SIEM incident and correlate with prior events from this source.")
            return recs

        # CRITICAL
        recs.append("Immediately quarantine the source; block all traffic from this IP.")
        recs.append("Redirect to honeypot to capture attacker TTPs.")
        recs.append("Page on-call; possible active intrusion.")
        if ctx.asset_sensitivity == "crown_jewel":
            recs.append("Initiate breach-response playbook — crown-jewel asset targeted.")
        return recs


# Module-level default instance for convenience.
engine = ZeroTrustEngine()


def evaluate(
    risk_score: float,
    model_scores: dict[str, float] | None = None,
    context: FlowContext | None = None,
) -> ZeroTrustDecision:
    """Convenience wrapper around the default ZeroTrustEngine."""
    return engine.evaluate(risk_score, model_scores, context)
