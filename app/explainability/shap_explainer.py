"""
SHAP-based explainability for the ensemble detector.

TreeExplainer is used for Random Forest and XGBoost. Their attack-class SHAP
values are combined using the same weights as the ensemble (RF 0.50, XGB 0.25,
renormalised because IsolationForest is not tree-SHAP-compatible) to produce
per-feature contributions. IsolationForest is not explained here — its
contribution to the risk score is acknowledged in the explanation text instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


# Human-readable tags for features whose raw names are opaque.
# Falls back to the raw column name when no entry matches.
_FRIENDLY: dict[str, str] = {
    "Flow Duration": "Flow Duration",
    "Total Fwd Packets": "Forward Packet Count",
    "Total Backward Packets": "Backward Packet Count",
    "Fwd Packets Length Total": "Forward Bytes",
    "Bwd Packets Length Total": "Backward Bytes",
    "Flow Bytes/s": "Flow Byte Rate",
    "Flow Packets/s": "Flow Packet Rate",
    "Flow IAT Mean": "Inter-Arrival Time (mean)",
    "Fwd IAT Mean": "Forward IAT (mean)",
    "Bwd IAT Mean": "Backward IAT (mean)",
    "Destination Port": "Destination Port",
    "Protocol": "Protocol",
    "SYN Flag Count": "SYN Flag Count",
    "ACK Flag Count": "ACK Flag Count",
    "FIN Flag Count": "FIN Flag Count",
    "PSH Flag Count": "PSH Flag Count",
    "URG Flag Count": "URG Flag Count",
    "RST Flag Count": "RST Flag Count",
    "Packet Length Mean": "Packet Length (mean)",
    "Packet Length Std": "Packet Length (std)",
    "Packet Length Variance": "Packet Length Variance",
    "Avg Packet Size": "Average Packet Size",
    "Init Fwd Win Bytes": "TCP Initial Window (fwd)",
    "Init Bwd Win Bytes": "TCP Initial Window (bwd)",
}


def _friendly(name: str) -> str:
    return _FRIENDLY.get(name, name)


def _attack_shap(values: Any) -> np.ndarray:
    """Normalise shap_values output to a 2-D array for the attack class.

    TreeExplainer returns:
      - list of arrays [class_0, class_1] for multi-output sklearn models
      - a single array for XGBoost binary classifiers
      - an Explanation object (newer shap); caller converts via .values first
    """
    if isinstance(values, list):
        return np.asarray(values[1])
    arr = np.asarray(values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    return arr


class EnsembleShapExplainer:
    """Lazy-initialised tree-SHAP explainer wrapping RF + XGBoost."""

    # Weights *among explainable models only* (RF + XGB), renormalised
    # from the ensemble's (0.50, 0.25) -> (0.667, 0.333).
    _TREE_WEIGHTS = {"random_forest": 2 / 3, "xgboost": 1 / 3}

    def __init__(self, ensemble, feature_columns: list[str]):
        if not ensemble.is_loaded:
            ensemble.load_models()
        self.ensemble = ensemble
        self.feature_columns = feature_columns
        self._explainers: dict[str, shap.TreeExplainer] = {
            "random_forest": shap.TreeExplainer(ensemble.models["random_forest"]),
            "xgboost": shap.TreeExplainer(ensemble.models["xgboost"]),
        }

    def _combined_shap(self, X_scaled: np.ndarray) -> np.ndarray:
        total = np.zeros_like(X_scaled, dtype=float)
        for name, weight in self._TREE_WEIGHTS.items():
            sv = self._explainers[name].shap_values(X_scaled)
            total += weight * _attack_shap(sv)
        return total

    def explain(
        self,
        df: pd.DataFrame,
        decision: str,
        top_n: int = 5,
    ) -> dict:
        """Explain the first row of `df`.

        Returns
        -------
        {
          "top_features": [{"feature","value","shap_value","direction"}, ...],
          "text": "BLOCK because: High Packet Rate (+0.32), ..."
        }
        """
        X = df[self.feature_columns].astype(float).values
        X_scaled = self.ensemble.scaler.transform(X)
        shap_vals = self._combined_shap(X_scaled)[0]  # first (and only) row

        # Rank by |shap value| descending.
        order = np.argsort(-np.abs(shap_vals))[:top_n]
        raw_values = df.iloc[0][self.feature_columns].values

        top_features = [
            {
                "feature": _friendly(self.feature_columns[i]),
                "value": float(raw_values[i]),
                "shap_value": float(shap_vals[i]),
                "direction": "+" if shap_vals[i] >= 0 else "-",
            }
            for i in order
        ]

        text = _format_explanation(decision, top_features)
        return {"top_features": top_features, "text": text}


def _format_explanation(decision: str, top_features: list[dict]) -> str:
    verb = {
        "BLOCK": "BLOCKED",
        "INSPECT": "FLAGGED for inspection",
        "ALLOW": "ALLOWED",
    }.get(decision, decision)

    parts = [
        f"{f['feature']} ({f['direction']}{abs(f['shap_value']):.2f})"
        for f in top_features
    ]
    if not parts:
        return f"{verb}."

    if decision == "ALLOW":
        return f"{verb} — top factors: " + ", ".join(parts)
    return f"{verb} because: " + ", ".join(parts)


# --- Module-level accessor used by the API ---------------------------------

_explainer: EnsembleShapExplainer | None = None


def get_explainer(ensemble, feature_columns: list[str]) -> EnsembleShapExplainer:
    """Return the singleton explainer, building it on first call."""
    global _explainer
    if _explainer is None:
        _explainer = EnsembleShapExplainer(ensemble, feature_columns)
    return _explainer
