"""
Evaluate the trained ensemble (Random Forest, XGBoost, Isolation Forest)
on the balanced test split and write a single JSON summary.

Decision conventions
--------------------
- The supervised models are binary (Benign=0, Attack=1).
- IsolationForest is an unsupervised one-class model; its raw output is
  mapped to 1 when the model flags an anomaly (predict() == -1) and the
  decision_function value is converted to a 0..1 risk score with the
  same sigmoid-around-offset transform the production ensemble uses, so
  ROC-AUC is comparable across models.
- The ensemble uses EnsembleDetector.predict() which returns 0/1/2
  (ALLOW/INSPECT/BLOCK). For the binary comparison we collapse
  {INSPECT, BLOCK} → attack and ALLOW → benign.

Run
---
    venv/bin/python -m training.evaluate_models
or:
    venv/bin/python training/evaluate_models.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow running as a script from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.ensemble import EnsembleDetector  # noqa: E402


MODELS_PATH = os.path.expanduser("~/sem6el/trained_models")
SPLITS_PATH = os.path.expanduser("~/sem6el/data/processed/balanced_splits.joblib")
RESULTS_PATH = _PROJECT_ROOT / "evaluation_results.json"

CLASS_NAMES = ["Benign", "Attack"]


# --- Metric helpers --------------------------------------------------------


def _confusion_rates(cm: np.ndarray) -> dict[str, float]:
    """Binary confusion matrix → TPR/FPR/FNR/TNR.

    Expects sklearn convention: cm[i,j] is true=i predicted=j with labels
    sorted ascending → [[TN, FP], [FN, TP]] for {0,1}.
    """
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "true_negative_rate": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def _metrics_block(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    inference_seconds: float,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "rates": _confusion_rates(cm),
        "classification_report": report,
        "n_samples": int(len(y_true)),
        "inference_seconds_total": float(inference_seconds),
        "inference_us_per_sample": float(inference_seconds / len(y_true) * 1_000_000.0),
        "throughput_samples_per_sec": float(len(y_true) / inference_seconds)
        if inference_seconds > 0
        else None,
    }


def _print_block(m: dict) -> None:
    print("\n" + "=" * 78)
    print(f"📊  {m['model']}")
    print("=" * 78)
    print(f"  Samples evaluated : {m['n_samples']:,}")
    print(f"  Inference total   : {m['inference_seconds_total']:.3f} s")
    print(f"  Inference / sample: {m['inference_us_per_sample']:.2f} µs")
    if m["throughput_samples_per_sec"]:
        print(f"  Throughput        : {m['throughput_samples_per_sec']:,.0f} samples/s")
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Precision         : {m['precision']:.4f}")
    print(f"  Recall            : {m['recall']:.4f}")
    print(f"  F1-Score          : {m['f1_score']:.4f}")
    if m["roc_auc"] is not None:
        print(f"  ROC-AUC           : {m['roc_auc']:.4f}")
    r = m["rates"]
    print(f"  False Positive Rate: {r['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {r['false_negative_rate']:.4f}")
    print("  Confusion matrix   :  (rows=true, cols=predicted)")
    print("                            Benign      Attack")
    print(f"    Benign (true)         {r['true_negative']:>8d}  {r['false_positive']:>10d}")
    print(f"    Attack (true)         {r['false_negative']:>8d}  {r['true_positive']:>10d}")
    print("\n  Per-class breakdown:")
    rep = m["classification_report"]
    print(f"    {'class':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    for cls in CLASS_NAMES + ["macro avg", "weighted avg"]:
        row = rep.get(cls)
        if not row:
            continue
        print(
            f"    {cls:<12} "
            f"{row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1-score']:>10.4f} {int(row['support']):>10d}"
        )


# --- Sigmoid risk score for Isolation Forest -----------------------------


def _iso_risk_score(iso_model, X_scaled: np.ndarray) -> np.ndarray:
    raw = iso_model.score_samples(X_scaled)
    anom = iso_model.offset_ - raw  # positive => more anomalous
    return 1.0 / (1.0 + np.exp(-anom * 10.0))


# --- Main ------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("AI-NGFW/IDS — Model Evaluation")
    print("=" * 78)
    print(f"Models path : {MODELS_PATH}")
    print(f"Splits path : {SPLITS_PATH}")

    splits = joblib.load(SPLITS_PATH)
    X_test = splits["X_test"]
    X_test_scaled = splits["X_test_scaled"]
    y_test = splits["y_test_binary"].astype(int)

    print(f"\nTest samples : {len(y_test):,}")
    counts = np.bincount(y_test, minlength=2)
    print(f"  Benign     : {counts[0]:,}")
    print(f"  Attack     : {counts[1]:,}")

    rf = joblib.load(os.path.join(MODELS_PATH, "random_forest.joblib"))
    xgb = joblib.load(os.path.join(MODELS_PATH, "xgboost.joblib"))
    iso = joblib.load(os.path.join(MODELS_PATH, "isolation_forest.joblib"))
    ensemble = EnsembleDetector(models_path=MODELS_PATH)
    ensemble.load_models()

    all_metrics: dict[str, dict] = {}

    # Random Forest --------------------------------------------------------
    t0 = time.perf_counter()
    rf_pred = rf.predict(X_test_scaled)
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    rf_dt = time.perf_counter() - t0
    all_metrics["random_forest"] = _metrics_block(
        "Random Forest", y_test, rf_pred, rf_proba, rf_dt
    )

    # XGBoost --------------------------------------------------------------
    t0 = time.perf_counter()
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_dt = time.perf_counter() - t0
    all_metrics["xgboost"] = _metrics_block(
        "XGBoost", y_test, xgb_pred, xgb_proba, xgb_dt
    )

    # Isolation Forest -----------------------------------------------------
    t0 = time.perf_counter()
    iso_raw_pred = iso.predict(X_test_scaled)        # 1 = inlier, -1 = outlier
    iso_pred = np.where(iso_raw_pred == -1, 1, 0)
    iso_score = _iso_risk_score(iso, X_test_scaled)
    iso_dt = time.perf_counter() - t0
    all_metrics["isolation_forest"] = _metrics_block(
        "Isolation Forest", y_test, iso_pred, iso_score, iso_dt
    )

    # Ensemble -------------------------------------------------------------
    # EnsembleDetector.scaler is applied internally — pass raw X_test.
    t0 = time.perf_counter()
    ensemble_score = ensemble.predict_ensemble(X_test)
    ensemble_decision = ensemble.predict(X_test)     # 0=ALLOW, 1=INSPECT, 2=BLOCK
    ensemble_dt = time.perf_counter() - t0
    ensemble_pred = (ensemble_decision >= 1).astype(int)   # INSPECT or BLOCK = attack
    ens_block = _metrics_block(
        "Ensemble (RF + XGB + IF, weighted)",
        y_test,
        ensemble_pred,
        ensemble_score,
        ensemble_dt,
    )
    # Add decision distribution for transparency.
    counts = np.bincount(ensemble_decision, minlength=3)
    ens_block["decision_distribution"] = {
        "ALLOW": int(counts[0]),
        "INSPECT": int(counts[1]),
        "BLOCK": int(counts[2]),
    }
    all_metrics["ensemble"] = ens_block

    # Print to stdout ------------------------------------------------------
    for key in ("random_forest", "xgboost", "isolation_forest", "ensemble"):
        _print_block(all_metrics[key])

    # Cross-model summary
    print("\n" + "=" * 78)
    print("📌  Summary")
    print("=" * 78)
    print(f"{'model':<40} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'auc':>7} {'µs/sample':>11}")
    for key in ("random_forest", "xgboost", "isolation_forest", "ensemble"):
        m = all_metrics[key]
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a "
        print(
            f"{m['model']:<40} "
            f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {auc:>7} "
            f"{m['inference_us_per_sample']:>10.2f}"
        )

    # Persist --------------------------------------------------------------
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_path": MODELS_PATH,
        "splits_path": SPLITS_PATH,
        "test_set": {
            "n_samples": int(len(y_test)),
            "n_benign": int(counts[0] if False else int(np.sum(y_test == 0))),
            "n_attack": int(np.sum(y_test == 1)),
        },
        "metrics": all_metrics,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n💾  Saved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
