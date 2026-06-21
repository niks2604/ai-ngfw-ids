"""
Cross-dataset evaluation: run the CICIDS-2017-trained ensemble on
CICIDS-2018.

Pipeline
--------
1. Load every parquet in ``~/sem6el/data/cicids2018/``. The dhoogla
   mirror ships one file per (attack class, day), already cleaned.
2. Reconcile the schema with the 51-feature CICIDS-2017 training
   schema. CICIDS-2018 uses CICFlowMeter v3 which renamed a handful
   of columns — we map them with :data:`COLUMN_ALIASES`. Anything
   still missing is filled with 0.0 so the matrix shape lines up.
3. Apply the **same scaler** that was fit on CICIDS-2017
   (``trained_models/scaler.joblib``). We deliberately do NOT refit
   — that would defeat the point of a cross-dataset test.
4. Score with Random Forest, XGBoost, Isolation Forest, and the
   weighted ensemble. Report accuracy / precision / recall / f1 /
   ROC-AUC + confusion matrix + per-class breakdown.
5. Diff every metric against ``evaluation_results.json`` (the in-dist
   CICIDS-2017 baseline) and persist everything to
   ``training/cicids2018_results.json``.

This is identical in spirit to ``training/evaluate_models.py`` — same
metric helpers, same JSON shape — but reads from the 2018 directory
instead of the in-distribution test split.
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
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.ensemble import EnsembleDetector  # noqa: E402


MODELS_PATH = os.path.expanduser("~/sem6el/trained_models")
DATA_DIR = os.path.expanduser("~/sem6el/data/cicids2018")
SCALER_PATH = os.path.join(MODELS_PATH, "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(MODELS_PATH, "feature_columns.joblib")

RESULTS_PATH = _PROJECT_ROOT / "training" / "cicids2018_results.json"
BASELINE_PATH = _PROJECT_ROOT / "evaluation_results.json"   # CICIDS-2017 results

CLASS_NAMES = ["Benign", "Attack"]

# CICFlowMeter v2 (used by CICIDS-2017) → v3 (used by CICIDS-2018).
# Some 2018 mirrors keep the old names; we map both directions so this
# works regardless of which release the parquet came from. Keys are
# names that may appear in the 2018 parquet; values are the 2017 names
# we trained on.
COLUMN_ALIASES: dict[str, str] = {
    # Total bytes per direction
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "TotLen Fwd Pkts": "Fwd Packets Length Total",
    "TotLen Bwd Pkts": "Bwd Packets Length Total",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    # Init window bytes
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "Init Fwd Win Byts": "Init Fwd Win Bytes",
    "Init Bwd Win Byts": "Init Bwd Win Bytes",
    # Active data packets / min segment
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "Fwd Act Data Pkts": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
    # Packet length stats
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    "Pkt Len Min": "Packet Length Min",
    "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Average Packet Size": "Avg Packet Size",
    "Pkt Size Avg": "Avg Packet Size",
    # Per-direction packet length stats (v3 abbreviates "Packet" -> "Pkt")
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    # Per-direction packet/byte rates
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow Byts/s": "Flow Bytes/s",
    # Header lengths
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    # Segment size
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    # Subflow
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    # IAT
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
}


def _label_to_binary(s: pd.Series) -> np.ndarray:
    """Map the 2018 label column to {0=Benign, 1=Attack}.

    Different mirrors encode labels differently:
    - The dhoogla mirror keeps the original string labels ("Benign",
      "DDoS attack-HOIC", ...).
    - The ekkykharismadhany mirror integer-encodes them: 1 = Benign,
      2..N = various attack types (Benign happens to be label 1
      because it's the majority class after sorting).

    We accept both: strings are matched case-insensitively, integers
    take the dominant class as Benign.
    """
    if pd.api.types.is_numeric_dtype(s):
        # Whichever encoded label has the highest support is Benign.
        # We confirmed this is class 1 on the ekkykharismadhany mirror.
        benign_id = int(s.value_counts().idxmax())
        return (s.astype(int) != benign_id).astype(np.int64).to_numpy()
    s = s.astype(str).str.strip()
    return np.where(s.str.lower() == "benign", 0, 1).astype(np.int64)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace + apply alias map so column names match the
    51-feature schema."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items()
                            if k in df.columns})
    return df


def _align_to_schema(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Project df onto feature_cols. Missing columns become 0.0.

    Returns the projected frame plus diagnostic lists (present, missing).
    """
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    out = df[feature_cols].astype("float64")
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out, present, missing


def _load_2018(data_dir: str, feature_cols: list[str]) -> tuple[
    np.ndarray, np.ndarray, dict
]:
    """Concat every parquet/csv under data_dir into one (X, y, meta) tuple.

    Supports both shipping formats we've seen on Kaggle:
    - dhoogla/csecicids2018: 10 parquet files, one per attack/day.
    - ekkykharismadhany/csecicids2018-cleaned: one big sampled CSV.
    """
    files = sorted(
        p for p in os.listdir(data_dir)
        if p.endswith(".parquet") or p.endswith(".csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No parquet/csv files in {data_dir}. Drop the CICIDS-2018 "
            f"mirror there and rerun."
        )
    print(f"Loading {len(files)} files from {data_dir}")

    frames: list[pd.DataFrame] = []
    per_file_stats: list[dict] = []
    for f in files:
        path = os.path.join(data_dir, f)
        if f.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        # Some mirrors include a pandas index column — drop it.
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = _normalise_columns(df)
        # The label column may be named "Label" or "label" — accept both.
        label_col = next(
            (c for c in df.columns if c.lower() == "label"), None
        )
        if label_col is None:
            print(f"  ! {f}: no label column, skipping")
            continue
        y = _label_to_binary(df[label_col])
        df = df.drop(columns=[label_col])
        n_attack = int(y.sum())
        n_benign = int((y == 0).sum())
        per_file_stats.append({
            "file": f,
            "rows": int(len(df)),
            "benign": n_benign,
            "attack": n_attack,
        })
        df["__y__"] = y
        frames.append(df)
        print(f"  {f:<70} {len(df):>10,} rows "
              f"(benign={n_benign:,} attack={n_attack:,})")

    full = pd.concat(frames, ignore_index=True)
    y_full = full["__y__"].to_numpy()
    full = full.drop(columns=["__y__"])
    X_full, present, missing = _align_to_schema(full, feature_cols)

    meta = {
        "n_files": len(files),
        "files": per_file_stats,
        "schema": {
            "n_expected": len(feature_cols),
            "n_present": len(present),
            "n_missing": len(missing),
            "missing_columns": missing,
        },
    }
    print(f"\nSchema: {len(present)}/{len(feature_cols)} features present, "
          f"{len(missing)} filled with zeros")
    if missing:
        print(f"  missing: {missing}")
    return X_full.to_numpy(), y_full, meta


# --- Metric plumbing reused from evaluate_models.py ------------------------


def _confusion_rates(cm: np.ndarray) -> dict[str, float]:
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
        y_true, y_pred, labels=[0, 1],
        target_names=CLASS_NAMES, digits=4,
        output_dict=True, zero_division=0,
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
        "inference_us_per_sample": float(
            inference_seconds / len(y_true) * 1_000_000.0
        ) if len(y_true) else 0.0,
        "throughput_samples_per_sec": float(len(y_true) / inference_seconds)
        if inference_seconds > 0 else None,
    }


def _print_block(m: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {m['model']}")
    print("=" * 78)
    print(f"  Samples evaluated : {m['n_samples']:,}")
    print(f"  Inference total   : {m['inference_seconds_total']:.3f} s")
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
    print(f"    Benign (true)         {r['true_negative']:>10d}  {r['false_positive']:>10d}")
    print(f"    Attack (true)         {r['false_negative']:>10d}  {r['true_positive']:>10d}")


def _iso_risk_score(iso_model, X_scaled: np.ndarray) -> np.ndarray:
    raw = iso_model.score_samples(X_scaled)
    anom = iso_model.offset_ - raw
    return 1.0 / (1.0 + np.exp(-anom * 10.0))


# --- Main -----------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("AI-NGFW/IDS — Cross-Dataset Evaluation on CICIDS-2018")
    print("=" * 78)
    print(f"Models  : {MODELS_PATH}")
    print(f"Data    : {DATA_DIR}")
    print(f"Scaler  : {SCALER_PATH}")

    feature_cols = joblib.load(FEATURE_COLS_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Expecting {len(feature_cols)} features (CICIDS-2017 schema)")

    X_raw, y_test, data_meta = _load_2018(DATA_DIR, feature_cols)
    print(f"\nTotal samples : {len(y_test):,}")
    print(f"  Benign      : {int((y_test == 0).sum()):,}")
    print(f"  Attack      : {int((y_test == 1).sum()):,}")

    print("\nScaling with the 2017-trained StandardScaler...")
    X_scaled = scaler.transform(X_raw)

    rf = joblib.load(os.path.join(MODELS_PATH, "random_forest.joblib"))
    xgb = joblib.load(os.path.join(MODELS_PATH, "xgboost.joblib"))
    iso = joblib.load(os.path.join(MODELS_PATH, "isolation_forest.joblib"))
    ensemble = EnsembleDetector(models_path=MODELS_PATH)
    ensemble.load_models()

    all_metrics: dict[str, dict] = {}

    # Random Forest --------------------------------------------------------
    print("\nRunning Random Forest...")
    t0 = time.perf_counter()
    rf_pred = rf.predict(X_scaled)
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    all_metrics["random_forest"] = _metrics_block(
        "Random Forest", y_test, rf_pred, rf_proba,
        time.perf_counter() - t0,
    )

    # XGBoost --------------------------------------------------------------
    print("Running XGBoost...")
    t0 = time.perf_counter()
    xgb_pred = xgb.predict(X_scaled)
    xgb_proba = xgb.predict_proba(X_scaled)[:, 1]
    all_metrics["xgboost"] = _metrics_block(
        "XGBoost", y_test, xgb_pred, xgb_proba,
        time.perf_counter() - t0,
    )

    # Isolation Forest -----------------------------------------------------
    print("Running Isolation Forest...")
    t0 = time.perf_counter()
    iso_raw_pred = iso.predict(X_scaled)
    iso_pred = np.where(iso_raw_pred == -1, 1, 0)
    iso_score = _iso_risk_score(iso, X_scaled)
    all_metrics["isolation_forest"] = _metrics_block(
        "Isolation Forest", y_test, iso_pred, iso_score,
        time.perf_counter() - t0,
    )

    # Ensemble -------------------------------------------------------------
    print("Running Ensemble...")
    t0 = time.perf_counter()
    ensemble_score = ensemble.predict_ensemble(X_raw)
    ensemble_decision = ensemble.predict(X_raw)
    ens_dt = time.perf_counter() - t0
    ensemble_pred = (ensemble_decision >= 1).astype(int)
    ens_block = _metrics_block(
        "Ensemble (RF + XGB + IF, weighted)",
        y_test, ensemble_pred, ensemble_score, ens_dt,
    )
    counts = np.bincount(ensemble_decision, minlength=3)
    ens_block["decision_distribution"] = {
        "ALLOW": int(counts[0]),
        "INSPECT": int(counts[1]),
        "BLOCK": int(counts[2]),
    }
    all_metrics["ensemble"] = ens_block

    for key in ("random_forest", "xgboost", "isolation_forest", "ensemble"):
        _print_block(all_metrics[key])

    # Cross-dataset comparison vs CICIDS-2017 baseline ---------------------
    comparison: dict[str, dict] = {}
    baseline_metrics: dict = {}
    if BASELINE_PATH.exists():
        baseline = json.load(open(BASELINE_PATH))
        baseline_metrics = baseline.get("metrics", {})
        print("\n" + "=" * 78)
        print("  CICIDS-2017 vs CICIDS-2018  (Δ = 2018 − 2017)")
        print("=" * 78)
        print(f"{'model':<40}  {'acc-17':>7} {'acc-18':>7} {'Δacc':>7}  "
              f"{'f1-17':>6} {'f1-18':>6} {'Δf1':>6}  "
              f"{'auc-17':>6} {'auc-18':>6} {'Δauc':>6}")
        for key in ("random_forest", "xgboost", "isolation_forest", "ensemble"):
            cur = all_metrics[key]
            base = baseline_metrics.get(key, {})
            if not base:
                continue
            d_acc = cur["accuracy"] - base["accuracy"]
            d_f1 = cur["f1_score"] - base["f1_score"]
            d_auc = (cur["roc_auc"] or 0.0) - (base["roc_auc"] or 0.0)
            comparison[key] = {
                "acc_2017": base["accuracy"],
                "acc_2018": cur["accuracy"],
                "delta_acc": d_acc,
                "f1_2017": base["f1_score"],
                "f1_2018": cur["f1_score"],
                "delta_f1": d_f1,
                "auc_2017": base["roc_auc"],
                "auc_2018": cur["roc_auc"],
                "delta_auc": d_auc,
                "precision_2017": base["precision"],
                "precision_2018": cur["precision"],
                "recall_2017": base["recall"],
                "recall_2018": cur["recall"],
            }
            print(
                f"{cur['model']:<40}  "
                f"{base['accuracy']:>7.4f} {cur['accuracy']:>7.4f} "
                f"{d_acc:>+7.4f}  "
                f"{base['f1_score']:>6.4f} {cur['f1_score']:>6.4f} "
                f"{d_f1:>+6.4f}  "
                f"{(base['roc_auc'] or 0):>6.4f} {(cur['roc_auc'] or 0):>6.4f} "
                f"{d_auc:>+6.4f}"
            )

    # Persist --------------------------------------------------------------
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_path": MODELS_PATH,
        "data_path": DATA_DIR,
        "scaler_path": SCALER_PATH,
        "test_set": {
            "n_samples": int(len(y_test)),
            "n_benign": int((y_test == 0).sum()),
            "n_attack": int((y_test == 1).sum()),
        },
        "data_meta": data_meta,
        "metrics": all_metrics,
        "comparison_with_cicids2017": comparison,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
