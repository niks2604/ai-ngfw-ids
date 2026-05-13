"""
Cross-dataset generalisation: train on a *common* feature set that exists
in both CICIDS2017 and NSL-KDD, then evaluate on each.

Why this exists
---------------
The full 51-feature ensemble scores ~98% on CICIDS2017 but generalises
poorly to NSL-KDD because most CICIDS flow statistics (inter-arrival
times, subflow bytes, header lengths, ...) have no NSL-KDD analogue.
This script trains a smaller model on ~14 features that *can* be
derived from both schemas, so cross-dataset accuracy is a fair test
of learned representations rather than feature engineering overlap.

Common features (CICIDS expression  ↔  NSL-KDD expression)
----------------------------------------------------------
    duration_sec       Flow Duration / 1e6       duration
    src_bytes          Fwd Packets Length Total  src_bytes
    dst_bytes          Bwd Packets Length Total  dst_bytes
    total_bytes        src + dst                 src + dst
    byte_ratio         dst / (src + 1)           dst / (src + 1)
    total_pkts         Total Fwd + Total Bwd Pkts  count           (rough)
    pkt_rate           Flow Packets/s            count / (dur + 1) (rough)
    byte_rate          Flow Bytes/s              (src+dst)/(dur+1)
    serror_rate        SYN_count / max(1, pkts)  serror_rate
    rerror_rate        RST_count / max(1, pkts)  rerror_rate
    proto_tcp          Protocol == 6             protocol_type == tcp
    proto_udp          Protocol == 17            protocol_type == udp
    proto_icmp         Protocol == 1             protocol_type == icmp
    log_total_bytes    log1p(src + dst)          log1p(src + dst)

NSL-KDD's destination-port → service mapping is intentionally dropped —
our processed CICIDS parquet does not retain destination port, so any
service-derived feature would be unavailable on the CICIDS side.

Outputs
-------
- ~/sem6el/trained_models/common/{random_forest,xgboost,scaler}_common.joblib
- ~/sem6el/trained_models/common/feature_columns_common.joblib
- <project>/cross_dataset_results.json
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CICIDS_PARQUET = os.path.expanduser("~/sem6el/data/processed/processed_data.parquet")
NSLKDD_TEST = os.path.expanduser("~/sem6el/data/nslkdd/KDDTest+.txt")
NSLKDD_CALIB = os.path.expanduser("~/sem6el/data/nslkdd/KDDTrain+_20Percent.txt")
OUT_DIR = Path(os.path.expanduser("~/sem6el/trained_models/common"))
RESULTS_PATH = _PROJECT_ROOT / "cross_dataset_results.json"

# Scale-invariant / log-compressed features. The raw byte counts in CICIDS
# (DDoS flows: 10^7+) and NSL-KDD (mostly < 10^4) live in different orders of
# magnitude, so we replace them with log1p-compressed versions and ratios that
# transfer cleanly across the two datasets.
COMMON_FEATURES = [
    "log_duration",
    "log_src_bytes",
    "log_dst_bytes",
    "log_total_bytes",
    "byte_ratio",
    "log_pkt_rate",
    "log_byte_rate",
    "serror_rate",
    "rerror_rate",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "no_response",       # dst_bytes == 0 — signature of scans / SYN floods
    "very_short_conn",   # duration_sec < 1 — both datasets can compute this
]

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]


# --- Feature derivation ----------------------------------------------------


def _pack_common(
    duration_sec: pd.Series,
    src_bytes: pd.Series,
    dst_bytes: pd.Series,
    total_pkts: pd.Series,
    serror_rate: pd.Series,
    rerror_rate: pd.Series,
    proto_tcp: pd.Series,
    proto_udp: pd.Series,
    proto_icmp: pd.Series,
) -> pd.DataFrame:
    """Shared feature-derivation logic for both datasets."""
    total_bytes = src_bytes + dst_bytes
    safe_dur = np.maximum(duration_sec, 1e-3)   # 1ms floor for rate stability
    byte_rate = total_bytes / safe_dur
    pkt_rate = total_pkts / safe_dur

    out = pd.DataFrame({
        "log_duration":    np.log1p(np.maximum(duration_sec, 0.0)),
        "log_src_bytes":   np.log1p(np.maximum(src_bytes, 0.0)),
        "log_dst_bytes":   np.log1p(np.maximum(dst_bytes, 0.0)),
        "log_total_bytes": np.log1p(np.maximum(total_bytes, 0.0)),
        "byte_ratio":      dst_bytes / (src_bytes + 1.0),
        "log_pkt_rate":    np.log1p(np.maximum(pkt_rate, 0.0)),
        "log_byte_rate":   np.log1p(np.maximum(byte_rate, 0.0)),
        "serror_rate":     serror_rate.clip(0.0, 1.0),
        "rerror_rate":     rerror_rate.clip(0.0, 1.0),
        "proto_tcp":       proto_tcp.astype(np.int8),
        "proto_udp":       proto_udp.astype(np.int8),
        "proto_icmp":      proto_icmp.astype(np.int8),
        "no_response":     (dst_bytes <= 0).astype(np.int8),
        "very_short_conn": (duration_sec < 1.0).astype(np.int8),
    })
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def cicids_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """Project CICIDS2017 rows onto the common feature space."""
    duration_sec = df["Flow Duration"].astype(float) / 1_000_000.0
    src_bytes = df["Fwd Packets Length Total"].astype(float)
    dst_bytes = df["Bwd Packets Length Total"].astype(float)
    total_pkts = (df["Total Fwd Packets"] + df["Total Backward Packets"]).astype(float)
    pkts_safe = np.maximum(total_pkts, 1.0)

    serror_rate = df["SYN Flag Count"].astype(float) / pkts_safe
    rerror_rate = df["RST Flag Count"].astype(float) / pkts_safe

    proto = df["Protocol"].astype(int)
    return _pack_common(
        duration_sec=duration_sec,
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        total_pkts=total_pkts,
        serror_rate=serror_rate,
        rerror_rate=rerror_rate,
        proto_tcp=(proto == 6),
        proto_udp=(proto == 17),
        proto_icmp=(proto == 1),
    )


def nslkdd_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """Project NSL-KDD rows onto the common feature space."""
    duration_sec = df["duration"].astype(float)
    src_bytes = df["src_bytes"].astype(float)
    dst_bytes = df["dst_bytes"].astype(float)
    # NSL-KDD `count` = number of connections to the same host in the past 2s.
    # It's the closest available proxy for "packets observed for this flow".
    total_pkts = df["count"].astype(float)

    proto = df["protocol_type"].astype(str).str.lower()
    return _pack_common(
        duration_sec=duration_sec,
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        total_pkts=total_pkts,
        serror_rate=df["serror_rate"].astype(float),
        rerror_rate=df["rerror_rate"].astype(float),
        proto_tcp=(proto == "tcp"),
        proto_udp=(proto == "udp"),
        proto_icmp=(proto == "icmp"),
    )


# --- Eval helpers ----------------------------------------------------------


def _binary_metrics(name: str, y_true, y_pred, y_score, t_seconds: float) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else None,
        "confusion_matrix": cm.tolist(),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "n_samples": int(len(y_true)),
        "inference_seconds": float(t_seconds),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Benign", "Attack"], output_dict=True, zero_division=0,
        ),
    }


def _print_block(title: str, m: dict) -> None:
    print("\n" + "─" * 78)
    print(f"  {title}")
    print("─" * 78)
    print(f"  samples     : {m['n_samples']:,}")
    print(f"  accuracy    : {m['accuracy']:.4f}")
    print(f"  precision   : {m['precision']:.4f}")
    print(f"  recall      : {m['recall']:.4f}")
    print(f"  f1          : {m['f1_score']:.4f}")
    if m["roc_auc"] is not None:
        print(f"  roc_auc     : {m['roc_auc']:.4f}")
    print(f"  FPR / FNR   : {m['false_positive_rate']:.4f} / {m['false_negative_rate']:.4f}")
    cm = m["confusion_matrix"]
    print("  confusion   :   (rows = true, cols = predicted)")
    print(f"      Benign  : {cm[0][0]:>10d}  {cm[0][1]:>10d}")
    print(f"      Attack  : {cm[1][0]:>10d}  {cm[1][1]:>10d}")


# --- Main ------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("AI-NGFW/IDS — Common-Feature Cross-Dataset Trainer")
    print("=" * 78)
    print("Train: CICIDS2017 (sampled)   Eval: CICIDS2017 + NSL-KDD")
    print(f"Common features ({len(COMMON_FEATURES)}): {COMMON_FEATURES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load + project CICIDS2017 ----------------------------------------
    print("\n[1/5] Loading CICIDS2017 parquet …")
    df = pd.read_parquet(CICIDS_PARQUET)
    print(f"      rows: {len(df):,}")

    print("[2/5] Projecting onto common feature space …")
    # The parquet underwent SMOTE oversampling upstream: ~1.5M rows have
    # textual Label='Benign' but were assigned attack_label/is_attack > 0.
    # Those synthetic rows have benign feature distributions, so trusting
    # the numeric labels makes the problem unlearnable. The textual `Label`
    # column is the only consistent ground truth — restrict to its rows.
    mask_clean = df["Label"].notna()
    df_clean = df.loc[mask_clean].reset_index(drop=True)
    X_cic = cicids_to_common(df_clean)
    y_cic = (df_clean["Label"].astype(str) != "Benign").astype(int).to_numpy()
    print(f"      class balance: benign={int((y_cic == 0).sum()):,}  "
          f"attack={int((y_cic == 1).sum()):,}")

    # Build a *binary-balanced* sample where the attack half is itself
    # stratified across CICIDS attack subtypes (capped per subtype). This
    # avoids the model overfitting to volumetric DoS Hulk / DDoS — those
    # two alone make up >70% of raw CICIDS attacks — at the expense of
    # rarer scan / brute-force patterns that matter for NSL-KDD transfer.
    rng = np.random.default_rng(42)
    cap_per_subtype = 25_000
    attack_parts = []
    for label, group in df_clean.loc[df_clean["Label"] != "Benign"].groupby("Label", sort=False):
        idx = group.index.to_numpy()
        if len(idx) > cap_per_subtype:
            idx = rng.choice(idx, size=cap_per_subtype, replace=False)
        attack_parts.append(idx)
    attack_pick = np.concatenate(attack_parts)

    benign_idx = df_clean.index[df_clean["Label"] == "Benign"].to_numpy()
    benign_pick = rng.choice(
        benign_idx, size=min(len(attack_pick), len(benign_idx)), replace=False
    )
    pick = np.concatenate([benign_pick, attack_pick])
    rng.shuffle(pick)

    X_cic_s = X_cic.iloc[pick].reset_index(drop=True)
    y_cic_s = y_cic[pick]
    bin_counts = np.bincount(y_cic_s, minlength=2)
    print(f"      stratified sample: {len(X_cic_s):,}  "
          f"benign={bin_counts[0]:,}  attack={bin_counts[1]:,}")
    # Show attack-subtype distribution so we can verify diversity.
    sub_dist = (
        df_clean.iloc[attack_pick]["Label"].value_counts().to_dict()
    )
    print(f"      attack subtype mix: {sub_dist}")

    # 80/20 split, stratified.
    X_train, X_test_cic, y_train, y_test_cic = train_test_split(
        X_cic_s.to_numpy(), y_cic_s, test_size=0.2, random_state=42, stratify=y_cic_s
    )

    # --- Scale ------------------------------------------------------------
    # RobustScaler (median/IQR) transfers across datasets better than
    # StandardScaler: CICIDS DDoS flows would otherwise dominate the mean
    # and shrink NSL-KDD's smaller-byte attacks toward "looks benign".
    print("[3/5] Fitting RobustScaler on CICIDS training rows …")
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_cic_s = scaler.transform(X_test_cic)

    # --- Train ------------------------------------------------------------
    print("[4/5] Training models …")
    t = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, class_weight="balanced", random_state=42,
    )
    rf.fit(X_train_s, y_train)
    print(f"      Random Forest trained in {time.perf_counter()-t:.1f}s")

    t = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    xgb.fit(X_train_s, y_train)
    print(f"      XGBoost       trained in {time.perf_counter()-t:.1f}s")

    # --- Persist ----------------------------------------------------------
    joblib.dump(rf, OUT_DIR / "random_forest_common.joblib")
    joblib.dump(xgb, OUT_DIR / "xgboost_common.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler_common.joblib")
    joblib.dump(COMMON_FEATURES, OUT_DIR / "feature_columns_common.joblib")

    # --- Evaluate on CICIDS held-out --------------------------------------
    print("[5/5] Evaluating …")
    results: dict[str, dict] = {}

    for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
        t = time.perf_counter()
        y_pred = model.predict(X_test_cic_s)
        y_score = model.predict_proba(X_test_cic_s)[:, 1]
        dt = time.perf_counter() - t
        m = _binary_metrics(name, y_test_cic, y_pred, y_score, dt)
        results.setdefault(name, {})["cicids2017"] = m

    # --- Evaluate on NSL-KDD ---------------------------------------------
    # We calibrate the decision threshold on the NSL-KDD *training* split
    # (KDDTrain+_20Percent.txt) — a fair source of in-domain validation that
    # the model never saw during training — and apply it to KDDTest+.txt.
    # Both pre- and post-calibration metrics are reported.
    print("      loading NSL-KDD calibration (train) + test sets …")
    nsl_calib = pd.read_csv(NSLKDD_CALIB, header=None, names=NSLKDD_COLUMNS)
    nsl_test = pd.read_csv(NSLKDD_TEST, header=None, names=NSLKDD_COLUMNS)
    X_calib_s = scaler.transform(nslkdd_to_common(nsl_calib).to_numpy())
    y_calib = (nsl_calib["label"].astype(str) != "normal").astype(int).to_numpy()
    X_nsl_s = scaler.transform(nslkdd_to_common(nsl_test).to_numpy())
    y_nsl = (nsl_test["label"].astype(str) != "normal").astype(int).to_numpy()
    print(f"      nsl-kdd calib: {len(nsl_calib):,}  test: {len(nsl_test):,}  "
          f"benign={int((y_nsl == 0).sum()):,}  attack={int((y_nsl == 1).sum()):,}")

    def _pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
        # Sweep candidate thresholds and pick the one that maximises accuracy
        # on the calibration set. Restricted to [0.05, 0.95] to avoid degenerate
        # all-attack / all-benign predictions.
        grid = np.linspace(0.05, 0.95, 19)
        best, best_acc = 0.5, -1.0
        for t in grid:
            acc = accuracy_score(y_true, (scores >= t).astype(int))
            if acc > best_acc:
                best, best_acc = float(t), float(acc)
        return best

    for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
        t = time.perf_counter()
        y_score_nsl = model.predict_proba(X_nsl_s)[:, 1]
        dt = time.perf_counter() - t

        # Pre-calibration: default 0.5 threshold.
        y_pred_default = (y_score_nsl >= 0.5).astype(int)
        results[name]["nslkdd"] = _binary_metrics(name, y_nsl, y_pred_default, y_score_nsl, dt)

        # Calibrated threshold from NSL-KDD train split.
        calib_scores = model.predict_proba(X_calib_s)[:, 1]
        thr = _pick_threshold(y_calib, calib_scores)
        y_pred_calib = (y_score_nsl >= thr).astype(int)
        calib_metrics = _binary_metrics(name, y_nsl, y_pred_calib, y_score_nsl, dt)
        calib_metrics["threshold"] = thr
        calib_metrics["calibration_source"] = "KDDTrain+_20Percent"
        results[name]["nslkdd_calibrated"] = calib_metrics

    # --- Print ------------------------------------------------------------
    for name in ("Random Forest", "XGBoost"):
        print("\n" + "=" * 78)
        print(f"📊  {name} (common features only, {len(COMMON_FEATURES)} dims)")
        print("=" * 78)
        _print_block("CICIDS2017 — in-distribution test", results[name]["cicids2017"])
        _print_block("NSL-KDD — cross-dataset (default threshold)", results[name]["nslkdd"])
        _print_block(
            f"NSL-KDD — cross-dataset (calibrated thr={results[name]['nslkdd_calibrated']['threshold']:.2f})",
            results[name]["nslkdd_calibrated"],
        )

    # Side-by-side summary
    print("\n" + "=" * 78)
    print("📌  Cross-dataset summary")
    print("=" * 78)
    print(f"{'model':<18} {'dataset':<28} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'auc':>7}")
    rows = (
        ("cicids2017",        "CICIDS2017 (in-dist)"),
        ("nslkdd",            "NSL-KDD (default thr=0.50)"),
        ("nslkdd_calibrated", "NSL-KDD (calibrated thr)"),
    )
    for name in ("Random Forest", "XGBoost"):
        for ds_key, ds_label in rows:
            m = results[name][ds_key]
            auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a "
            print(
                f"{name:<18} {ds_label:<28} "
                f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {auc:>7}"
            )

    # --- Persist JSON -----------------------------------------------------
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "common_features": COMMON_FEATURES,
        "train_dataset": "CICIDS2017",
        "train_sample_size": int(len(X_train)),
        "test_sample_size_cicids": int(len(X_test_cic)),
        "test_sample_size_nslkdd": int(len(nsl_test)),
        "calibration_source": "KDDTrain+_20Percent.txt",
        "models": {
            name: {
                "cicids2017": results[name]["cicids2017"],
                "nslkdd": results[name]["nslkdd"],
                "nslkdd_calibrated": results[name]["nslkdd_calibrated"],
            }
            for name in ("Random Forest", "XGBoost")
        },
        "artifacts": {
            "random_forest": str(OUT_DIR / "random_forest_common.joblib"),
            "xgboost": str(OUT_DIR / "xgboost_common.joblib"),
            "scaler": str(OUT_DIR / "scaler_common.joblib"),
            "feature_columns": str(OUT_DIR / "feature_columns_common.joblib"),
        },
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n💾  Saved cross-dataset results to {RESULTS_PATH}")
    print(f"💾  Saved common-feature models to   {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
