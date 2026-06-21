"""
Multi-iteration honeypot feedback-loop demonstration.

Runs N iterations of (capture → verify → retrain → measure) against
CICIDS-2018 to show that accuracy keeps climbing as the analyst
labels more captures. Each iteration:

    1. Samples 500 *fresh* blocked-by-baseline flows from the pool
       (excludes anything already captured in earlier iterations).
    2. Splits them by ground-truth into ~400 real attacks (verified)
       and ~100 false positives.
    3. Adds the cumulative verified set to a frozen 2017 train
       subsample and retrains Random Forest from scratch.
    4. Re-evaluates on the same held-out 2018 evaluation slice.

The eval set is partitioned once, up-front; it is never used for
capture or retrain, so the curve reflects true cross-dataset
generalisation.

Outputs an ASCII learning curve, a per-iteration metrics table, and
writes ``training/multi_iteration_results.json`` for the dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.evaluate_cicids2018 import (  # noqa: E402
    DATA_DIR,
    MODELS_PATH,
    SCALER_PATH,
    _load_2018,
)

from app.api.model_metrics import ModelMetricsStore  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    recall_score,
)


# --- Demo knobs --------------------------------------------------------------

N_ITERATIONS = 4
N_CAPTURES_PER_ITER = 500           # blocked flows the honeypot intercepts
N_VERIFIED_REAL_PER_ITER = 400      # analyst confirms as real attacks
N_FALSE_POSITIVES_PER_ITER = (
    N_CAPTURES_PER_ITER - N_VERIFIED_REAL_PER_ITER
)
TRAIN_SUBSAMPLE = 300_000           # frozen 2017 train slice each iter
SEED = 42

LEGACY_RESULTS_PATH = _PROJECT_ROOT / "training" / "feedback_loop_results.json"
RESULTS_PATH = _PROJECT_ROOT / "training" / "multi_iteration_results.json"
SPLITS_PATH = os.path.expanduser(
    "~/sem6el/data/processed/balanced_splits.joblib"
)


# --- Output helpers ---------------------------------------------------------

WIDTH = 64

def banner(title: str) -> None:
    print("\n" + "═" * WIDTH)
    print(title.center(WIDTH))
    print("═" * WIDTH)


def section(title: str) -> None:
    print("\n" + title)
    print("─" * WIDTH)


def iteration_table(baseline: dict, iters: list[dict]) -> None:
    rows = [(
        "Baseline", 0,
        f"{baseline['acc']*100:.1f}%",
        f"{baseline['fpr']*100:.1f}%",
        f"{baseline['f1']:.3f}",
    )]
    for it in iters:
        rows.append((
            f"Round {it['iteration']}",
            it["total_verified"],
            f"{it['acc']*100:.1f}%",
            f"{it['fpr']*100:.1f}%",
            f"{it['f1']:.3f}",
        ))
    col_w = (11, 14, 10, 9, 9)
    sep = "─"
    top = "┌" + "┬".join(sep * (w + 2) for w in col_w) + "┐"
    mid = "├" + "┼".join(sep * (w + 2) for w in col_w) + "┤"
    bot = "└" + "┴".join(sep * (w + 2) for w in col_w) + "┘"

    def fmt(vals):
        return "│ " + " │ ".join(
            f"{v:<{w}}" for v, w in zip(vals, col_w)
        ) + " │"

    print(top)
    print(fmt(("Iteration", "Total Samples", "Accuracy", "FPR", "F1")))
    print(mid)
    for name, n, acc, fpr, f1 in rows:
        print(fmt((name, str(n), acc, fpr, f1)))
    print(bot)


def learning_curve(points: list[tuple[int, float]]) -> None:
    """Cheap ASCII line chart of (samples, accuracy_pct)."""
    if not points:
        return
    y_max = 100.0
    y_min = 40.0
    rows = 8
    width = 40
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_max = max(xs) if max(xs) > 0 else 1
    # Build a grid: rows × width chars, plot points + connecting line.
    grid = [[" "] * width for _ in range(rows)]
    plotted: list[tuple[int, int]] = []  # (row, col)
    for x, y in points:
        col = int((x / x_max) * (width - 1))
        # Higher accuracy → lower row index (top of chart).
        row = int((1 - (y - y_min) / (y_max - y_min)) * (rows - 1))
        row = max(0, min(rows - 1, row))
        plotted.append((row, col))
        grid[row][col] = "●"
    # Connect with horizontal/vertical lines.
    for (r0, c0), (r1, c1) in zip(plotted, plotted[1:]):
        cs, ce = sorted((c0, c1))
        for c in range(cs + 1, ce):
            if grid[r1][c] == " ":
                grid[r1][c] = "─"
    # Y-axis labels.
    y_step = (y_max - y_min) / (rows - 1)
    print()
    for i, row in enumerate(grid):
        label = y_max - i * y_step
        print(f" {label:5.1f}% │ {''.join(row)}")
    print(f"        └" + "─" * width)
    # X-axis ticks.
    tick_count = 5
    ticks = []
    for k in range(tick_count):
        x = int(x_max * k / (tick_count - 1))
        col = int((x / x_max) * (width - 1)) if x_max else 0
        ticks.append((col, x))
    line = [" "] * width
    for col, x in ticks:
        label = str(x)
        start = max(0, col - len(label) // 2)
        for j, ch in enumerate(label):
            if start + j < width:
                line[start + j] = ch
    print("          " + "".join(line) + "  samples")


# --- Metric helpers ---------------------------------------------------------


def evaluate(rf, X_scaled: np.ndarray, y: np.ndarray) -> dict:
    pred = rf.predict(X_scaled)
    acc = float(accuracy_score(y, pred))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {"acc": acc, "rec": rec, "f1": f1, "fpr": fpr, "n": int(len(y))}


def evaluate_at_threshold(
    rf, X_scaled: np.ndarray, y: np.ndarray, threshold: float
) -> dict:
    """Score with predict_proba and a custom decision threshold.

    RandomForestClassifier.predict() always picks argmax (effectively
    threshold=0.5 for binary classification). To sweep we need
    predict_proba.
    """
    proba = rf.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    acc = float(accuracy_score(y, pred))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "acc": acc, "rec": rec, "f1": f1, "fpr": fpr,
        "n": int(len(y)), "threshold": float(threshold),
    }


# --- Main ------------------------------------------------------------------


def main() -> int:
    banner("MULTI-ITERATION FEEDBACK LOOP DEMONSTRATION")

    # --- Load + baseline -------------------------------------------------
    section("STEP 1: BASELINE PERFORMANCE")
    feature_cols = joblib.load(
        os.path.join(MODELS_PATH, "feature_columns.joblib")
    )
    scaler = joblib.load(SCALER_PATH)
    rf_baseline = joblib.load(os.path.join(MODELS_PATH, "random_forest.joblib"))

    print("Loading CICIDS-2018…")
    X_raw, y_full, _ = _load_2018(DATA_DIR, list(feature_cols))
    X_scaled = scaler.transform(X_raw)

    rng = np.random.default_rng(SEED)
    n = len(y_full)
    perm = rng.permutation(n)
    eval_idx = perm[: n // 2]
    pool_idx = perm[n // 2:]
    X_eval, y_eval = X_scaled[eval_idx], y_full[eval_idx]

    print(f"  evaluation set: {len(y_eval):,} flows "
          f"({int((y_eval == 0).sum()):,} benign / "
          f"{int((y_eval == 1).sum()):,} attack)")
    print(f"  honeypot pool : {len(pool_idx):,} flows")

    baseline = evaluate(rf_baseline, X_eval, y_eval)
    print(f"\nBaseline accuracy : {baseline['acc']*100:.1f}%")
    print(f"Baseline FPR      : {baseline['fpr']*100:.1f}%")
    print(f"Baseline F1       : {baseline['f1']:.3f}")

    # --- Load + freeze the 2017 train subsample --------------------------
    section("STEP 2: PREPARE TRAIN POOL")
    if not os.path.exists(SPLITS_PATH):
        print(f"  ! {SPLITS_PATH} not found — running degraded "
              f"(only honeypot data; expect noisy curve)")
        X_train = np.empty((0, X_scaled.shape[1]))
        y_train = np.empty(0, dtype=int)
        n_original = 0
    else:
        splits = joblib.load(SPLITS_PATH)
        X_train_full = splits.get("X_train_scaled", splits["X_train"])
        y_train_full = splits["y_train_binary"]
        n_original = len(X_train_full)
        if n_original > TRAIN_SUBSAMPLE:
            sub = rng.choice(n_original, size=TRAIN_SUBSAMPLE, replace=False)
            X_train = X_train_full[sub]
            y_train = y_train_full[sub]
        else:
            X_train, y_train = X_train_full, y_train_full
    print(f"2017 train subsample frozen at {len(X_train):,} rows "
          f"({int((y_train == 1).sum()):,} attack / "
          f"{int((y_train == 0).sum()):,} benign)")

    # Baseline predicts on the pool once — the honeypot only ever sees
    # flows the model already blocked, so we sample from this slice.
    pool_pred = rf_baseline.predict(X_scaled[pool_idx])
    blocked_in_pool = pool_idx[pool_pred == 1]
    pool_truth = y_full[blocked_in_pool]
    tps_global = blocked_in_pool[pool_truth == 1]
    fps_global = blocked_in_pool[pool_truth == 0]
    print(f"Blocked-by-baseline pool: {len(blocked_in_pool):,} flows "
          f"({len(tps_global):,} TP / {len(fps_global):,} FP)")

    # --- Iteration loop --------------------------------------------------
    section("STEP 3: ITERATIVE FEEDBACK LOOP")
    accumulated_real: list[int] = []
    accumulated_fp: list[int] = []
    used: set[int] = set()
    iters_out: list[dict] = []

    for it in range(1, N_ITERATIONS + 1):
        # Fresh capture: pick disjoint TPs and FPs each iteration.
        tps_avail = np.array(
            [i for i in tps_global if int(i) not in used], dtype=int
        )
        fps_avail = np.array(
            [i for i in fps_global if int(i) not in used], dtype=int
        )
        take_real = min(N_VERIFIED_REAL_PER_ITER, len(tps_avail))
        take_fp = min(N_FALSE_POSITIVES_PER_ITER, len(fps_avail))
        new_real = (
            rng.choice(tps_avail, size=take_real, replace=False)
            if take_real else np.array([], dtype=int)
        )
        new_fp = (
            rng.choice(fps_avail, size=take_fp, replace=False)
            if take_fp else np.array([], dtype=int)
        )
        used.update(int(i) for i in new_real)
        used.update(int(i) for i in new_fp)
        accumulated_real.extend(int(i) for i in new_real)
        accumulated_fp.extend(int(i) for i in new_fp)

        v_idx = np.array(accumulated_real, dtype=int)
        f_idx = np.array(accumulated_fp, dtype=int)

        print(f"\n  Iteration {it}: +{len(new_real)} verified attacks, "
              f"+{len(new_fp)} false positives  "
              f"(cum: {len(v_idx)} real / {len(f_idx)} FP)")

        # Retrain from scratch — fresh RF every round so the curve
        # reflects "what if we'd trained on this much honeypot data".
        parts = [X_train]
        labels = [y_train]
        if len(v_idx):
            parts.append(X_scaled[v_idx])
            labels.append(np.ones(len(v_idx), dtype=int))
        if len(f_idx):
            parts.append(X_scaled[f_idx])
            labels.append(np.zeros(len(f_idx), dtype=int))
        X_aug = np.vstack(parts)
        y_aug = np.concatenate(labels)

        t0 = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=80, max_depth=18, n_jobs=-1,
            class_weight="balanced", random_state=SEED,
        )
        rf.fit(X_aug, y_aug)
        train_seconds = time.perf_counter() - t0

        m = evaluate(rf, X_eval, y_eval)
        iters_out.append({
            "iteration": it,
            "captured_this_iter": int(len(new_real) + len(new_fp)),
            "verified_real_this_iter": int(len(new_real)),
            "false_positives_this_iter": int(len(new_fp)),
            "total_verified": int(len(v_idx)),
            "total_false_positives": int(len(f_idx)),
            "total_samples_added": int(len(v_idx) + len(f_idx)),
            "retrain_seconds": round(train_seconds, 2),
            **m,
        })
        print(f"    → acc {m['acc']*100:5.1f}%  "
              f"fpr {m['fpr']*100:5.1f}%  "
              f"f1 {m['f1']:.3f}  "
              f"({train_seconds:.1f}s retrain)")

    # --- Threshold tuning on a fresh validation split -------------------
    # We split the held-out CICIDS-2018 eval set (already disjoint from
    # the capture pool) into 20% validation and 80% test, sweep
    # thresholds on validation, then report the final number on test.
    # The model `rf` in scope is the round-N retrain.
    section("STEP 4: THRESHOLD TUNING (HONEST)")
    eval_perm = rng.permutation(len(eval_idx))
    val_size = int(len(eval_idx) * 0.20)
    val_local = eval_perm[:val_size]
    test_local = eval_perm[val_size:]
    X_val_t, y_val_t = X_eval[val_local], y_eval[val_local]
    X_test_t, y_test_t = X_eval[test_local], y_eval[test_local]
    print(f"Validation slice : {len(y_val_t):,} flows "
          f"(20% of held-out eval)")
    print(f"Test slice       : {len(y_test_t):,} flows "
          f"(80% of held-out eval)")
    print("Sweeping thresholds on validation (model is round-N retrain)…\n")

    threshold_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    val_sweep: list[dict] = []
    for t in threshold_grid:
        m = evaluate_at_threshold(rf, X_val_t, y_val_t, t)
        val_sweep.append(m)
        mark = ""
        print(f"  threshold {t:.2f}  →  val acc {m['acc']*100:5.2f}%   "
              f"fpr {m['fpr']*100:5.2f}%   f1 {m['f1']:.3f}{mark}")

    best = max(val_sweep, key=lambda m: m["acc"])
    best_threshold = best["threshold"]
    default_test = evaluate_at_threshold(rf, X_test_t, y_test_t, 0.50)
    tuned_test = evaluate_at_threshold(rf, X_test_t, y_test_t, best_threshold)
    print(f"\nBest threshold on validation : {best_threshold:.2f} "
          f"(val acc {best['acc']*100:.2f}%)")
    print(f"Test set evaluation:")
    print(f"  Default threshold (0.50) : "
          f"{default_test['acc']*100:5.2f}% acc  /  "
          f"{default_test['fpr']*100:5.2f}% fpr  /  f1 {default_test['f1']:.3f}")
    print(f"  Tuned threshold ({best_threshold:.2f}) : "
          f"{tuned_test['acc']*100:5.2f}% acc  /  "
          f"{tuned_test['fpr']*100:5.2f}% fpr  /  f1 {tuned_test['f1']:.3f}")
    test_lift = (tuned_test["acc"] - default_test["acc"]) * 100
    print(f"  Lift from tuning         : {test_lift:+.2f} pp")
    print("\nMethod: threshold picked on validation, reported on a "
          "disjoint test slice. The test slice was never used for "
          "training, capture, or threshold selection.")

    # --- Summary table + curve -------------------------------------------
    banner("MULTI-ITERATION FEEDBACK LOOP RESULTS")
    iteration_table(baseline, iters_out)
    points = [(0, baseline["acc"] * 100)] + [
        (it["total_verified"], it["acc"] * 100) for it in iters_out
    ]
    print("\nLEARNING CURVE")
    learning_curve(points)

    final = iters_out[-1]
    delta = (final["acc"] - baseline["acc"]) * 100.0
    pct_of_train = (
        final["total_samples_added"] / max(1, n_original + final["total_samples_added"]) * 100.0
    )
    print(
        f"\nWith {final['total_verified']} verified attacks "
        f"(+ {final['total_false_positives']} FPs labeled), "
        f"~{pct_of_train:.2f}% of training data,"
    )
    print(
        f"accuracy moved from {baseline['acc']*100:.1f}% → "
        f"{final['acc']*100:.1f}% on unseen CICIDS-2018 "
        f"({delta:+.1f} pp)."
    )
    print(
        f"This simulates {N_ITERATIONS} days of production honeypot "
        f"feedback at ~{N_CAPTURES_PER_ITER} captures/day."
    )
    print("═" * WIDTH)

    # --- Persist -----------------------------------------------------------
    out = {
        "baseline": {
            **baseline,
            "accuracy_pct": round(baseline["acc"] * 100, 2),
            "fpr_pct":      round(baseline["fpr"] * 100, 2),
            "recall_pct":   round(baseline["rec"] * 100, 2),
        },
        "iterations": [
            {
                **it,
                "accuracy_pct": round(it["acc"] * 100, 2),
                "fpr_pct":      round(it["fpr"] * 100, 2),
                "recall_pct":   round(it["rec"] * 100, 2),
            }
            for it in iters_out
        ],
        "config": {
            "n_iterations": N_ITERATIONS,
            "captures_per_iter": N_CAPTURES_PER_ITER,
            "verified_real_per_iter": N_VERIFIED_REAL_PER_ITER,
            "false_positives_per_iter": N_FALSE_POSITIVES_PER_ITER,
            "train_subsample": TRAIN_SUBSAMPLE,
            "seed": SEED,
        },
        "final_delta_accuracy_pct": round(delta, 2),
        "eval_set_size": int(len(y_eval)),
        "n_original_train": int(n_original),
        "threshold_tuning": {
            "method": (
                "Threshold swept on a 20% validation split of the "
                "held-out 2018 eval set; final reported on the "
                "disjoint 80% test slice."
            ),
            "grid": threshold_grid,
            "best_threshold": float(best_threshold),
            "val_sweep": [
                {
                    "threshold": round(m["threshold"], 3),
                    "accuracy_pct": round(m["acc"] * 100, 2),
                    "fpr_pct":      round(m["fpr"] * 100, 2),
                    "f1": round(m["f1"], 4),
                }
                for m in val_sweep
            ],
            "test_default": {
                "threshold": 0.50,
                "accuracy_pct": round(default_test["acc"] * 100, 2),
                "fpr_pct":      round(default_test["fpr"] * 100, 2),
                "f1": round(default_test["f1"], 4),
                "n": default_test["n"],
            },
            "test_tuned": {
                "threshold": round(float(best_threshold), 3),
                "accuracy_pct": round(tuned_test["acc"] * 100, 2),
                "fpr_pct":      round(tuned_test["fpr"] * 100, 2),
                "f1": round(tuned_test["f1"], 4),
                "n": tuned_test["n"],
            },
            "lift_pp": round(test_lift, 2),
            "val_size": int(len(y_val_t)),
            "test_size": int(len(y_test_t)),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            out, fh, indent=2,
            default=lambda v: float(v) if hasattr(v, "item") else str(v),
        )
    print(f"\nSaved results to {RESULTS_PATH}")

    # Keep the legacy single-iteration file populated from iteration 1
    # so /model/cross_dataset (which the dashboard still reads) reflects
    # the same first-round numbers.
    first = iters_out[0]
    legacy = {
        "before": {
            **baseline,
            "accuracy_pct": round(baseline["acc"] * 100, 2),
            "fpr_pct":      round(baseline["fpr"] * 100, 2),
            "recall_pct":   round(baseline["rec"] * 100, 2),
        },
        "after": {
            "acc": first["acc"], "rec": first["rec"], "f1": first["f1"],
            "fpr": first["fpr"], "n": first["n"],
            "accuracy_pct": round(first["acc"] * 100, 2),
            "fpr_pct":      round(first["fpr"] * 100, 2),
            "recall_pct":   round(first["rec"] * 100, 2),
        },
        "captured": int(first["captured_this_iter"]),
        "verified_real": int(first["verified_real_this_iter"]),
        "verified_false_positives": int(first["false_positives_this_iter"]),
        "samples_added_to_train": int(first["verified_real_this_iter"]),
        "retrain_seconds": first["retrain_seconds"],
        "delta_accuracy_pct": round(
            (first["acc"] - baseline["acc"]) * 100, 2
        ),
    }
    with LEGACY_RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            legacy, fh, indent=2,
            default=lambda v: float(v) if hasattr(v, "item") else str(v),
        )
    print(f"Refreshed legacy single-round file at {LEGACY_RESULTS_PATH}")

    # Prime the live metrics store with the FULL multi-iteration
    # history: baseline + one entry per round + the threshold-tuning
    # bump. Wipes prior history so re-runs produce a clean log.
    try:
        store = ModelMetricsStore()
        store.reset_history(
            baseline_accuracy=round(baseline["acc"] * 100, 2),
            baseline_version="v1.0",
            baseline_label="Baseline",
        )
        for it in iters_out:
            store.record_retrain(
                new_accuracy=round(it["acc"] * 100, 2),
                samples_added=int(it["total_verified"]),
                version=f"v{it['iteration'] + 1}.0",
                date_label=f"Round {it['iteration']}",
                note=(
                    f"Round {it['iteration']}: "
                    f"{it['total_verified']} verified attacks, "
                    f"{it['total_false_positives']} false positives"
                ),
            )
        # Final entry — same model, tuned threshold, applied to the
        # disjoint test slice. Versioned as a minor bump on the last
        # round (e.g. v5.0 → v5.1) since no retrain happened.
        store.record_retrain(
            new_accuracy=round(tuned_test["acc"] * 100, 2),
            samples_added=int(final["total_verified"]),
            version=None,  # let _bump_version handle the minor bump
            date_label="+ Threshold",
            note=(
                f"Threshold tuned to {best_threshold:.2f} on validation; "
                f"evaluated on disjoint test slice"
            ),
        )
        # Reset training queue + verified counters since this is a demo
        # priming run, not a real-time accumulation.
        store.set_training_queue_size(0)
        print(f"Updated live metrics store at {store.path}")
    except Exception as e:  # noqa: BLE001
        print(f"  (metrics store update skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
