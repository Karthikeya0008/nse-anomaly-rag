"""
SHAP explanations for the five ensemble detectors and the ensemble itself.

For every flagged day: how much did each feature push each model's anomaly score
up (towards "anomalous") or down, relative to a typical day?

Method -- exact interventional Shapley values (what SHAP's KernelExplainer
approximates). With 4 shock / 6 regime features, all 2^d feature coalitions S
can be evaluated exactly:

    v(S)   = mean over background days b of f(x_S, b_rest)
             (the day's own values for features in S, a typical day's for the rest)
    phi_i  = sum over S not containing i of |S|!(d-|S|-1)!/d! * (v(S + i) - v(S))

so  base value v({}) + sum(phi) = f(x)  exactly (checked below). The background
is a fixed random sample of non-flagged days, so the base value is "the score of
a typical, normal day".

f is each model's anomaly score (higher = more anomalous), fitted on the full
history as in production. For LOF and kNN, f scores a day against the stored
training days (novelty mode). The ensemble is explained through its own
function: the average, over models, of the score's percentile among training
days -- exactly how the production ensemble is formed.

Usage:
    python explain_shap.py            # NIFTY50
Writes:
    db/processed/<SECTOR>_shap.json
"""
import json
import math
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

import train_multi_model as tm

SETS = ("shock", "regime")
MODELS = tm.ENSEMBLE_MODELS
BACKGROUND_SIZE = 100
SEED = 0

FEATURE_LABELS = {
    "log_return": "Daily return",
    "gap": "Opening gap",
    "range_vol": "Intraday range",
    "return_zscore": "Return z-score (vs 20 days)",
    "return_5d": "5-day return",
    "return_20d": "20-day return",
    "return_60d": "60-day return",
    "volatility_5d": "5-day volatility",
    "volatility_20d": "20-day volatility",
    "volatility_60d": "60-day volatility",
    "ma_deviation_60d": "Distance from 60-day average",
    "rolling_sharpe_20d": "20-day trend consistency",
}
# How to display a feature value: as a percentage or as a plain number.
PERCENT_FEATURES = {"log_return", "gap", "range_vol", "return_5d", "return_20d", "return_60d",
                    "volatility_5d", "volatility_20d", "volatility_60d", "ma_deviation_60d"}


def shapley_weights(d: int) -> np.ndarray:
    """w[s] = s! (d-s-1)! / d! for coalition size s."""
    return np.array([math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d) for s in range(d)])


def exact_shapley(values: np.ndarray, d: int) -> np.ndarray:
    """values: (n_days, 2^d) coalition values v(S), S encoded as a bitmask.
    Returns phi: (n_days, d)."""
    w = shapley_weights(d)
    phi = np.zeros((values.shape[0], d))
    for mask in range(2 ** d):
        size = bin(mask).count("1")
        for i in range(d):
            if not mask >> i & 1:
                phi[:, i] += w[size] * (values[:, mask | 1 << i] - values[:, mask])
    return phi


def coalition_inputs(X_days: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Rows for every (day, coalition, background day): the day's features where
    the coalition bit is set, the background day's elsewhere.
    Shape (n_days * 2^d * B, d), ordered day-major, then coalition, then background."""
    n, d = X_days.shape
    masks = np.array(list(product([0, 1], repeat=d)))[:, ::-1].astype(bool)  # row k = bitmask k
    Z = np.where(masks[None, :, None, :], X_days[:, None, None, :], background[None, None, :, :])
    return Z.reshape(-1, d)


def explain_set(features: pd.DataFrame, comparison: pd.DataFrame, fs: str, rng) -> dict:
    cols = tm.FEATURE_SETS[fs]
    d = len(cols)
    X = features[cols].to_numpy()
    flagged = comparison["ensemble_anomaly"].to_numpy()
    days = np.flatnonzero(flagged)
    background = X[rng.choice(np.flatnonzero(~flagged), BACKGROUND_SIZE, replace=False)]

    Z = coalition_inputs(X[days], background)
    shape = (len(days), 2 ** d, BACKGROUND_SIZE)
    coalition_values, model_scores, ecdf_values = {}, {}, []

    for m in MODELS:
        t = time.time()
        _, train_scores = tm.MODELS[m](X)          # in-sample, as production
        _, z_scores = tm.MODELS[m](X, Z)           # same fit, scoring the coalition rows
        coalition_values[m] = z_scores.reshape(shape).mean(axis=2)
        # Percentile of each score among training days (production ranks).
        sorted_train = np.sort(train_scores)
        ecdf_values.append(np.searchsorted(sorted_train, z_scores, side="right") / len(sorted_train))
        print(f"  {fs}/{m}: {len(Z):,} evaluations in {time.time() - t:.0f}s")
    coalition_values["ensemble"] = np.mean(ecdf_values, axis=0).reshape(shape).mean(axis=2)

    out = {"features": cols, "models": [*MODELS, "ensemble"], "base_values": {},
           "typical": {c: float(np.median(features[c])) for c in cols},
           "global_importance": {}, "days": {}}
    max_err = 0.0
    for m, v in coalition_values.items():
        phi = exact_shapley(v, d)
        full, empty = v[:, -1], v[:, 0]
        max_err = max(max_err, float(np.abs(phi.sum(axis=1) - (full - empty)).max()))
        out["base_values"][m] = float(empty[0])
        out["global_importance"][m] = np.abs(phi).mean(axis=0).round(6).tolist()
        for k, row in enumerate(days):
            date = features["date"].iloc[row].strftime("%Y-%m-%d")
            day = out["days"].setdefault(date, {"values": X[row].round(6).tolist(), "models": {}})
            day["models"][m] = {"score": round(float(full[k]), 6), "phi": phi[k].round(6).tolist()}

    # The ensemble function vs the production ensemble score (LOF/kNN score a day
    # against training days that include itself, so they differ slightly).
    ens = np.array([out["days"][features["date"].iloc[r].strftime("%Y-%m-%d")]["models"]["ensemble"]["score"]
                    for r in days])
    corr = float(np.corrcoef(ens, comparison["ensemble_score"].to_numpy()[days])[0, 1])
    out["checks"] = {"max_additivity_error": max_err, "ensemble_vs_production_corr": corr,
                     "explained_days": int(len(days))}
    print(f"  {fs}: additivity error {max_err:.2e}, ensemble vs production r = {corr:.3f}")
    return out


def main():
    sector = sys.argv[1] if len(sys.argv) > 1 else "NIFTY50"
    rng = np.random.default_rng(SEED)
    features = pd.read_csv(tm.PROCESSED_DIR / f"{sector}_features.csv", parse_dates=["date"])
    result = {
        "sector": sector,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "exact interventional Shapley values over all feature coalitions",
        "background": f"{BACKGROUND_SIZE} random non-flagged days (seed {SEED})",
        "feature_labels": FEATURE_LABELS,
        "percent_features": sorted(PERCENT_FEATURES),
        "sets": {},
    }
    for fs in SETS:
        comparison = pd.read_csv(tm.PROCESSED_DIR / f"{sector}_{fs}_model_comparison.csv")
        print(f"{fs}: explaining {int(comparison['ensemble_anomaly'].sum())} flagged days")
        result["sets"][fs] = explain_set(features, comparison, fs, rng)

    out_path = tm.PROCESSED_DIR / f"{sector}_shap.json"
    out_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
