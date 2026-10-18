"""Train several unsupervised anomaly detectors on one sector's features and
combine them into an ensemble -- separately for each feature set:

    shock  -- single-day features: "was TODAY unusual?"
    regime -- 20/60-day features:  "is the market in an unusual PERIOD?"
    all    -- all 12 features together (what train_isolation_forest.py uses)

Usage:
    python train_multi_model.py                    # NIFTY50, all three sets
    python train_multi_model.py NIFTYBANK
    python train_multi_model.py NIFTY50 shock

Writes:
    db/models/multi/<SECTOR>_<set>_<model>.joblib
    db/processed/<SECTOR>_<set>_model_comparison.csv

Does NOT touch <SECTOR>_anomalies.csv -- that file is still produced by
train_isolation_forest.py and consumed by the RAG explainer.
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from train_isolation_forest import (
    FEATURE_COLS, FINAL_CONTAMINATION, MODELS_DIR, PROCESSED_DIR, annotate_rebalancing
)

EVENTS_DIR = PROCESSED_DIR.parent / "events"
# Kept apart from MODELS_DIR so we never overwrite train_isolation_forest.py's models.
MULTI_MODELS_DIR = MODELS_DIR / "multi"

FEATURE_SETS = {
    "shock": ["log_return", "gap", "range_vol", "return_zscore"],
    # 5-day features are left out of both: tried in either set, they blurred the
    # shock model (more 2020 clustering) without making the regime model cleaner.
    "regime": ["return_20d", "return_60d", "volatility_20d", "volatility_60d",
               "ma_deviation_60d", "rolling_sharpe_20d"],
    "all": FEATURE_COLS,
}

LOF_NEIGHBORS = 50   # large-ish: crash days cluster together (Mar 2020), and a small
                     # neighbourhood lets them vouch for each other as "normal"
KNN_NEIGHBORS = 10
PCA_VARIANCE = 0.90  # keep components explaining 90% of variance

EVENT_WINDOW = 5     # trading days either side of an event that still count as "near" it
EPISODE_GAP = 5      # flagged days at most this many trading days apart form one episode


# Every scorer returns an "anomaly score" where HIGHER = MORE ANOMALOUS, so all
# models can be ranked and compared on the same footing.

def score_isolation_forest(X):
    model = IsolationForest(n_estimators=200, contamination=FINAL_CONTAMINATION,
                            random_state=42).fit(X)
    return model, -model.score_samples(X)


def score_one_class_svm(X):
    # nu ~ upper bound on the fraction of outliers; RBF kernel learns a smooth
    # boundary around the "normal" cloud of days.
    model = make_pipeline(RobustScaler(),
                          OneClassSVM(kernel="rbf", nu=FINAL_CONTAMINATION, gamma="scale"))
    model.fit(X)
    return model, -model.score_samples(X)


def score_lof(X):
    # Compares each day's local density with that of its neighbours.
    scaler = RobustScaler().fit(X)
    model = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=FINAL_CONTAMINATION)
    model.fit(scaler.transform(X))
    return make_pipeline(scaler, model), -model.negative_outlier_factor_


def score_elliptic_envelope(X):
    # Robust (MCD) Mahalanobis distance: how far a day is from the bulk of days,
    # taking feature correlations into account.
    model = make_pipeline(RobustScaler(),
                          EllipticEnvelope(contamination=FINAL_CONTAMINATION,
                                           support_fraction=0.9, random_state=42))
    model.fit(X)
    return model, -model.score_samples(X)


def score_pca_reconstruction(X):
    # Days that don't fit the main correlation patterns between features are
    # reconstructed badly from the top principal components.
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=PCA_VARIANCE, random_state=42).fit(Xs)
    recon = pca.inverse_transform(pca.transform(Xs))
    return make_pipeline(scaler, pca), ((Xs - recon) ** 2).sum(axis=1)


def score_knn_distance(X):
    # Mean distance to the k nearest other days: isolated days are far from everyone.
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS + 1).fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    return make_pipeline(scaler, nn), dist[:, 1:].mean(axis=1)  # column 0 is the day itself


MODELS = {
    "isoforest": score_isolation_forest,
    "ocsvm": score_one_class_svm,
    "lof": score_lof,
    "elliptic": score_elliptic_envelope,
    "pca_recon": score_pca_reconstruction,
    "knn": score_knn_distance,
}

# PCA reconstruction is still trained and reported, but kept out of the ensemble:
# it measures "breaks the usual feature correlations", not "extreme", so it agreed
# least with the other models and caught the fewest known events in every set.
ENSEMBLE_MODELS = [m for m in MODELS if m != "pca_recon"]

# A day counts as a vote anomaly if a majority of the ensemble models flag it.
MIN_VOTES = len(ENSEMBLE_MODELS) // 2 + 1


def run_models(df: pd.DataFrame, sector: str, feature_set: str) -> pd.DataFrame:
    X = df[FEATURE_SETS[feature_set]].to_numpy()
    out = df.copy()

    for name, scorer in MODELS.items():
        model, score = scorer(X)
        out[f"{name}_score"] = score
        # Same rule for every model: flag the top FINAL_CONTAMINATION share of days.
        # (OCSVM's nu is only an upper bound, so its own predict() wouldn't give 5%.)
        out[f"{name}_anomaly"] = score >= np.quantile(score, 1 - FINAL_CONTAMINATION)
        joblib.dump(model, MULTI_MODELS_DIR / f"{sector}_{feature_set}_{name}.joblib")

    flag_cols = [f"{m}_anomaly" for m in ENSEMBLE_MODELS]
    rank_cols = [out[f"{m}_score"].rank(pct=True) for m in ENSEMBLE_MODELS]

    out["votes"] = out[flag_cols].sum(axis=1)
    out["vote_anomaly"] = out["votes"] >= MIN_VOTES

    # Rank averaging: each model's score -> percentile, then average. Scales differ
    # wildly between models (distances vs. log-densities), percentiles don't.
    out["ensemble_score"] = pd.concat(rank_cols, axis=1).mean(axis=1)
    out["ensemble_anomaly"] = out["ensemble_score"] >= out["ensemble_score"].quantile(
        1 - FINAL_CONTAMINATION)

    return out


def near_event(flag: pd.Series, event_pos: list[int]) -> int:
    """How many events have at least one flagged day within EVENT_WINDOW trading days."""
    return sum(flag.iloc[max(0, i - EVENT_WINDOW): i + EVENT_WINDOW + 1].any()
               for i in event_pos)


def episode_runs(flags) -> list[np.ndarray]:
    """Row positions of flagged days, grouped into contiguous episodes
    (flagged days at most EPISODE_GAP trading days apart)."""
    pos = np.flatnonzero(np.asarray(flags))
    if len(pos) == 0:
        return []
    return np.split(pos, np.flatnonzero(np.diff(pos) > EPISODE_GAP) + 1)


def episodes(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """One row per episode of flagged days."""
    rows = []
    for run in episode_runs(df[flag_col]):
        start, end = run[0], run[-1]
        before = df["close"].iloc[max(start - 1, 0)]
        rows.append({
            "start": df["date"].iloc[start].date(),
            "end": df["date"].iloc[end].date(),
            "flagged_days": len(run),
            "move_%": round((df["close"].iloc[end] / before - 1) * 100, 1),
        })
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, events: pd.DataFrame) -> dict:
    events = events[events["date"].isin(df["date"])]
    event_pos = [df.index[df["date"] == d][0] for d in events["date"]]
    systems = list(MODELS) + ["vote", "ensemble"]

    print(f"--- Known events caught on the exact day ({len(events)} in range) ---")
    print(f"{'date':<12}{'event':<32}" + "".join(f"{s[:9]:>10}" for s in systems))
    for (_, ev), i in zip(events.iterrows(), event_pos):
        row = df.loc[i]
        marks = "".join(f"{'X' if row[f'{s}_anomaly'] else '.':>10}" for s in systems)
        print(f"{ev['date'].date()!s:<12}{ev['event'][:31]:<32}{marks}")

    print(f"\n--- Summary ---")
    print(f"{'system':<12}{'flagged':>9}{'exact':>8}{f'±{EVENT_WINDOW}d':>8}"
          f"{'mean|ret|%':>12}{'|z|>2 %':>9}{'2020 share':>12}{'episodes':>10}")
    summary = {}
    for s in systems:
        flagged = df[df[f"{s}_anomaly"]]
        exact = int(df.loc[event_pos, f"{s}_anomaly"].sum())
        near = near_event(df[f"{s}_anomaly"], event_pos)
        summary[s] = {
            "flagged": len(flagged), "exact": exact, "near": near,
            "mean_abs_ret": flagged["log_return"].abs().mean() * 100,
            "share_2020": (flagged["date"].dt.year == 2020).mean() * 100,
            "episodes": len(episodes(df, f"{s}_anomaly")),
        }
        r = summary[s]
        print(f"{s:<12}{r['flagged']:>9}{f'{exact}/{len(events)}':>8}"
              f"{f'{near}/{len(events)}':>8}{r['mean_abs_ret']:>12.2f}"
              f"{(flagged['return_zscore'].abs() > 2).mean() * 100:>9.0f}"
              f"{r['share_2020']:>11.0f}%{r['episodes']:>10}")

    print(f"\n--- Overlap between models (Jaccard of flagged days) ---")
    sets = {m: set(df.loc[df[f"{m}_anomaly"], "date"]) for m in MODELS}
    print(" " * 11 + "".join(f"{m[:9]:>10}" for m in MODELS))
    for a in MODELS:
        print(f"{a:<11}" + "".join(
            f"{len(sets[a] & sets[b]) / len(sets[a] | sets[b]):>10.2f}" for b in MODELS))

    print(f"\n--- Days flagged by k of {len(ENSEMBLE_MODELS)} ensemble models ---")
    print(df["votes"].value_counts().sort_index().to_string())

    return summary


def run_feature_set(df: pd.DataFrame, sector: str, feature_set: str):
    print(f"\n{'=' * 100}\n=== {sector} / {feature_set}: {FEATURE_SETS[feature_set]}\n"
          f"=== {len(df)} days, contamination={FINAL_CONTAMINATION}, "
          f"ensemble={ENSEMBLE_MODELS}, vote threshold={MIN_VOTES}\n")

    df = run_models(df, sector, feature_set)
    df = annotate_rebalancing(df, sector)

    summary = None
    events_path = EVENTS_DIR / f"{sector}_known_events.csv"
    if events_path.exists():
        summary = evaluate(df, pd.read_csv(events_path, parse_dates=["date"]))
    else:
        print(f"(no {events_path.name}; skipping evaluation)")

    if feature_set == "regime":
        print(f"\n--- Ensemble-flagged regime episodes ---")
        print(episodes(df, "ensemble_anomaly").to_string(index=False))

    out_cols = (["date", "close", "log_return", "return_zscore", "return_60d"]
                + [f"{m}_{k}" for m in MODELS for k in ("score", "anomaly")]
                + ["votes", "vote_anomaly", "ensemble_score", "ensemble_anomaly",
                   "near_rebalancing", "rebalancing_notes"])
    out_path = PROCESSED_DIR / f"{sector}_{feature_set}_model_comparison.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return summary


if __name__ == "__main__":
    SECTOR = sys.argv[1] if len(sys.argv) > 1 else "NIFTY50"
    sets_to_run = [sys.argv[2]] if len(sys.argv) > 2 else list(FEATURE_SETS)
    MULTI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / f"{SECTOR}_features.csv", parse_dates=["date"])
    summaries = {fs: run_feature_set(df, SECTOR, fs) for fs in sets_to_run}

    if len(summaries) > 1 and all(summaries.values()):
        print(f"\n{'=' * 100}\n=== Feature sets compared (ensemble = rank average) ===")
        print(f"{'set':<8}{'exact':>7}{f'±{EVENT_WINDOW}d':>7}{'mean|ret|%':>12}"
              f"{'2020 share':>12}{'episodes':>10}   per-model exact catches")
        for fs, s in summaries.items():
            e = s["ensemble"]
            per_model = " ".join(f"{m[:5]}={s[m]['exact']}" for m in MODELS)
            print(f"{fs:<8}{e['exact']:>7}{e['near']:>7}{e['mean_abs_ret']:>12.2f}"
                  f"{e['share_2020']:>11.0f}%{e['episodes']:>10}   {per_model}")
