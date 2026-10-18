"""Train several unsupervised anomaly detectors on one sector's features and
combine them into an ensemble.

Usage:
    python train_multi_model.py            # NIFTY50
    python train_multi_model.py NIFTYBANK

Writes:
    db/models/multi/<SECTOR>_<model>.joblib
    db/processed/<SECTOR>_model_comparison.csv

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

# A day counts as an ensemble anomaly if at least this many of the models flag it.
MIN_VOTES = 4

LOF_NEIGHBORS = 50   # large-ish: crash days cluster together (Mar 2020), and a small
                     # neighbourhood lets them vouch for each other as "normal"
KNN_NEIGHBORS = 10
PCA_VARIANCE = 0.90  # keep components explaining 90% of variance


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


def run_models(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    X = df[FEATURE_COLS].to_numpy()
    out = df.copy()

    for name, scorer in MODELS.items():
        model, score = scorer(X)
        out[f"{name}_score"] = score
        # Same rule for every model: flag the top FINAL_CONTAMINATION share of days.
        # (OCSVM's nu is only an upper bound, so its own predict() wouldn't give 5%.)
        out[f"{name}_anomaly"] = score >= np.quantile(score, 1 - FINAL_CONTAMINATION)
        joblib.dump(model, MULTI_MODELS_DIR / f"{sector}_{name}.joblib")

    flag_cols = [f"{m}_anomaly" for m in MODELS]
    rank_cols = [out[f"{m}_score"].rank(pct=True) for m in MODELS]

    out["votes"] = out[flag_cols].sum(axis=1)
    out["vote_anomaly"] = out["votes"] >= MIN_VOTES

    # Rank averaging: each model's score -> percentile, then average. Scales differ
    # wildly between models (distances vs. log-densities), percentiles don't.
    out["ensemble_score"] = pd.concat(rank_cols, axis=1).mean(axis=1)
    out["ensemble_anomaly"] = out["ensemble_score"] >= out["ensemble_score"].quantile(
        1 - FINAL_CONTAMINATION)

    return out


def evaluate(df: pd.DataFrame, events: pd.DataFrame) -> None:
    events = events[events["date"].isin(df["date"])]
    by_date = df.set_index("date")
    systems = list(MODELS) + ["vote", "ensemble"]

    print(f"--- Known events caught ({len(events)} in range) ---")
    header = f"{'date':<12}{'event':<32}" + "".join(f"{s[:9]:>10}" for s in systems)
    print(header)
    for _, ev in events.iterrows():
        row = by_date.loc[ev["date"]]
        marks = "".join(f"{'X' if row[f'{s}_anomaly'] else '.':>10}" for s in systems)
        print(f"{ev['date'].date()!s:<12}{ev['event'][:31]:<32}{marks}")

    print(f"\n--- Summary ---")
    print(f"{'system':<12}{'flagged':>9}{'events':>9}{'mean|ret|%':>12}"
          f"{'|z|>2 %':>9}{'2020 share':>12}")
    for s in systems:
        flagged = df[df[f"{s}_anomaly"]]
        caught = by_date.loc[events["date"], f"{s}_anomaly"].sum()
        print(f"{s:<12}{len(flagged):>9}{f'{caught}/{len(events)}':>9}"
              f"{flagged['log_return'].abs().mean() * 100:>12.2f}"
              f"{(flagged['return_zscore'].abs() > 2).mean() * 100:>9.0f}"
              f"{(flagged['date'].dt.year == 2020).mean() * 100:>11.0f}%")

    print(f"\n--- Overlap between models (Jaccard of flagged days) ---")
    sets = {m: set(df.loc[df[f"{m}_anomaly"], "date"]) for m in MODELS}
    print(" " * 11 + "".join(f"{m[:9]:>10}" for m in MODELS))
    for a in MODELS:
        print(f"{a:<11}" + "".join(
            f"{len(sets[a] & sets[b]) / len(sets[a] | sets[b]):>10.2f}" for b in MODELS))

    print(f"\n--- Days flagged by k models ---")
    print(df["votes"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    SECTOR = sys.argv[1] if len(sys.argv) > 1 else "NIFTY50"
    MULTI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / f"{SECTOR}_features.csv", parse_dates=["date"])
    print(f"=== {SECTOR}: {len(df)} days, {len(MODELS)} models, "
          f"contamination={FINAL_CONTAMINATION}, vote threshold={MIN_VOTES} ===\n")

    df = run_models(df, SECTOR)
    df = annotate_rebalancing(df, SECTOR)

    events_path = EVENTS_DIR / f"{SECTOR}_known_events.csv"
    if events_path.exists():
        evaluate(df, pd.read_csv(events_path, parse_dates=["date"]))
    else:
        print(f"(no {events_path.name}; skipping evaluation)")

    out_cols = (["date", "close", "log_return", "return_zscore", "return_60d"]
                + [f"{m}_{k}" for m in MODELS for k in ("score", "anomaly")]
                + ["votes", "vote_anomaly", "ensemble_score", "ensemble_anomaly",
                   "near_rebalancing", "rebalancing_notes"])
    out_path = PROCESSED_DIR / f"{SECTOR}_model_comparison.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
