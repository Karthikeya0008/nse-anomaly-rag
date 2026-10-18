"""Combine the shock and regime ensembles into one labelled anomaly file.

Each day gets anomaly_type:
    shock   -- the day itself was extreme (single-day features)
    regime  -- the day sits inside an unusual 20-60 day period
    both    -- both at once (e.g. the March 2020 crash days)
    ""      -- normal

Run train_multi_model.py first (it writes the shock/regime comparison CSVs).

Usage:
    python build_final_anomalies.py            # NIFTY50
    python build_final_anomalies.py NIFTYBANK

Writes:
    db/processed/<SECTOR>_final_anomalies.csv   (read by rag/explain_anomalies.py)
"""
import sys

import numpy as np
import pandas as pd

from train_multi_model import PROCESSED_DIR, episode_runs

KEEP_COLS = ["date", "close", "log_return", "return_zscore", "return_60d"]
ENSEMBLE_COLS = ["ensemble_score", "votes", "ensemble_anomaly"]


def load_ensemble(sector: str, feature_set: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{sector}_{feature_set}_model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run train_multi_model.py {sector} first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.rename(columns={
        "ensemble_score": f"{feature_set}_score",
        "votes": f"{feature_set}_votes",
        "ensemble_anomaly": f"{feature_set}_anomaly",
    })


def add_regime_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each regime day with the start/end of the episode it belongs to, so the
    explainer can treat an 80-day episode as one event instead of 80."""
    df["regime_episode_start"] = pd.NaT
    df["regime_episode_end"] = pd.NaT
    for run in episode_runs(df["regime_anomaly"]):
        rows = df.index[run]
        df.loc[rows, "regime_episode_start"] = df["date"].iloc[run[0]]
        df.loc[rows, "regime_episode_end"] = df["date"].iloc[run[-1]]
    return df


def build(sector: str) -> pd.DataFrame:
    shock = load_ensemble(sector, "shock")
    regime = load_ensemble(sector, "regime")

    df = shock[KEEP_COLS + ["shock_score", "shock_votes", "shock_anomaly"]].merge(
        regime[["date", "regime_score", "regime_votes", "regime_anomaly",
                "near_rebalancing", "rebalancing_notes"]],
        on="date", validate="one_to_one")

    df = add_regime_episodes(df)
    df["anomaly_type"] = np.select(
        [df["shock_anomaly"] & df["regime_anomaly"], df["shock_anomaly"], df["regime_anomaly"]],
        ["both", "shock", "regime"], default="")
    df["anomaly"] = df["anomaly_type"] != ""
    df["rebalancing_notes"] = df["rebalancing_notes"].fillna("")

    dp_cols = add_dp_verdicts(df, sector)

    return df[KEEP_COLS + [
        "anomaly", "anomaly_type",
        "shock_score", "shock_votes", "shock_anomaly",
        "regime_score", "regime_votes", "regime_anomaly",
        "regime_episode_start", "regime_episode_end",
        "near_rebalancing", "rebalancing_notes",
        *dp_cols,
    ]]


def add_dp_verdicts(df: pd.DataFrame, sector: str) -> list[str]:
    """Verdicts of the differentially private ensembles (privacy/train_dp_models.py),
    if they've been trained -- shown alongside the production labels, not used for them."""
    cols = []
    for fs in ("shock", "regime"):
        path = PROCESSED_DIR / f"{sector}_{fs}_dp_comparison.csv"
        if not path.exists():
            continue
        dp = pd.read_csv(path, parse_dates=["date"])[["date", "dp_anomaly", "dp_votes"]]
        dp.columns = ["date", f"dp_{fs}_anomaly", f"dp_{fs}_votes"]
        merged = df[["date"]].merge(dp, on="date", how="left", validate="one_to_one")
        df[f"dp_{fs}_anomaly"] = merged[f"dp_{fs}_anomaly"].fillna(False).astype(bool).to_numpy()
        df[f"dp_{fs}_votes"] = merged[f"dp_{fs}_votes"].fillna(0).astype(int).to_numpy()
        cols += [f"dp_{fs}_anomaly", f"dp_{fs}_votes"]
    return cols


if __name__ == "__main__":
    SECTOR = sys.argv[1] if len(sys.argv) > 1 else "NIFTY50"
    df = build(SECTOR)

    out_path = PROCESSED_DIR / f"{SECTOR}_final_anomalies.csv"
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")

    counts = df["anomaly_type"].replace("", "normal").value_counts()
    print(f"{SECTOR}: {len(df)} days")
    for t in ["shock", "regime", "both", "normal"]:
        print(f"  {t:<7}{counts.get(t, 0):>6}")
    n_episodes = df.loc[df["regime_anomaly"], "regime_episode_start"].nunique()
    print(f"  regime days fall into {n_episodes} episodes")
    print(f"Wrote {out_path}")
