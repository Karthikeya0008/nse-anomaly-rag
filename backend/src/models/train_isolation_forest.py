import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "db" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[3] / "db" / "models"
REBALANCING_DIR = Path(__file__).resolve().parents[3] / "db" / "rebalancing"

SECTORS = [
    "NIFTY50", "NIFTYBANK", "NIFTYPHARMA", "NIFTYIT",
    "NIFTYAUTO", "NIFTYFMCG", "NIFTYREALTY"
]

FEATURE_COLS = [
    "log_return", "return_5d", "return_20d", "return_60d",
    "volatility_5d", "volatility_20d", "volatility_60d",
    "range_vol", "return_zscore", "gap",
    "ma_deviation_60d", "rolling_sharpe_20d"
]

FINAL_CONTAMINATION = 0.05  # locked in after sweep: caught 8/9 known events

# How many calendar days on either side of a rebalancing event date counts as
# "near" it, for annotation purposes. 3 is a reasonable starting window -- tune
# later if you find it's too wide/narrow once you inspect results.
REBALANCING_WINDOW_DAYS = 3


def train_sector_model(sector: str, contamination: float):
    df = pd.read_csv(PROCESSED_DIR / f"{sector}_features.csv", parse_dates=["date"])
    X = df[FEATURE_COLS]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    model.fit(X)

    df["anomaly"] = model.predict(X) == -1
    df["anomaly_score"] = model.decision_function(X)

    return model, df


def annotate_rebalancing(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Flag anomalies that fall within REBALANCING_WINDOW_DAYS of a known/scheduled
    index rebalancing event. Does not alter the anomaly flag itself -- only adds
    context columns so you can distinguish 'possible rebalancing effect' from
    'likely genuine market shock' during interpretation."""
    reb_path = REBALANCING_DIR / f"{sector}_rebalancing_events.csv"
    if not reb_path.exists():
        df = df.copy()
        df["near_rebalancing"] = False
        df["rebalancing_notes"] = ""
        return df

    reb = pd.read_csv(reb_path, parse_dates=["date"])

    df = df.copy()
    df["near_rebalancing"] = False
    df["rebalancing_notes"] = ""

    for _, event in reb.iterrows():
        mask = (df["date"] >= event["date"] - pd.Timedelta(days=REBALANCING_WINDOW_DAYS)) & \
               (df["date"] <= event["date"] + pd.Timedelta(days=REBALANCING_WINDOW_DAYS))
        df.loc[mask, "near_rebalancing"] = True
        note = f"{event['event_type']}: {event['notes']}"
        df.loc[mask, "rebalancing_notes"] = df.loc[mask, "rebalancing_notes"].apply(
            lambda existing, n=note: (existing + " | " + n) if existing else n
        )

    return df


def score_contamination(df: pd.DataFrame, known_events) -> tuple[int, int]:
    flagged_dates = set(df.loc[df["anomaly"], "date"])
    events_in_range = [d for d in known_events if d in set(df["date"])]
    caught = sum(1 for d in events_in_range if d in flagged_dates)
    return caught, len(events_in_range)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Training models with locked contamination={FINAL_CONTAMINATION} ===\n")

    for sector in SECTORS:
        model, df = train_sector_model(sector, contamination=FINAL_CONTAMINATION)
        df = annotate_rebalancing(df, sector)

        n_anomalies = df["anomaly"].sum()
        n_near_reb = (df["anomaly"] & df["near_rebalancing"]).sum()
        print(f"{sector}: {n_anomalies} anomalies flagged out of {len(df)} days "
              f"({n_anomalies/len(df)*100:.1f}%) -- {n_near_reb} of those fall near a "
              f"rebalancing event")

        joblib.dump(model, MODELS_DIR / f"{sector}_isoforest.joblib")

        out_cols = ["date", "close", "log_return", "return_zscore", "return_60d",
                    "rolling_sharpe_20d", "anomaly", "anomaly_score",
                    "near_rebalancing", "rebalancing_notes"]
        df[out_cols].to_csv(PROCESSED_DIR / f"{sector}_anomalies.csv", index=False)

        top5 = df.nsmallest(5, "anomaly_score")[
            ["date", "close", "log_return", "return_zscore", "near_rebalancing"]
        ]
        print(top5.to_string(index=False))
        print()
        
        
        