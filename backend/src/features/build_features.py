import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "db" / "processed"

SECTORS = [
    "NIFTY50", "NIFTYBANK", "NIFTYPHARMA", "NIFTYIT",
    "NIFTYAUTO", "NIFTYFMCG", "NIFTYREALTY"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # --- Point-level (shock) features ---

    # Daily log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling returns
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # Rolling volatility (std dev of daily log returns)
    df["volatility_5d"] = df["log_return"].rolling(5).std()
    df["volatility_20d"] = df["log_return"].rolling(20).std()

    # Range-based volatility proxy
    df["range_vol"] = (df["high"] - df["low"]) / df["close"]

    # Return z-score (vs its own 20-day rolling mean/std) -- shock signal
    roll_mean_20 = df["log_return"].rolling(20).mean()
    roll_std_20 = df["log_return"].rolling(20).std()
    df["return_zscore"] = (df["log_return"] - roll_mean_20) / roll_std_20

    # Overnight gap
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # --- Period-level (trend/regime) features ---

    # 60-day rolling return -- captures sustained drift, not just a single shock day
    df["return_60d"] = df["close"].pct_change(60)

    # 60-day rolling volatility -- regime-level risk level
    df["volatility_60d"] = df["log_return"].rolling(60).std()

    # Price deviation from its own 60-day moving average
    # Positive = trading above trend (momentum up), negative = below trend (momentum down)
    ma_60 = df["close"].rolling(60).mean()
    df["ma_deviation_60d"] = (df["close"] - ma_60) / ma_60

    # Rolling Sharpe-like ratio (20-day mean return / 20-day std)
    # Distinguishes a strong, consistent trend (high value) from a choppy, directionless
    # period with similar volatility (near-zero value) -- same volatility can mean very
    # different things depending on whether returns are consistently one-directional.
    df["rolling_sharpe_20d"] = roll_mean_20 / roll_std_20

    # Drop initial rows with NaNs from the largest rolling window (60-day)
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for sector in SECTORS:
        in_path = PROCESSED_DIR / f"{sector}_merged.csv"
        df = pd.read_csv(in_path, parse_dates=["date"])

        featured = build_features(df)

        out_path = PROCESSED_DIR / f"{sector}_features.csv"
        featured.to_csv(out_path, index=False)

        print(f"{sector}: {len(featured)} rows after feature engineering -> {out_path}")
        print(featured[["date", "log_return", "volatility_20d", "return_zscore",
                         "return_60d", "ma_deviation_60d", "rolling_sharpe_20d"]].tail(3))
        print()
        