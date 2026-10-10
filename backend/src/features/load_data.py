import pandas as pd
from pathlib import Path

# Paths — adjust if your root differs
RAW_DIR = Path(__file__).resolve().parents[3] / "db" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "db" / "processed"

SECTORS = [
    "NIFTY50", "NIFTYBANK", "NIFTYPHARMA", "NIFTYIT",
    "NIFTYAUTO", "NIFTYFMCG", "NIFTYREALTY"
]

def load_sector(sector: str) -> pd.DataFrame:
    """Load and merge all yearly CSVs for one sector into a single clean dataframe."""
    sector_dir = RAW_DIR / sector
    csv_files = sorted(sector_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {sector_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Clean column names (strip quotes/spaces, lowercase)
    merged.columns = [c.strip().lower().replace(" ", "_") for c in merged.columns]

    # Parse date — format is like "01 Aug 2017"
    merged["date"] = pd.to_datetime(merged["date"], format="%d %b %Y")

    # Sort chronologically and drop duplicates (year files can overlap by 1 day)
    merged = merged.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

    # Keep only what we need
    merged = merged[["date", "open", "high", "low", "close"]]

    return merged


def load_all_sectors() -> dict[str, pd.DataFrame]:
    """Load all 7 sectors into a dict of dataframes."""
    return {sector: load_sector(sector) for sector in SECTORS}


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_sectors()

    for sector, df in data.items():
        print(f"{sector}: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
        out_path = PROCESSED_DIR / f"{sector}_merged.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> saved to {out_path}")
        
