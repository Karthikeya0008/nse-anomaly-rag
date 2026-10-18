"""Bundle everything the dashboard needs into one JSON file.

Reads:
    db/processed/<SECTOR>_final_anomalies.csv      (models/build_final_anomalies.py)
    db/explanations/<SECTOR>_explanations.json     (rag/explain_anomalies.py, optional)
    db/events/<SECTOR>_known_events.csv            (optional)

Writes:
    frontend/data/<SECTOR>.json

server.py serves the same bundle live from /api/data, so this file is only
needed when the page is opened without the server (plain static hosting).

Usage:
    python export_frontend_data.py            # NIFTY50
    python export_frontend_data.py NIFTYBANK
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "db" / "processed"
EXPLANATIONS_DIR = PROJECT_ROOT / "db" / "explanations"
EVENTS_DIR = PROJECT_ROOT / "db" / "events"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "data"


def format_explanation(entry: dict) -> dict:
    """The parts of an explain_anomalies.py entry the page shows."""
    context = entry.get("context") or [{"title": t} for t in entry.get("context_titles", [])]
    return {
        "explanation": entry.get("explanation", ""),
        "context": context,
        "urls": entry.get("context_urls", []),
        "notes": entry.get("retrieval_notes", []),
    }


def load_explanations(sector: str) -> dict:
    path = EXPLANATIONS_DIR / f"{sector}_explanations.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["date"]: format_explanation(e) for e in entries}


def explanation_date(row, explanations: dict, episode_explained: dict):
    """Which explanation belongs to this day. The explainer explains each shock/both
    day itself, but only one representative day per regime episode -- so every
    regime-only day in an episode shares that episode's explanation."""
    if row["date"] in explanations:
        return row["date"]
    if row["anomaly_type"] == "regime":
        return episode_explained.get(row["regime_episode_start"])
    return None


def build(sector: str) -> dict:
    df = pd.read_csv(PROCESSED_DIR / f"{sector}_final_anomalies.csv", keep_default_na=False)
    explanations = load_explanations(sector)

    anomalies = df[df["anomaly"].astype(str) == "True"].copy()
    regime_only = anomalies[anomalies["anomaly_type"] == "regime"]
    episode_explained = {
        r["regime_episode_start"]: r["date"]
        for _, r in regime_only.iterrows() if r["date"] in explanations
    }
    anomalies["explanation_date"] = anomalies.apply(
        explanation_date, axis=1, explanations=explanations, episode_explained=episode_explained)

    events = {}
    events_path = EVENTS_DIR / f"{sector}_known_events.csv"
    if events_path.exists():
        ev = pd.read_csv(events_path)
        events = {r["date"]: {"event": r["event"], "notes": r["notes"]} for _, r in ev.iterrows()}

    episodes = (anomalies.loc[anomalies["regime_episode_start"] != "",
                              ["regime_episode_start", "regime_episode_end"]]
                .drop_duplicates()
                .rename(columns={"regime_episode_start": "start", "regime_episode_end": "end"}))

    anomaly_cols = ["date", "close", "log_return", "return_zscore", "return_60d", "anomaly_type",
                    "shock_votes", "regime_votes", "shock_score", "regime_score",
                    "regime_episode_start", "regime_episode_end", "near_rebalancing",
                    "rebalancing_notes", "explanation_date"]
    records = anomalies[anomaly_cols].to_dict(orient="records")
    for r in records:
        r["near_rebalancing"] = str(r["near_rebalancing"]) == "True"
        r["event"] = events.get(r["date"])
        # Empty strings -> null; also catches the NaN pandas puts in explanation_date
        # once some (but not all) days have an explanation -- NaN is not valid JSON.
        for k in ("regime_episode_start", "regime_episode_end", "rebalancing_notes",
                  "explanation_date"):
            r[k] = r[k] if isinstance(r[k], str) and r[k] else None

    return {
        "sector": sector,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ensemble_size": 5,
        "contamination": 0.05,
        "prices": {
            "date": df["date"].tolist(),
            "close": df["close"].round(2).tolist(),
            "log_return": df["log_return"].round(6).tolist(),
        },
        "anomalies": records,
        "episodes": episodes.to_dict(orient="records"),
        "explanations": explanations,
    }


if __name__ == "__main__":
    SECTOR = sys.argv[1] if len(sys.argv) > 1 else "NIFTY50"
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = build(SECTOR)
    out_path = FRONTEND_DATA_DIR / f"{SECTOR}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    n_explained = sum(1 for a in data["anomalies"] if a["explanation_date"])
    print(f"{SECTOR}: {len(data['prices']['date'])} days, {len(data['anomalies'])} anomalies, "
          f"{len(data['episodes'])} regime episodes, {n_explained} anomalies with an explanation")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
