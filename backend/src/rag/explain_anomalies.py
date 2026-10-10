"""
Anomaly Explanation Pipeline (RAG + LLM) -- Balanced Retrieval Version
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import pandas as pd
from google import genai


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "db" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "db" / "vectorstore"
EXPLANATIONS_DIR = PROJECT_ROOT / "db" / "explanations"


# ============================================================================
# CONFIGURATION
# ============================================================================

COLLECTION_NAME = "market_context"
GEMINI_MODEL = "gemini-3.6-flash"

DATE_WINDOW_DAYS = 5

CHUNKS_PER_SOURCE = {
    "sebi": 2,
    "business_standard": 2,
    "wikipedia": 2,
}

DELAY_BETWEEN_CALLS = 7
MAX_ANOMALIES_PER_RUN = 30
SECTOR = "NIFTY50"
MIN_ANOMALY_DATE = "2023-01-01"


# ============================================================================
# GEMINI
# ============================================================================

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set.\n\n"
            "In PowerShell run:\n"
            '$env:GEMINI_API_KEY = "your-key-here"'
        )
    return genai.Client(api_key=api_key)


# ============================================================================
# CHROMA
# ============================================================================

def get_chroma_collection():
    print("Chroma configuration:")
    print(f"  Project root:       {PROJECT_ROOT.resolve()}")
    print(f"  Vectorstore path:   {VECTORSTORE_DIR.resolve()}")
    print(f"  Vectorstore exists: {VECTORSTORE_DIR.exists()}")
    print(f"  Collection:         {COLLECTION_NAME}")
    print()

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Chroma vectorstore directory does not exist:\n{VECTORSTORE_DIR}\n\n"
            "Run ingest_all_to_chroma.py first."
        )

    chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collections = chroma_client.list_collections()
    collection_names = [getattr(c, "name", str(c)) for c in collections]

    print("Available Chroma collections:")
    for name in collection_names or ["(none)"]:
        print(f"  - {name}")
    print()

    if COLLECTION_NAME not in collection_names:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' does not exist.\n"
            "Make sure ingest_all_to_chroma.py completed successfully."
        )

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Connected to '{COLLECTION_NAME}' with {collection.count()} chunks.\n")
    return collection


# ============================================================================
# DATE HELPERS
# ============================================================================

def normalize_date(value) -> str:
    if pd.isna(value):
        raise ValueError("Anomaly date is missing.")
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    value = str(value).strip()
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    try:
        return pd.to_datetime(value, errors="raise").strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"Could not parse date: {value!r}") from exc


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# BALANCED RETRIEVAL
# ============================================================================

def _query_source(collection, query_text: str, source: str, k: int, date_range):
    if collection.count() == 0:
        return []

    where_conditions = [{"source": {"$eq": source}}]
    if date_range is not None:
        start, end = date_range
        where_conditions.append({"date_int": {"$gte": start}})
        where_conditions.append({"date_int": {"$lte": end}})

    where = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

    try:
        results = collection.query(query_texts=[query_text], n_results=k, where=where)
    except Exception:
        return []

    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return []

    metadatas = results.get("metadatas", [[]])[0] or []

    chunks = []
    for doc, meta in zip(documents[0], metadatas):
        meta = meta or {}
        chunks.append({
            "text": doc,
            "source": meta.get("source", source),
            "date": meta.get("date", "unknown"),
            "title": meta.get("title", "untitled"),
            "url": meta.get("url", ""),
        })
    return chunks


def retrieve_balanced_context(collection, anomaly_date: str, sector: str):
    anomaly_date = normalize_date(anomaly_date)
    dt = datetime.strptime(anomaly_date, "%Y-%m-%d")
    start = int((dt - timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y%m%d"))
    end = int((dt + timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y%m%d"))

    results = {}
    event_query = f"{sector} index price movement {anomaly_date} market conditions"

    results["sebi"] = _query_source(collection, event_query, "sebi", CHUNKS_PER_SOURCE["sebi"], (start, end))
    results["business_standard"] = _query_source(
        collection, event_query, "business_standard", CHUNKS_PER_SOURCE["business_standard"], (start, end)
    )

    background_query = f"{sector} Indian stock market crash crisis regulation history"
    results["wikipedia"] = _query_source(
        collection, background_query, "wikipedia", CHUNKS_PER_SOURCE["wikipedia"], None
    )

    return results


# ============================================================================
# PROMPT
# ============================================================================

def _format_source_block(chunks: list, label: str, is_point_in_time: bool) -> str:
    if not chunks:
        return f"### {label}\n(No relevant {label.lower()} found for this anomaly.)"

    framing = "point-in-time" if is_point_in_time else "general background (not necessarily from this specific date)"
    lines = [f"### {label}  [{framing}]"]
    for c in chunks:
        lines.append(f"\n[{c['date']} | {c['title']}]\n{c['text'][:800]}")
    return "\n".join(lines)


def build_prompt(anomaly_row: pd.Series, sector: str, context_by_source: dict) -> str:
    anomaly_date = normalize_date(anomaly_row["date"])
    close = anomaly_row.get("close", "N/A")
    log_return = safe_float(anomaly_row.get("log_return", 0.0))
    return_zscore = safe_float(anomaly_row.get("return_zscore", 0.0))
    return_60d = anomaly_row.get("return_60d", "N/A")
    near_rebalancing = anomaly_row.get("near_rebalancing", "N/A")

    regulatory_block = _format_source_block(context_by_source.get("sebi", []), "Regulatory Filings (SEBI)", True)
    news_block = _format_source_block(context_by_source.get("business_standard", []), "News Coverage (Business Standard)", True)
    background_block = _format_source_block(context_by_source.get("wikipedia", []), "Background Reference (Wikipedia)", False)

    return f"""You are analyzing an anomalous trading day flagged by an Isolation Forest model on the {sector} index.

ANOMALY DETAILS
- Date: {anomaly_date}
- Closing price: {close}
- Daily log return: {log_return:.4f}
- Return z-score (vs 20-day mean/std): {return_zscore:.2f}
- 60-day cumulative return: {return_60d}
- Near a known index rebalancing event: {near_rebalancing}

RETRIEVED CONTEXT

{regulatory_block}

{news_block}

{background_block}

INSTRUCTIONS
Regulatory Filings and News Coverage above are dated near the anomaly and are your primary evidence. Background Reference is general knowledge that may or may not relate to this specific date -- use it only for context, never as evidence of what happened on {anomaly_date} specifically.

Respond in exactly this structure:

**Likely Cause:** One or two sentences stating the most plausible explanation.

**Supporting Evidence:** Which specific retrieved item(s) support this, citing source type (Regulatory/News/Background) and date. If no retrieved item clearly supports a cause, say so explicitly instead of guessing.

**Confidence:** High / Medium / Low, based on how directly the retrieved context (not general market knowledge) explains this specific anomaly.

**Caveats:** Anything retrieved context does not explain, or where you are inferring rather than citing.

Do not invent news events, regulatory actions, or causes not present in the retrieved context above."""


# ============================================================================
# EXPLANATION GENERATION
# ============================================================================

def explain_anomaly(client, collection, anomaly_row: pd.Series, sector: str) -> dict:
    anomaly_date = normalize_date(anomaly_row["date"])
    context_by_source = retrieve_balanced_context(collection, anomaly_date, sector)
    prompt = build_prompt(anomaly_row, sector, context_by_source)

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    explanation = getattr(response, "text", None) or "Gemini returned no explanation text."

    all_chunks = [c for chunks in context_by_source.values() for c in chunks]

    return {
        "date": anomaly_date,
        "sector": sector,
        "log_return": safe_float(anomaly_row.get("log_return", 0.0)),
        "return_zscore": safe_float(anomaly_row.get("return_zscore", 0.0)),
        "chunks_per_source": {src: len(chunks) for src, chunks in context_by_source.items()},
        "context_titles": [c["title"] for c in all_chunks],
        "context_urls": [c["url"] for c in all_chunks if c.get("url")],
        "explanation": explanation.strip(),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run():
    print("=" * 70)
    print("ANOMALY EXPLANATION PIPELINE (balanced retrieval)")
    print("=" * 70)
    print(f"Sector:       {SECTOR}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print()

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Processed directory does not exist:\n{PROCESSED_DIR}")
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"Vectorstore directory does not exist:\n{VECTORSTORE_DIR}\n\nRun ingest_all_to_chroma.py first.")
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

    collection = get_chroma_collection()
    client = get_gemini_client()

    anomalies_path = PROCESSED_DIR / f"{SECTOR}_anomalies.csv"
    if not anomalies_path.exists():
        raise FileNotFoundError(f"Anomaly CSV does not exist:\n{anomalies_path}")

    # FIX: encoding="utf-8-sig" strips a BOM if present, which otherwise
    # makes pandas read the first column as "\ufeffdate" instead of "date".
    
    df = pd.read_csv(anomalies_path, header=None, encoding="utf-8-sig")

    df.columns = [
        "date",
        "close",
        "log_return",
        "return_zscore",
        "return_60d",
        "return_60d_zscore",
        "near_rebalancing",
        "volatility_20d",
        "anomaly",
        "unused",
    ]

    print(f"Columns found in CSV: {list(df.columns)}\n")

    missing_columns = [c for c in ["date", "anomaly"] if c not in df.columns]
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_columns) +
            f"\nActual columns present: {list(df.columns)}"
        )

    anomaly_mask = df["anomaly"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    flagged = df[anomaly_mask].copy()

    date_mask = flagged["date"].astype(str) >= MIN_ANOMALY_DATE
    flagged = flagged[date_mask].copy()
    print(f"Filtering to anomalies on or after {MIN_ANOMALY_DATE}")

    out_path = EXPLANATIONS_DIR / f"{SECTOR}_explanations.json"
    existing = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                for entry in existing_data:
                    if "date" in entry:
                        try:
                            existing[normalize_date(entry["date"])] = entry
                        except ValueError:
                            continue
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not load existing explanations: {exc}\n")

    print(f"{len(flagged)} flagged anomalies for {SECTOR} (from {MIN_ANOMALY_DATE})")
    print(f"{len(existing)} already explained")
    print(f"Processing up to {MAX_ANOMALIES_PER_RUN} new anomalies this run.\n")

    results = list(existing.values())
    processed_this_run = 0

    for _, row in flagged.iterrows():
        try:
            anomaly_date = normalize_date(row["date"])
        except ValueError as exc:
            print(f"Skipping invalid date: {exc}")
            continue

        if anomaly_date in existing:
            continue
        if processed_this_run >= MAX_ANOMALIES_PER_RUN:
            print(f"Reached MAX_ANOMALIES_PER_RUN ({MAX_ANOMALIES_PER_RUN}). Rerun to continue.")
            break

        print(f"[{processed_this_run + 1}/{MAX_ANOMALIES_PER_RUN}] Processing {anomaly_date}...")

        try:
            result = explain_anomaly(client, collection, row, SECTOR)
            results.append(result)
            existing[anomaly_date] = result
            processed_this_run += 1

            counts = result["chunks_per_source"]
            print(f"  Chunks used -- SEBI: {counts.get('sebi', 0)}, "
                  f"News: {counts.get('business_standard', 0)}, "
                  f"Wikipedia: {counts.get('wikipedia', 0)}")
            print(f"  EXPLANATION:\n{result['explanation']}\n")

        except Exception as exc:
            print(f"  FAILED: {exc}")

        if processed_this_run < MAX_ANOMALIES_PER_RUN:
            time.sleep(DELAY_BETWEEN_CALLS)
        print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"New explanations generated this run: {processed_this_run}")
    print(f"Total explanations on disk: {len(results)}")
    print(f"Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    run()