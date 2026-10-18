"""
Compare the original retrieval (v1) with the anomaly-aware pipeline (v2).

No language model is involved: this measures only what context each version
would hand to the explainer.

  * Known events (db/events/<SECTOR>_known_events.csv, `keywords` column):
    does any retrieved chunk mention the actual cause?
  * All flagged days: how much context, how much of it is about the market,
    how close in date, how often the same article repeats, and how often
    Wikipedia is used / is about an unrelated period.

Usage:
    python evaluate_retrieval.py            # NIFTY50
Writes:
    db/validation/<SECTOR>_retrieval_eval.md
"""

import re
import sys
from datetime import datetime, timedelta

import pandas as pd

import explain_anomalies as ex
import retrieval as rv

VALIDATION_DIR = ex.PROJECT_ROOT / "db" / "validation"
EVENTS_DIR = ex.PROJECT_ROOT / "db" / "events"

# A Wikipedia chunk counts as "unrelated period" if its article is dated further
# than this from the anomaly.
OFF_PERIOD_DAYS = 90


# ----------------------------------------------------------------------------
# v1: the original retrieval, reproduced over the same in-memory index so both
# versions search exactly the same chunks.
# ----------------------------------------------------------------------------

_EMBED_CACHE: dict = {}


def embed_one(index: rv.ContextIndex, query: str):
    if query not in _EMBED_CACHE:
        _EMBED_CACHE[query] = rv.np.asarray(index.embed([query]), dtype=rv.np.float32)
    return _EMBED_CACHE[query]


def retrieve_v1(index: rv.ContextIndex, row, sector: str) -> dict:
    day = datetime.strptime(str(row["date"]), "%Y-%m-%d")
    lo = int((day - timedelta(days=5)).strftime("%Y%m%d"))
    hi = int((day + timedelta(days=5)).strftime("%Y%m%d"))

    def top(query, source, k, start=None, end=None):
        hits = index.search(embed_one(index, query), source, start, end, k=k)
        return [{
            "text": index.documents[i], "source": source,
            "date": index.metadatas[i].get("date", ""), "title": index.metadatas[i].get("title", ""),
            "url": index.metadatas[i].get("url", ""), "similarity": round(s, 3),
        } for i, s in hits]

    event_query = f"{sector} index price movement {row['date']} market conditions"
    return {
        "sebi": top(event_query, "sebi", 2, lo, hi),
        "business_standard": top(event_query, "business_standard", 2, lo, hi),
        "wikipedia": top(f"{sector} Indian stock market crash crisis regulation history", "wikipedia", 2),
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def chunks_of(ctx: dict) -> list[dict]:
    return [c for s in ("business_standard", "sebi", "wikipedia") for c in ctx.get(s, [])]


def day_metrics(ctx: dict, row) -> dict:
    day = datetime.strptime(str(row["date"]), "%Y-%m-%d")
    start, end, kind = rv.anomaly_window(row)
    chunks = chunks_of(ctx)
    news = ctx.get("business_standard", [])
    wiki = ctx.get("wikipedia", [])
    docs = [c["url"] or c["title"] for c in chunks]

    def parsed(c):
        try:
            return datetime.strptime(c["date"][:10], "%Y-%m-%d")
        except ValueError:
            return None

    # Shock days only: regime days legitimately draw on the whole episode.
    news_days = [abs((d - day).days) for d in map(parsed, news) if d] if kind == "shock" else []

    def outside_window(c):
        d = parsed(c)
        return d is not None and (d < start - timedelta(days=OFF_PERIOD_DAYS)
                                  or d > end + timedelta(days=OFF_PERIOD_DAYS))

    return {
        "chunks": len(chunks),
        "market_share": (sum(bool(rv.MARKET_TERMS.search(c["text"])) for c in chunks) / len(chunks)) if chunks else None,
        "duplicate_share": (1 - len(set(docs)) / len(docs)) if docs else 0.0,
        "has_news": bool(news),
        "news_days_off": (sum(news_days) / len(news_days)) if news_days else None,
        "uses_wiki": bool(wiki),
        "wiki_off_period": sum(outside_window(c) for c in wiki),
        "wiki_chunks": len(wiki),
        "mean_similarity": (sum(c["similarity"] for c in chunks) / len(chunks)) if chunks else None,
    }


def event_hit(ctx: dict, keywords: str) -> bool:
    pattern = re.compile(keywords, re.IGNORECASE)
    return any(pattern.search(c["text"]) or pattern.search(c["title"]) for c in chunks_of(ctx))


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    wiki_chunks = df["wiki_chunks"].sum()
    return {
        "Chunks per day (mean)": f"{df['chunks'].mean():.2f}",
        "Days with no context at all": f"{(df['chunks'] == 0).mean():.0%}",
        "Chunks mentioning the market": f"{df['market_share'].mean():.0%}",
        "Repeated-article chunks": f"{df['duplicate_share'].mean():.0%}",
        "Days with news coverage": f"{df['has_news'].mean():.0%}",
        "News distance from a shock day (days, mean)": f"{df['news_days_off'].mean():.2f}",
        "Days using Wikipedia": f"{df['uses_wiki'].mean():.0%}",
        f"Wikipedia chunks about an unrelated period (>{OFF_PERIOD_DAYS} d outside the anomaly/episode)":
            f"{(df['wiki_off_period'].sum() / wiki_chunks) if wiki_chunks else 0:.0%}",
        "Mean similarity of included chunks": f"{df['mean_similarity'].mean():.3f}",
    }


# ----------------------------------------------------------------------------

def main(sector: str):
    print("Loading index...")
    index = ex.get_context_index()
    df = ex.load_final_anomalies(sector)
    flagged = df[ex.is_flagged(df)]
    events = pd.read_csv(EVENTS_DIR / f"{sector}_known_events.csv")

    v1_rows, v2_rows, notes_by_kind = [], [], {"news": 0, "wiki": 0, "none": 0}
    contexts = {}
    for _, row in flagged.iterrows():
        c1, c2 = retrieve_v1(index, row, sector), rv.retrieve_context(index, row)
        contexts[row["date"]] = (c1, c2)
        v1_rows.append(day_metrics(c1, row))
        v2_rows.append(day_metrics(c2, row))
        notes_by_kind["news" if c2["business_standard"] else "wiki" if c2["wikipedia"] else "none"] += 1

    by_date = flagged.set_index("date")
    event_lines, hits = [], {"v1": 0, "v2": 0}
    for _, ev in events.iterrows():
        if ev["date"] not in contexts:
            continue
        c1, c2 = contexts[ev["date"]]
        h1, h2 = event_hit(c1, ev["keywords"]), event_hit(c2, ev["keywords"])
        hits["v1"] += h1
        hits["v2"] += h2
        srcs = ", ".join(f"{k.replace('business_standard', 'news')}:{len(c2[k])}"
                         for k in ("business_standard", "sebi", "wikipedia") if c2[k])
        event_lines.append(f"| {ev['date']} | {ev['event']} | {'yes' if h1 else 'no'} | "
                           f"{'yes' if h2 else 'no'} | {srcs or 'none'} |")
    n_events = len(event_lines)

    s1, s2 = summarise(v1_rows), summarise(v2_rows)
    lines = [
        f"# Retrieval evaluation -- {sector}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `backend/src/rag/evaluate_retrieval.py`. "
        f"No language model involved: this compares the context each retrieval version would give the explainer.",
        "",
        "* **v1** -- original: one generic query, top 2 chunks per source within +/-5 days, "
        "Wikipedia always included with a fixed query and no date filter.",
        "* **v2** -- anomaly-aware pipeline in `rag/retrieval.py`: direction/type-specific queries, "
        "shock or episode date windows, similarity floor, one chunk per article, "
        "Wikipedia only when news is thin and only for events dated near the anomaly.",
        "",
        f"## Known events -- does the context mention the actual cause? ({n_events} events)",
        "",
        f"**v1: {hits['v1']}/{n_events} · v2: {hits['v2']}/{n_events}**",
        "",
        "| Date | Event | v1 | v2 | v2 sources used |",
        "|---|---|---|---|---|",
        *event_lines,
        "",
        f"Business Standard coverage in the store starts in 2022, so for earlier events "
        f"only SEBI filings and Wikipedia event articles are available.",
        "",
        f"## All flagged days ({len(flagged)})",
        "",
        "| Metric | v1 | v2 |",
        "|---|---|---|",
        *[f"| {k} | {s1[k]} | {s2[k]} |" for k in s1],
        "",
        f"v2 context source: news-based on {notes_by_kind['news']} days, "
        f"Wikipedia background on {notes_by_kind['wiki']}, nothing relevant on {notes_by_kind['none']} "
        f"(the explainer then says so instead of guessing).",
        "",
        "Caveats: \"mentions the market\" is a keyword check and v2 gives a small score bonus for those "
        "terms, so that row favours v2 by construction; the known-event check is the stronger signal. "
        "Similarity values are not comparable across versions because the queries differ.",
    ]
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out = VALIDATION_DIR / f"{sector}_retrieval_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ex.SECTOR)
