"""
Anomaly-aware retrieval for the explainer.

Pipeline for one flagged day:

  1. Query building   -- 2-3 queries worded for the anomaly: direction (fall/rise),
                         kind (single-day shock vs multi-week regime) and date.
  2. Date windowing   -- point-in-time sources (news, SEBI) are searched only near
                         the day: a few days around a shock, or the whole regime
                         episode plus padding.
  3. Scoring          -- cosine similarity (best over the queries), plus bonuses for
                         closeness in date (pieces dated before a shock count as
                         further away), for mentioning the market, and -- on shock
                         days -- for news describing a move in the same direction.
  4. Filtering        -- a per-source similarity floor drops off-topic chunks
                         instead of always returning k of them; at most one chunk
                         per document, so one article can't fill the context.
  5. Background gate  -- Wikipedia is used only when news coverage is thin, and
                         only articles about events dated near the anomaly.

Search runs in memory over the embeddings stored in the Chroma collection
(about 11k chunks): exact, and milliseconds per query instead of seconds for
Chroma's metadata-filtered queries.
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# ============================================================================
# CONFIGURATION
# ============================================================================

NEWS = "business_standard"
SEBI = "sebi"
WIKI = "wikipedia"

# Coverage of a move appears on the day or in the following days' reports.
SHOCK_DAYS_BEFORE = 2
SHOCK_DAYS_AFTER = 3
EPISODE_PAD_DAYS = 5          # around a regime episode
BACKGROUND_PAD_DAYS = 45      # a Wikipedia event article must be dated this close

CANDIDATES = 40               # per source, before filtering (Wikipedia: all in window)
# Similarity floors. News/SEBI windows hold hundreds of chunks, so the floor does
# the filtering. A Wikipedia article already had to pass the date gate (an event
# within BACKGROUND_PAD_DAYS), so its floor is low and date closeness ranks it.
MIN_SIMILARITY = {NEWS: 0.45, SEBI: 0.50, WIKI: 0.20}
MAX_CHUNKS = {NEWS: 4, SEBI: 2, WIKI: 2}
MIN_NEWS_BEFORE_BACKGROUND = 2  # fewer relevant news chunks than this -> allow Wikipedia

# Bonus for a chunk dated on the anomaly (or inside its episode), decaying with
# distance in days. Strong for Wikipedia: an event article dated on the day of a
# shock is very likely about its cause.
DATE_WEIGHT = {NEWS: 0.08, SEBI: 0.08, WIKI: 0.30}
DATE_DECAY_DAYS = {NEWS: 2.0, SEBI: 2.0, WIKI: 7.0}
# Articles dated before the anomaly count this many times further away: coverage
# of a move appears on or after the day, while earlier pieces are previews.
BEFORE_DAY_FACTOR = 2.0

MARKET_BONUS = 0.03
MARKET_TERMS = re.compile(
    r"\b(nifty|sensex|stock markets?|equit(?:y|ies)|shares|benchmark indices|bourses?|investors)\b",
    re.IGNORECASE,
)

# On shock days, prefer news describing a move in the same direction: on a
# crash day, a rally preview from the day before is about a different move.
DIRECTION_BONUS = 0.06
DIRECTION_TERMS = {
    "down": re.compile(r"\b(fell|fall(?:s|ing)?|slump\w*|crash\w*|plung\w*|tumbl\w*|sell-?off|slid\w*|"
                       r"declin\w*|tank\w*|drag\w*|sea of red|bloodbath|los(?:t|es|ses))\b", re.IGNORECASE),
    "up": re.compile(r"\b(rose|rise[sn]?|rising|rall(?:y|ies|ied)|surg\w*|jump\w*|soar\w*|gain\w*|"
                     r"climb\w*|record high|best day|zoom\w*)\b", re.IGNORECASE),
}


# ============================================================================
# IN-MEMORY INDEX
# ============================================================================

@dataclass
class ContextIndex:
    ids: list
    documents: list
    metadatas: list
    embeddings: np.ndarray      # (n, d), unit-normalised
    sources: np.ndarray         # (n,) source name per chunk
    date_ints: np.ndarray       # (n,) YYYYMMDD, 0 when unknown
    embed: object               # callable: list[str] -> embeddings

    @classmethod
    def from_collection(cls, collection, batch_size: int = 2000) -> "ContextIndex":
        ids, documents, metadatas, embeddings = [], [], [], []
        total = collection.count()
        for offset in range(0, total, batch_size):
            got = collection.get(limit=batch_size, offset=offset,
                                 include=["documents", "metadatas", "embeddings"])
            ids += got["ids"]
            documents += got["documents"]
            metadatas += got["metadatas"]
            embeddings.append(np.asarray(got["embeddings"], dtype=np.float32))
        emb = np.vstack(embeddings)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return cls(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=emb,
            sources=np.array([m.get("source", "") for m in metadatas]),
            date_ints=np.array([int(m.get("date_int") or 0) for m in metadatas]),
            # The collection was built with Chroma's default embedding model.
            embed=DefaultEmbeddingFunction(),
        )

    def search(self, query_vectors: np.ndarray, source: str, start: int | None = None,
               end: int | None = None, k: int | None = CANDIDATES):
        """Top-k chunks (all when k is None) of one source, optionally within
        [start, end] as YYYYMMDD, scored by the best cosine similarity over all
        query vectors."""
        mask = self.sources == source
        if start is not None:
            mask &= (self.date_ints >= start) & (self.date_ints <= end)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        sims = (self.embeddings[idx] @ query_vectors.T).max(axis=1)
        top = np.argsort(-sims)[:k] if k else np.argsort(-sims)
        return [(int(idx[i]), float(sims[i])) for i in top]


# ============================================================================
# HELPERS
# ============================================================================

def _parse(date_str: str) -> datetime:
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")


def _as_int(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def _float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def anomaly_window(row) -> tuple[datetime, datetime, str]:
    """Date span the anomaly is about: the day itself for a shock, the whole
    episode for a regime (or both) day."""
    day = _parse(row["date"])
    kind = str(row.get("anomaly_type", "") or "shock")
    start, end = day, day
    if kind in ("regime", "both") and row.get("regime_episode_start"):
        start = min(start, _parse(row["regime_episode_start"]))
        end = max(end, _parse(row["regime_episode_end"]))
    return start, end, kind


def build_queries(row) -> list[str]:
    day = _parse(row["date"])
    when = f"{day.day} {day:%B %Y}"
    kind = str(row.get("anomaly_type", "") or "shock")
    queries = []

    if kind in ("shock", "both"):
        down = _float(row.get("log_return")) < 0
        move = "fall sharply, crash, sell-off" if down else "surge, rally, jump"
        queries += [
            f"Sensex and Nifty {move} on {when}",
            f"why did Indian stock markets {'fall' if down else 'rise'} today, Sensex Nifty, investors",
        ]
    if kind in ("regime", "both"):
        down = _float(row.get("return_60d")) < 0
        trend = ("correction, weeks of selling, Nifty decline, volatility"
                 if down else "sustained rally, Nifty record highs, strong momentum")
        queries.append(f"Indian stock market {trend} in {day:%B %Y}")
    return queries


def _date_distance(date_int: int, start: datetime, end: datetime) -> tuple[int, int] | None:
    """(days outside [start, end], -1 if before / 0 inside / 1 after), or None if undated."""
    if not date_int:
        return None
    dt = datetime.strptime(str(date_int), "%Y%m%d")
    if dt < start:
        return (start - dt).days, -1
    if dt > end:
        return (dt - end).days, 1
    return 0, 0


def move_direction(row) -> str | None:
    """'down' / 'up' for the day's move on shock days, None for regime-only days."""
    if str(row.get("anomaly_type", "") or "shock") == "regime":
        return None
    return "down" if _float(row.get("log_return")) < 0 else "up"


def _select(index: ContextIndex, hits, source: str, start: datetime, end: datetime,
            direction: str | None = None) -> list[dict]:
    """Score, filter by similarity floor, keep one chunk per document, take top N."""
    scored = []
    for i, sim in hits:
        if sim < MIN_SIMILARITY[source]:
            continue
        meta = index.metadatas[i]
        dist = _date_distance(index.date_ints[i], start, end)
        days = dist[0] if dist else None
        score = sim
        if dist is not None:
            effective = days * (BEFORE_DAY_FACTOR if dist[1] < 0 and source != WIKI else 1)
            score += DATE_WEIGHT[source] * math.exp(-effective / DATE_DECAY_DAYS[source])
        text = index.documents[i]
        if MARKET_TERMS.search(text):
            score += MARKET_BONUS
        if direction and source == NEWS:
            title_and_text = f"{meta.get('title', '')} {text[:600]}"
            same = len(DIRECTION_TERMS[direction].findall(title_and_text))
            other = len(DIRECTION_TERMS["up" if direction == "down" else "down"].findall(title_and_text))
            if same > other:
                score += DIRECTION_BONUS
        scored.append((score, sim, days, i, meta))

    scored.sort(key=lambda s: -s[0])
    chosen, seen_docs = [], set()
    for score, sim, days, i, meta in scored:
        doc_key = meta.get("url") or meta.get("title") or index.ids[i]
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        chosen.append({
            "text": index.documents[i],
            "source": meta.get("source", source),
            "date": meta.get("date", "unknown"),
            "title": meta.get("title", "untitled"),
            "url": meta.get("url", ""),
            "similarity": round(sim, 3),
            "days_from_anomaly": days,
        })
        if len(chosen) >= MAX_CHUNKS[source]:
            break
    return chosen


# ============================================================================
# PUBLIC
# ============================================================================

def retrieve_context(index: ContextIndex, row) -> dict:
    """Context for one flagged day.

    Returns {"business_standard": [...], "sebi": [...], "wikipedia": [...],
             "queries": [...], "notes": [...]} -- each chunk a dict with text,
    source, date, title, url, similarity and days_from_anomaly.
    """
    start, end, kind = anomaly_window(row)
    queries = build_queries(row)
    qvec = np.asarray(index.embed(queries), dtype=np.float32)
    qvec /= np.linalg.norm(qvec, axis=1, keepdims=True)

    if kind == "shock":
        lo, hi = start - timedelta(days=SHOCK_DAYS_BEFORE), end + timedelta(days=SHOCK_DAYS_AFTER)
    else:
        lo, hi = start - timedelta(days=EPISODE_PAD_DAYS), end + timedelta(days=EPISODE_PAD_DAYS)

    notes = []
    result = {"queries": queries}
    for source in (NEWS, SEBI):
        hits = index.search(qvec, source, _as_int(lo), _as_int(hi))
        result[source] = _select(index, hits, source, start, end, move_direction(row))

    n_news = len(result[NEWS])
    if n_news >= MIN_NEWS_BEFORE_BACKGROUND:
        result[WIKI] = []
        notes.append(f"{n_news} relevant news chunks found; background reference not needed.")
    else:
        blo = start - timedelta(days=BACKGROUND_PAD_DAYS)
        bhi = end + timedelta(days=BACKGROUND_PAD_DAYS)
        # Every chunk in the window (only a handful of articles), so one long
        # article's many chunks can't crowd out a shorter, better-dated one.
        hits = index.search(qvec, WIKI, _as_int(blo), _as_int(bhi), k=None)
        result[WIKI] = _select(index, hits, WIKI, start, end)
        reason = "no relevant news coverage" if n_news == 0 else "only one relevant news chunk"
        if result[WIKI]:
            notes.append(f"{reason.capitalize()} in the date window; added background articles "
                         f"about events dated within {BACKGROUND_PAD_DAYS} days.")
        else:
            notes.append(f"{reason.capitalize()} in the date window, and no background article "
                         f"about an event within {BACKGROUND_PAD_DAYS} days.")

    if not result[SEBI]:
        notes.append("No SEBI filing near the date was relevant enough to include.")
    result["notes"] = notes
    return result
