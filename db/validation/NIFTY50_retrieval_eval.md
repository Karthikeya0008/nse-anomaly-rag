# Retrieval evaluation -- NIFTY50

Generated 2026-10-19 01:32 by `backend/src/rag/evaluate_retrieval.py`. No language model involved: this compares the context each retrieval version would give the explainer.

* **v1** -- original: one generic query, top 2 chunks per source within +/-5 days, Wikipedia always included with a fixed query and no date filter.
* **v2** -- anomaly-aware pipeline in `rag/retrieval.py`: direction/type-specific queries, shock or episode date windows, similarity floor, one chunk per article, Wikipedia only when news is thin and only for events dated near the anomaly.

## Known events -- does the context mention the actual cause? (13 events)

**v1: 5/13 · v2: 9/13**

| Date | Event | v1 | v2 | v2 sources used |
|---|---|---|---|---|
| 2016-11-09 | Demonetisation announcement | no | yes | wikipedia:1 |
| 2018-02-02 | Union Budget 2018 (LTCG tax) | no | no | none |
| 2019-05-20 | Lok Sabha exit polls | no | yes | wikipedia:1 |
| 2019-09-20 | Corporate tax cut | no | no | none |
| 2020-02-01 | Union Budget 2020 | no | no | wikipedia:2 |
| 2020-03-12 | COVID-19 declared pandemic | yes | yes | wikipedia:2 |
| 2020-03-23 | COVID-19 lockdown crash | yes | yes | wikipedia:2 |
| 2020-04-07 | COVID-19 relief rally | yes | yes | wikipedia:2 |
| 2021-02-01 | Union Budget 2021 | no | no | none |
| 2022-02-24 | Russia invades Ukraine | no | yes | wikipedia:1 |
| 2024-06-04 | Lok Sabha election results | yes | yes | news:4 |
| 2025-04-07 | US reciprocal tariffs crash | yes | yes | news:4 |
| 2025-05-12 | India-Pakistan ceasefire rally | no | yes | news:4 |

Business Standard coverage in the store starts in 2022, so for earlier events only SEBI filings and Wikipedia event articles are available.

## All flagged days (194)

| Metric | v1 | v2 |
|---|---|---|
| Chunks per day (mean) | 4.51 | 2.12 |
| Days with no context at all | 0% | 19% |
| Chunks mentioning the market | 77% | 94% |
| Repeated-article chunks | 7% | 0% |
| Days with news coverage | 28% | 28% |
| News distance from a shock day (days, mean) | 2.06 | 1.11 |
| Days using Wikipedia | 100% | 53% |
| Wikipedia chunks about an unrelated period (>90 d outside the anomaly/episode) | 56% | 0% |
| Mean similarity of included chunks | 0.464 | 0.606 |

v2 context source: news-based on 54 days, Wikipedia background on 103, nothing relevant on 37 (the explainer then says so instead of guessing).

Caveats: "mentions the market" is a keyword check and v2 gives a small score bonus for those terms, so that row favours v2 by construction; the known-event check is the stronger signal. Similarity values are not comparable across versions because the queries differ.
