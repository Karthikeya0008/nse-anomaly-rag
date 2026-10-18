# Model validation -- NIFTY50

Generated 2026-10-19 01:31 by `backend/src/models/validate_models.py`. Ensembles: rank-average of isoforest, ocsvm, lof, elliptic, knn; top 5% of days flagged. 13 reference events from `db/events/NIFTY50_known_events.csv`.

## Summary

* **Known events:** the shock ensemble flags 13/13 on the exact day (p = 7e-18 vs. random 5% flags); the all-features model 11/13.
* **No look-ahead (walk-forward):** trained only on earlier years, the shock ensemble still catches 11/11 events in its test years; out-of-sample and full-history scores have rank correlation 0.97 (shock) / 0.85 (regime).
* **Stability:** across seeds the ensemble's flagged days overlap 0.97 (shock) / 0.99 (regime) with production; on 80% subsamples 0.91 / 0.92.
* **Synthetic shocks:** 93% of injected 3σ jumps and 100% of 5σ jumps are flagged (untouched calm days: 0%).
* **Synthetic regimes:** volatility x2 39%; volatility x3 56%; volatility x4 72%; drift 0.4%/day 50%; drift 0.7%/day 67%; drift 1.0%/day 83% detected (control periods ~2%).
* **Forward volatility:** after shock-flagged days, the next 20 days are 1.7x as volatile as after other days (regime: 1.9x).

## 1. Known events and chance level

How many reference events fall on a flagged day, and the probability of doing at least that well by flagging the same number of days at random (hypergeometric test).

| Feature set | Flagged days | Events (exact day) | Events (±1 day) | Lift over random | p-value (random 5% flags) |
|---|---|---|---|---|---|
| shock | 121 | 13/13 | 13/13 | 20.0x | 6.5e-18 |
| regime | 121 | 4/13 | 4/13 | 6.1x | 3.0e-03 |
| all | 121 | 11/13 | 11/13 | 16.9x | 2.3e-13 |

The reference list is small (13 events) and was used when choosing the feature split, so treat it as a sanity check rather than an unbiased accuracy estimate -- the walk-forward and injection tests below don't depend on it.

## 2. Sensitivity to the contamination threshold

**Shock ensemble**

| Contamination | Flagged days | Events (exact) | Events (±1 day) |
|---|---|---|---|
| 1.0% | 25 | 8/13 | 8/13 |
| 2.0% | 49 | 11/13 | 11/13 |
| 3.0% | 73 | 12/13 | 12/13 |
| 5.0% | 121 | 13/13 | 13/13 |
| 7.5% | 182 | 13/13 | 13/13 |
| 10.0% | 242 | 13/13 | 13/13 |

**Regime ensemble**

| Contamination | Flagged days | Events (exact) | Events (±1 day) |
|---|---|---|---|
| 1.0% | 25 | 3/13 | 3/13 |
| 2.0% | 50 | 3/13 | 3/13 |
| 3.0% | 73 | 3/13 | 3/13 |
| 5.0% | 121 | 4/13 | 4/13 |
| 7.5% | 182 | 6/13 | 7/13 |
| 10.0% | 242 | 6/13 | 7/13 |

## 3. Walk-forward (out-of-sample) test

For each year, every model is trained only on the years before it, and the test year's days are ranked against the training scores with the threshold fixed in advance -- exactly what a live system could have done. The production model instead fits all history at once.

**Shock ensemble** -- tested 2019-2026, 1880 days

| Test year | Train days | Test days | Flagged (OOS) | Flag rate | Flagged (in-sample) | Jaccard vs in-sample |
|---|---|---|---|---|---|---|
| 2019 | 538 | 245 | 10 | 4.1% | 5 | 0.50 |
| 2020 | 783 | 252 | 76 | 30.2% | 49 | 0.64 |
| 2021 | 1035 | 248 | 4 | 1.6% | 11 | 0.36 |
| 2022 | 1283 | 248 | 18 | 7.3% | 22 | 0.74 |
| 2023 | 1531 | 246 | 2 | 0.8% | 2 | 1.00 |
| 2024 | 1777 | 249 | 7 | 2.8% | 8 | 0.88 |
| 2025 | 2026 | 249 | 3 | 1.2% | 3 | 1.00 |
| 2026 | 2275 | 143 | 9 | 6.3% | 8 | 0.89 |

* Known events in test years caught out-of-sample: **11/11** (in-sample model: 11/11)
* Rank correlation of out-of-sample vs in-sample scores: **0.97** (Spearman)
* Out-of-sample flags that the full-history model also flags: 74%; full-history flags recovered out-of-sample: 88%; overall flag rate 6.9% (target 5%)

**Regime ensemble** -- tested 2019-2026, 1880 days

| Test year | Train days | Test days | Flagged (OOS) | Flag rate | Flagged (in-sample) | Jaccard vs in-sample |
|---|---|---|---|---|---|---|
| 2019 | 538 | 245 | 102 | 41.6% | 0 | 0.00 |
| 2020 | 783 | 252 | 182 | 72.2% | 86 | 0.47 |
| 2021 | 1035 | 248 | 1 | 0.4% | 2 | 0.00 |
| 2022 | 1283 | 248 | 25 | 10.1% | 17 | 0.62 |
| 2023 | 1531 | 246 | 0 | 0.0% | 0 | 1.00 |
| 2024 | 1777 | 249 | 17 | 6.8% | 4 | 0.24 |
| 2025 | 2026 | 249 | 1 | 0.4% | 0 | 0.00 |
| 2026 | 2275 | 143 | 17 | 11.9% | 10 | 0.59 |

* Known events in test years caught out-of-sample: **6/11** (in-sample model: 4/11)
* Rank correlation of out-of-sample vs in-sample scores: **0.85** (Spearman)
* Out-of-sample flags that the full-history model also flags: 34%; full-history flags recovered out-of-sample: 97%; overall flag rate 18.4% (target 5%)

## 4. Stability

**Shock**

| Model | Seed change (10 seeds) | 80% subsample (20 runs) |
|---|---|---|
| isoforest | 0.90 ± 0.02 | 0.91 ± 0.03 |
| ocsvm | deterministic | 0.84 ± 0.02 |
| lof | deterministic | 0.86 ± 0.03 |
| elliptic | 1.00 ± 0.00 | 0.98 ± 0.02 |
| knn | deterministic | 0.90 ± 0.03 |
| ensemble | 0.97 ± 0.01 | 0.91 ± 0.02 |

**Regime**

| Model | Seed change (10 seeds) | 80% subsample (20 runs) |
|---|---|---|
| isoforest | 0.90 ± 0.04 | 0.91 ± 0.03 |
| ocsvm | deterministic | 0.86 ± 0.02 |
| lof | deterministic | 0.82 ± 0.03 |
| elliptic | 1.00 ± 0.00 | 0.97 ± 0.02 |
| knn | deterministic | 0.82 ± 0.03 |
| ensemble | 0.99 ± 0.01 | 0.92 ± 0.02 |

Values are the Jaccard overlap of flagged days with the production run (1.00 = identical set of days).

## 5. Synthetic anomaly injection

Known-size anomalies are planted in calm stretches of the real NIFTY 50 price series; features are rebuilt and the ensemble refitted (3 trial(s) per row). Control stretches are equally calm but left untouched, so their rate is the false-alarm baseline.

**Shocks** -- a permanent jump of k × the day's 20-day volatility from the open (10 days per trial):

| Jump size | isoforest | ocsvm | lof | elliptic | knn | Ensemble | Untouched control days |
|---|---|---|---|---|---|---|---|
| 2 σ | 70% | 53% | 60% | 33% | 43% | **63%** | 0% |
| 3 σ | 93% | 90% | 77% | 50% | 80% | **93%** | 0% |
| 4 σ | 100% | 93% | 97% | 73% | 97% | **100%** | 0% |
| 5 σ | 100% | 100% | 100% | 90% | 100% | **100%** | 0% |
| 6 σ | 100% | 100% | 100% | 97% | 100% | **100%** | 0% |

**Regimes** -- 20 trading days with returns scaled (volatility burst) or shifted (drift); a period counts as detected if any day in it or the 10 days after is flagged (6 periods per trial):

| Injected period (20 days) | Periods detected | Share of days flagged | Untouched control periods |
|---|---|---|---|
| volatility x2 | **39%** | 5% | 0% |
| volatility x3 | **56%** | 18% | 0% |
| volatility x4 | **72%** | 21% | 0% |
| drift 0.4%/day | **50%** | 6% | 11% |
| drift 0.7%/day | **67%** | 12% | 0% |
| drift 1.0%/day | **83%** | 16% | 0% |

## 6. Separation and forward-looking validity

**Shock ensemble**

| Measure | Median, flagged | Median, other days | Ratio | Effect size (rank-biserial) | p (Mann-Whitney, one-sided) |
|---|---|---|---|---|---|
| |daily return| | 2.32% | 0.49% | 4.7x | 0.80 | 1.3e-49 |
| intraday range | 2.64% | 0.87% | 3.0x | 0.87 | 3.4e-59 |
| |opening gap| | 1.53% | 0.27% | 5.7x | 0.76 | 7.3e-46 |
| volatility, next 5 days | 1.24% | 0.65% | 1.9x | 0.58 | 1.9e-27 |
| volatility, next 20 days | 1.26% | 0.73% | 1.7x | 0.61 | 9.7e-30 |

**Regime ensemble**

| Measure | Median, flagged | Median, other days | Ratio | Effect size (rank-biserial) | p (Mann-Whitney, one-sided) |
|---|---|---|---|---|---|
| |daily return| | 1.22% | 0.50% | 2.5x | 0.47 | 1.2e-18 |
| intraday range | 1.93% | 0.87% | 2.2x | 0.72 | 3.6e-41 |
| |opening gap| | 0.76% | 0.27% | 2.8x | 0.56 | 2.5e-25 |
| volatility, next 5 days | 1.46% | 0.64% | 2.3x | 0.69 | 3.9e-38 |
| volatility, next 20 days | 1.34% | 0.72% | 1.9x | 0.78 | 6.7e-48 |

Forward volatility is measured over the days *after* each day, which the models never see. Daily observations overlap and are autocorrelated, so p-values are optimistic; the effect sizes and ratios are the more reliable read.

## 7. Agreement between models (Spearman rank correlation of scores)

**Shock** (mean pairwise ρ = 0.68)

|  | isoforest | ocsvm | lof | elliptic | knn |
|---|---|---|---|---|---|
| isoforest | 1.00 | 0.56 | 0.76 | 0.94 | 0.83 |
| ocsvm | 0.56 | 1.00 | 0.39 | 0.53 | 0.26 |
| lof | 0.76 | 0.39 | 1.00 | 0.81 | 0.88 |
| elliptic | 0.94 | 0.53 | 0.81 | 1.00 | 0.87 |
| knn | 0.83 | 0.26 | 0.88 | 0.87 | 1.00 |

**Regime** (mean pairwise ρ = 0.56)

|  | isoforest | ocsvm | lof | elliptic | knn |
|---|---|---|---|---|---|
| isoforest | 1.00 | 0.55 | 0.54 | 0.88 | 0.67 |
| ocsvm | 0.55 | 1.00 | 0.38 | 0.43 | 0.14 |
| lof | 0.54 | 0.38 | 1.00 | 0.60 | 0.65 |
| elliptic | 0.88 | 0.43 | 0.60 | 1.00 | 0.75 |
| knn | 0.67 | 0.14 | 0.65 | 0.75 | 1.00 |

Runtime 474s.
