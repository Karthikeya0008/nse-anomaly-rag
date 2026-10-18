# Differential privacy report -- NIFTY50

Generated 2026-10-19 01:51 by `backend/src/privacy/train_dp_models.py`.

## What is protected

* **Unit of privacy:** one trading day's feature row. A model is ε-DP if adding or removing any one day changes the probability of every possible fitted model by at most a factor e^ε.
* **What it covers:** the fitted model files and everything derived from them (detector parameters, score distributions, thresholds). They can be shared, published or run by third parties without revealing the days they were trained on.
* **What it does not cover:** the per-day flags. A day's own flag is computed from that day's own features, which is the point of anomaly detection; DP limits what the *model* reveals about the *other* days.
* **Raw prices:** one closing price feeds up to 61 feature rows (60-day windows), so by group privacy the guarantee for an individual price is ε × 61. Rolling features make row-level DP the meaningful unit here; for client- or trade-level data (see README) the unit would be the client.

## 1. Why: the non-DP models leak their training data

Fitted non-DP models were loaded from `db/models/multi/` and inverted with their own saved scaler:

| Model file | Training rows stored | Days whose features are recovered exactly |
|---|---|---|
| shock / knn | 2418 of 2418 | 100% |
| shock / lof | 2418 of 2418 | 100% |
| shock / ocsvm | 133 support vectors | these rows, exactly |
| regime / knn | 2418 of 2418 | 100% |
| regime / lof | 2418 of 2418 | 100% |
| regime / ocsvm | 130 support vectors | these rows, exactly |

LOF and kNN score new days by comparing them with stored training days, so the file *is* the training data; One-Class SVM keeps its support vectors, which are training days -- the most unusual ones. Isolation Forest stores split thresholds drawn between the minimum and maximum of sub-samples, which leak value ranges. The DP models below store only noisy counts, noisy means and spreads, random data-independent splits and noisy score distributions.

## 2. Deployed DP models

| Feature set | ε | Days flagged | Flag rate | Overlap with production (Jaccard) | Reference events caught | Max share of values clipped to public bounds |
|---|---|---|---|---|---|---|
| shock | 2 | 49 | 2.0% | 0.40 | 13 / 13 | 0.00% |
| regime | 2 | 159 | 6.6% | 0.64 | 3 / 13 | 0.00% |

Shock and regime models are trained on the same days, so by sequential composition a trading day's total privacy loss is **ε = 4** (4 × 61 = 244 for a single raw price). These models were fitted with fresh random noise; re-running gives slightly different flags.

## 3. Privacy ledger

| Set | Component | ε | Composition | Mechanism |
|---|---|---|---|---|
| shock | dp_random_trees | 0.500 | sequential | 8 trees on disjoint slices (parallel composition), Laplace(1/eps) leaf counts |
| shock | dp_histogram | 0.400 | sequential | 4 histograms x 20 bins, Laplace(d/eps) |
| shock | dp_gaussian | 0.300 | sequential | noisy sums of (u-0.5) and (u-0.5)^2, Laplace |
| shock | calibrate dp_random_trees | 0.200 | sequential | score histogram, Laplace(1/eps) |
| shock | calibrate dp_histogram | 0.200 | sequential | score histogram, Laplace(1/eps) |
| shock | calibrate dp_gaussian | 0.200 | sequential | score histogram, Laplace(1/eps) |
| shock | threshold | 0.200 | sequential | histogram of combined score, Laplace(1/eps) |
| regime | dp_random_trees | 0.500 | sequential | 8 trees on disjoint slices (parallel composition), Laplace(1/eps) leaf counts |
| regime | dp_histogram | 0.400 | sequential | 6 histograms x 20 bins, Laplace(d/eps) |
| regime | dp_gaussian | 0.300 | sequential | noisy sums of (u-0.5) and (u-0.5)^2, Laplace |
| regime | calibrate dp_random_trees | 0.200 | sequential | score histogram, Laplace(1/eps) |
| regime | calibrate dp_histogram | 0.200 | sequential | score histogram, Laplace(1/eps) |
| regime | calibrate dp_gaussian | 0.200 | sequential | score histogram, Laplace(1/eps) |
| regime | threshold | 0.200 | sequential | histogram of combined score, Laplace(1/eps) |

## 4. Privacy-utility trade-off

Smaller ε = stronger privacy = more noise. "∞" runs the same three detectors without noise, which separates the cost of using DP-compatible detectors from the cost of the noise itself.

**Shock ensemble** (10 independent noise draws per row)

| ε (per feature set) | Days flagged | Overlap with production flags (Jaccard) | Reference events caught |
|---|---|---|---|
| 0.25 | 7.5% | 0.24 ± 0.12 | 8.5 / 13 |
| 0.5 | 4.9% | 0.31 ± 0.19 | 9.5 / 13 |
| 1 | 4.4% | 0.38 ± 0.18 | 10.4 / 13 |
| 2 | 4.1% | 0.56 ± 0.07 | 12.8 / 13 |
| 5 | 4.0% | 0.59 ± 0.10 | 12.9 / 13 |
| 10 | 4.1% | 0.61 ± 0.08 | 13.0 / 13 |
| ∞ (no noise) | 3.9% | 0.60 ± 0.06 | 13.0 / 13 |

**Regime ensemble** (10 independent noise draws per row)

| ε (per feature set) | Days flagged | Overlap with production flags (Jaccard) | Reference events caught |
|---|---|---|---|
| 0.25 | 10.6% | 0.14 ± 0.13 | 3.7 / 13 |
| 0.5 | 6.7% | 0.24 ± 0.22 | 2.7 / 13 |
| 1 | 5.0% | 0.34 ± 0.24 | 2.6 / 13 |
| 2 | 4.2% | 0.43 ± 0.19 | 3.0 / 13 |
| 5 | 4.1% | 0.58 ± 0.08 | 3.2 / 13 |
| 10 | 4.2% | 0.62 ± 0.07 | 3.1 / 13 |
| ∞ (no noise) | 3.9% | 0.63 ± 0.07 | 3.0 / 13 |

## 5. Caveats

* The feature bounds and scales are fixed constants (market rules, maths, typical magnitudes), but they were chosen by people who had seen this data. In a real deployment they must be fixed before looking at the private data.
* `dp_mechanisms.py` is a transparent teaching implementation on numpy's floating-point Laplace sampler; production use should rely on a hardened library such as OpenDP.
* The DP guarantee requires the noise to be secret: the deployed models use fresh OS randomness; fixed seeds are used only in the evaluation sweep above.
