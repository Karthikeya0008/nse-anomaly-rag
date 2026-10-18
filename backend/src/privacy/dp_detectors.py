"""
Differentially private anomaly detectors and ensemble.

Which of the production detectors can be made DP?

  * Isolation Forest   -> DPRandomTrees: the same idea (anomalies sit in sparsely
                          populated cells of a random partition), but splits are
                          drawn without looking at the data and only noisy leaf
                          counts are learned.
  * Elliptic Envelope  -> DPGaussian: distance from a noisy mean in units of a
                          noisy per-feature spread (diagonal Mahalanobis).
  * (new) histogram    -> DPHistogram: HBOS-style per-feature density from noisy
                          histograms -- the cheapest DP density estimate.
  * LOF, kNN distance  -> not DP-able as released models: they *store the
                          training rows* and score against them directly.
  * One-Class SVM      -> stores support vectors (training rows) too; DP
                          versions need objective perturbation over a kernel
                          approximation -- out of scope here.

Everything the detectors learn passes through a mechanism in dp_mechanisms.py;
the feature transform uses only public constants, never the data.
"""
import math

import numpy as np

from dp_mechanisms import HistogramCDF, PrivacyLedger, dp_histogram, laplace


# ============================================================================
# Public feature transform
# ============================================================================

# A-priori bounds, fixed from market rules and maths rather than from the data:
#   * NSE's market-wide circuit breaker halts trading at a 20% index move, so a
#     day's return, gap or range can't exceed about 20%;
#   * a z-score of a value within its own 20-day window satisfies
#     |z| <= (n-1)/sqrt(n) = 4.25 for n = 20;
#   * multi-day returns, volatilities and ratios use generous ranges well beyond
#     anything seen in Indian index history.
PUBLIC_BOUNDS = {
    "log_return": (-0.20, 0.20),
    "gap": (-0.20, 0.20),
    "range_vol": (0.0, 0.20),
    "return_zscore": (-4.25, 4.25),
    "return_5d": (-0.30, 0.30),
    "return_20d": (-0.40, 0.40),
    "return_60d": (-0.60, 0.60),
    "volatility_5d": (0.0, 0.10),
    "volatility_20d": (0.0, 0.08),
    "volatility_60d": (0.0, 0.06),
    "ma_deviation_60d": (-0.40, 0.40),
    "rolling_sharpe_20d": (-1.50, 1.50),
}

# Typical magnitudes from public market knowledge (NIFTY 50's daily volatility is
# about 1%, i.e. 15-20% a year). Used as the scale of an asinh transform so that
# ordinary days aren't squashed into one corner of the bounded range.
PUBLIC_SCALES = {
    "log_return": 0.01, "gap": 0.005, "range_vol": 0.01, "return_zscore": 1.0,
    "return_5d": 0.02, "return_20d": 0.04, "return_60d": 0.07,
    "volatility_5d": 0.01, "volatility_20d": 0.01, "volatility_60d": 0.01,
    "ma_deviation_60d": 0.04, "rolling_sharpe_20d": 0.3,
}


class PublicTransform:
    """clip to public bounds -> asinh(x / public scale) -> rescale to [0, 1]."""

    def __init__(self, features: list[str]):
        self.features = list(features)
        self.lo = np.array([PUBLIC_BOUNDS[f][0] for f in features])
        self.hi = np.array([PUBLIC_BOUNDS[f][1] for f in features])
        self.scale = np.array([PUBLIC_SCALES[f] for f in features])
        self._tlo = np.arcsinh(self.lo / self.scale)
        self._thi = np.arcsinh(self.hi / self.scale)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        t = np.arcsinh(np.clip(X, self.lo, self.hi) / self.scale)
        return (t - self._tlo) / (self._thi - self._tlo)

    def clip_rate(self, X: np.ndarray) -> dict:
        clipped = (X < self.lo) | (X > self.hi)
        return {f: float(clipped[:, j].mean()) for j, f in enumerate(self.features)}


# ============================================================================
# Detectors (all operate on transformed data in [0, 1]^d)
# ============================================================================

class DPHistogram:
    """HBOS: score = sum over features of -log(noisy bin probability)."""
    name = "dp_histogram"

    def __init__(self, bins: int = 20):
        self.bins = bins

    def fit(self, U, epsilon, rng, ledger: PrivacyLedger, n_public: int):
        d = U.shape[1]
        # One record lands in one bin of *each* feature's histogram: L1 sensitivity d.
        ledger.spend(self.name, epsilon, note=f"{d} histograms x {self.bins} bins, Laplace(d/eps)")
        edges = np.linspace(0, 1, self.bins + 1)
        self.probs = []
        for j in range(d):
            counts = dp_histogram(U[:, j], edges, epsilon, rng, sensitivity=d)
            counts = np.maximum(counts, 1.0)
            self.probs.append(counts / counts.sum())
        self.max_score = d * math.log(n_public + self.bins)   # public upper bound
        return self

    def score(self, U):
        idx = np.clip((U * self.bins).astype(int), 0, self.bins - 1)
        return sum(-np.log(self.probs[j][idx[:, j]]) for j in range(U.shape[1]))


class DPGaussian:
    """Diagonal Mahalanobis distance from a noisy mean with noisy spreads."""
    name = "dp_gaussian"
    Z_CLIP = 10.0

    def fit(self, U, epsilon, rng, ledger: PrivacyLedger, n_public: int):
        d = U.shape[1]
        ledger.spend(self.name, epsilon, note="noisy sums of (u-0.5) and (u-0.5)^2, Laplace")
        C = U - 0.5                                   # each entry in [-0.5, 0.5]
        e1, e2 = epsilon / 2, epsilon / 2
        # Removing one record changes sum(C) by at most 0.5 per feature (L1: d/2)
        # and sum(C^2) by at most 0.25 per feature (L1: d/4).
        s1 = laplace(C.sum(axis=0), d * 0.5, e1, rng)
        s2 = laplace((C ** 2).sum(axis=0), d * 0.25, e2, rng)
        self.mean = s1 / n_public + 0.5
        var = s2 / n_public - (s1 / n_public) ** 2
        self.std = np.sqrt(np.clip(var, 1e-4, None))  # floor keeps the scale sane
        self.max_score = self.Z_CLIP
        return self

    def score(self, U):
        z = np.clip((U - self.mean) / self.std, -self.Z_CLIP, self.Z_CLIP)
        return np.sqrt((z ** 2).mean(axis=1))


class DPRandomTrees:
    """Isolation-style random partition trees with noisy leaf counts.

    Each tree splits [0,1]^d at random features and random points chosen without
    looking at the data, and is trained on its own disjoint slice of the days. A
    day therefore affects one leaf count of one tree: by parallel composition the
    whole forest costs epsilon once. Score = average of -log(leaf mass)."""
    name = "dp_random_trees"

    def __init__(self, n_trees: int = 8, depth: int = 5):
        self.n_trees, self.depth = n_trees, depth

    def _route(self, U, tree):
        node = np.zeros(len(U), dtype=int)
        for _ in range(self.depth):
            f, v = tree["feature"][node], tree["value"][node]
            node = 2 * node + 1 + (U[np.arange(len(U)), f] > v)
        return node - (2 ** self.depth - 1)           # leaf index

    def fit(self, U, epsilon, rng, ledger: PrivacyLedger, n_public: int):
        d = U.shape[1]
        ledger.spend(self.name, epsilon,
                     note=f"{self.n_trees} trees on disjoint slices (parallel composition), Laplace(1/eps) leaf counts")
        n_internal = 2 ** self.depth - 1
        slices = np.array_split(rng.permutation(len(U)), self.n_trees)
        self.trees = []
        for part in slices:
            feature = rng.integers(0, d, n_internal)
            value = np.empty(n_internal)
            lo, hi = np.zeros((n_internal, d)), np.ones((n_internal, d))
            for i in range(n_internal):                # heap order: children 2i+1, 2i+2
                f = feature[i]
                value[i] = rng.uniform(lo[i, f], hi[i, f])
                for child, side in ((2 * i + 1, "left"), (2 * i + 2, "right")):
                    if child < n_internal:
                        lo[child], hi[child] = lo[i], hi[i]
                        if side == "left":
                            hi[child, f] = value[i]
                        else:
                            lo[child, f] = value[i]
            tree = {"feature": feature, "value": value}
            counts = np.bincount(self._route(U[part], tree), minlength=2 ** self.depth)
            noisy = np.clip(laplace(counts, 1.0, epsilon, rng), 0, None)
            tree["mass"] = (noisy + 1) / (noisy.sum() + 2 ** self.depth)
            self.trees.append(tree)
        n_slice = n_public / self.n_trees
        self.max_score = math.log(n_slice + 2 ** self.depth)
        return self

    def score(self, U):
        return np.mean([-np.log(t["mass"][self._route(U, t)]) for t in self.trees], axis=0)


# ============================================================================
# Ensemble
# ============================================================================

# Share of the epsilon budget per component. Detectors learn the shape of normal
# days; calibration learns each score's distribution (to put the detectors on a
# common 0-1 scale, like the production rank-averaging) and the final threshold.
BUDGET_SPLIT = {
    "dp_random_trees": 0.25, "dp_histogram": 0.20, "dp_gaussian": 0.15,
    "calibration": 0.30,   # 3 score histograms, 0.10 each
    "threshold": 0.10,
}
CALIBRATION_BINS = 50


class DPEnsemble:
    """Three DP detectors, each mapped to [0,1] by a DP estimate of its score
    distribution, averaged, and thresholded at a DP estimate of the top
    `contamination` share. The fitted object holds only DP outputs, so it can be
    shared or published without exposing the training days."""

    def __init__(self, features: list[str], epsilon: float, contamination: float = 0.05,
                 seed: int | None = None):
        # seed=None draws fresh OS randomness. The guarantee assumes the noise is
        # unknown to an attacker, so fixed seeds are only for evaluation runs.
        self.features, self.epsilon, self.contamination = list(features), epsilon, contamination
        self.seed = seed
        self.transform = PublicTransform(features)
        self.detectors = [DPRandomTrees(), DPHistogram(), DPGaussian()]

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.seed = None   # don't keep it on the (shareable) fitted object
        n = len(X)   # the number of trading days is public (exchange calendar)
        self.ledger = PrivacyLedger(budget=self.epsilon)
        U = self.transform(X)

        for det in self.detectors:
            det.fit(U, self.epsilon * BUDGET_SPLIT[det.name], rng, self.ledger, n)

        # Calibration: a DP histogram of each detector's training scores. The score
        # function is already a DP output (post-processing), and each day adds one
        # count: sensitivity 1.
        e_cal = self.epsilon * BUDGET_SPLIT["calibration"] / len(self.detectors)
        self.cdfs = []
        for det in self.detectors:
            edges = np.linspace(0, det.max_score, CALIBRATION_BINS + 1)
            self.ledger.spend(f"calibrate {det.name}", e_cal, note="score histogram, Laplace(1/eps)")
            noisy = dp_histogram(det.score(U), edges, e_cal, rng, clamp=False)
            self.cdfs.append(HistogramCDF(edges, noisy, n))
        self.detector_thresholds = [cdf.quantile(1 - self.contamination) for cdf in self.cdfs]

        e_thr = self.epsilon * BUDGET_SPLIT["threshold"]
        self.ledger.spend("threshold", e_thr, note="histogram of combined score, Laplace(1/eps)")
        edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
        noisy = dp_histogram(self._combine(U), edges, e_thr, rng, clamp=False)
        self.threshold = HistogramCDF(edges, noisy, n).quantile(1 - self.contamination)
        return self

    def _combine(self, U):
        return np.mean([cdf(det.score(U)) for det, cdf in zip(self.detectors, self.cdfs)], axis=0)

    # -- post-processing only (no further privacy cost) -----------------------

    def score(self, X):
        return self._combine(self.transform(X))

    def detector_scores(self, X) -> dict:
        U = self.transform(X)
        return {det.name: det.score(U) for det in self.detectors}

    def predict(self, X):
        """(flags, combined score, votes): votes = detectors whose own score is
        above their DP top-`contamination` threshold."""
        U = self.transform(X)
        raw = [det.score(U) for det in self.detectors]
        combined = np.mean([cdf(r) for r, cdf in zip(raw, self.cdfs)], axis=0)
        votes = np.sum([r >= t for r, t in zip(raw, self.detector_thresholds)], axis=0)
        return combined >= self.threshold, combined, votes
