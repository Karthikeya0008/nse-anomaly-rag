"""
Differential-privacy building blocks (pure epsilon-DP, add/remove-one neighbours).

A mechanism M is epsilon-DP if, for any two datasets D and D' that differ by one
record and any set of outputs S,  P[M(D) in S] <= e^epsilon * P[M(D') in S].
Here a "record" is one trading day's feature row (see privacy/README section in
the project README for what that does and doesn't protect).

Composition rules used by the layer:
  * sequential -- mechanisms run on the same records add up their epsilons;
  * parallel   -- mechanisms on disjoint subsets of records cost only the max;
  * post-processing -- anything computed from DP outputs alone is free.

Note: this is a transparent teaching implementation built on numpy's floating-
point Laplace sampler. Production systems should use a hardened library
(e.g. OpenDP), which also defends against floating-point side channels.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PrivacyLedger:
    """Records every epsilon spent so the total can be checked and reported."""
    budget: float
    entries: list = field(default_factory=list)

    @property
    def spent(self) -> float:
        return float(sum(e["epsilon"] for e in self.entries if e["composition"] == "sequential"))

    def spend(self, name: str, epsilon: float, composition: str = "sequential", note: str = "") -> float:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if composition == "sequential" and self.spent + epsilon > self.budget + 1e-9:
            raise RuntimeError(f"privacy budget exceeded: {self.spent:.3f} + {epsilon:.3f} > {self.budget:.3f}")
        self.entries.append({"component": name, "epsilon": float(epsilon),
                             "composition": composition, "note": note})
        return epsilon

    def as_dict(self) -> dict:
        return {"budget": self.budget, "spent": self.spent, "entries": self.entries}


def laplace(values, sensitivity: float, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Laplace mechanism: add Lap(sensitivity / epsilon) noise (L1 sensitivity)."""
    values = np.asarray(values, dtype=float)
    return values + rng.laplace(0.0, sensitivity / epsilon, size=values.shape)


def dp_histogram(values, edges, epsilon: float, rng: np.random.Generator,
                 sensitivity: float = 1.0, clamp: bool = True) -> np.ndarray:
    """Noisy bin counts. Each record falls in exactly one bin, so adding or
    removing it changes one count by 1: L1 sensitivity 1. Values outside the
    (public) edges are clipped into the end bins. With clamp, negative noisy
    counts are set to 0 (post-processing); keep them for unbiased sums."""
    v = np.clip(np.asarray(values, dtype=float), edges[0], edges[-1])
    counts, _ = np.histogram(v, bins=edges)
    noisy = laplace(counts, sensitivity, epsilon, rng)
    return np.clip(noisy, 0, None) if clamp else noisy


class HistogramCDF:
    """A cumulative distribution from a DP histogram (post-processing only).

    Built from *unclamped* noisy counts divided by the public record count:
    clamping negatives to zero would add mass to every empty bin and push
    upper quantiles too high. The running maximum then makes it monotone."""

    def __init__(self, edges: np.ndarray, noisy_counts: np.ndarray, n_public: int):
        self.edges = np.asarray(edges, dtype=float)
        cdf = np.concatenate([[0.0], np.cumsum(noisy_counts) / n_public])
        self.cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        self.cdf[-1] = 1.0

    def __call__(self, values) -> np.ndarray:
        """P(score <= v), linearly interpolated within bins."""
        v = np.clip(np.asarray(values, dtype=float), self.edges[0], self.edges[-1])
        return np.interp(v, self.edges, self.cdf)

    def quantile(self, q: float) -> float:
        """Smallest edge where the CDF reaches q."""
        i = int(np.searchsorted(self.cdf, q, side="left"))
        return float(self.edges[min(i, len(self.edges) - 1)])
