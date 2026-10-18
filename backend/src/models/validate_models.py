"""
Validation suite for the shock and regime anomaly ensembles.

Unsupervised detectors have no ground-truth labels, so no single number proves
they work. This runs several complementary checks:

  1. Baseline        -- known-event recall, and how unlikely that is by chance
                        (hypergeometric test against random 5% flag sets).
  2. Sensitivity     -- the same, across contamination thresholds 1%..10%.
  3. Walk-forward    -- train only on past years, flag the next year, repeat:
                        no look-ahead, as if the model had been run live.
  4. Stability       -- re-run with other random seeds and on 80% subsamples;
                        how much do the flagged days change?
  5. Injection       -- plant synthetic shocks / volatility bursts / drifts of
                        known size into calm stretches of the real price series,
                        rebuild features, refit, and measure how many are caught
                        (against un-injected control stretches).
  6. Separation      -- are flagged days statistically different on raw market
                        measures, and is volatility higher in the days after?
  7. Agreement       -- rank correlation between the individual models.

Usage:
    python validate_models.py                # NIFTY50, full run (a few minutes)
    python validate_models.py NIFTY50 --quick
Writes:
    db/validation/<SECTOR>_model_validation.md   (report)
    db/validation/<SECTOR>_model_validation.json (key numbers)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import train_multi_model as tm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from build_features import build_features  # noqa: E402

VALIDATION_DIR = tm.PROCESSED_DIR.parent / "validation"
CONTAMINATION = tm.FINAL_CONTAMINATION
SETS = ("shock", "regime")
MODELS = tm.ENSEMBLE_MODELS
DETERMINISTIC = ("ocsvm", "lof", "knn")   # unaffected by the random seed

SENSITIVITY_RATES = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
MIN_TRAIN_ROWS = 500                      # walk-forward: first test year needs this much history

SHOCK_SIGMAS = (2, 3, 4, 5, 6)            # injected jump size, in units of 20-day volatility
SHOCK_DAYS_PER_TRIAL = 10
REGIME_SCENARIOS = (                      # (label, kind, size)
    ("volatility x2", "burst", 2.0), ("volatility x3", "burst", 3.0), ("volatility x4", "burst", 4.0),
    ("drift 0.4%/day", "drift", 0.004), ("drift 0.7%/day", "drift", 0.007), ("drift 1.0%/day", "drift", 0.010),
)
REGIME_WINDOW = 20                        # trading days per injected period
REGIME_LAG = 10                           # regime features lag; also count flags this soon after
REGIME_WINDOWS_PER_TRIAL = 6


# ============================================================================
# Helpers
# ============================================================================

def fit_scores(X, X_new=None, seed=42, models=MODELS) -> dict:
    return {m: tm.MODELS[m](X, X_new, seed=seed)[1] for m in models}


def rank_ensemble(scores: dict) -> np.ndarray:
    return np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in scores.values()], axis=0)


def top_share(score: np.ndarray, rate: float = CONTAMINATION) -> np.ndarray:
    return score >= np.quantile(score, 1 - rate)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def event_positions(df: pd.DataFrame, events: pd.DataFrame) -> list[int]:
    pos = {d: i for i, d in enumerate(df["date"].dt.strftime("%Y-%m-%d"))}
    return [pos[d] for d in events["date"] if d in pos]


def caught(flags: np.ndarray, positions: list[int], tolerance: int = 0) -> int:
    return sum(flags[max(0, p - tolerance): p + tolerance + 1].any() for p in positions)


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


# ============================================================================
# 1-2. Baseline and sensitivity
# ============================================================================

def baseline(comparisons: dict, events_pos: list[int]) -> tuple[list[str], dict]:
    lines, data = [], {}
    rows = []
    for fs in ("shock", "regime", "all"):
        df = comparisons[fs]
        flags = df["ensemble_anomaly"].to_numpy()
        n, k, m = len(df), int(flags.sum()), len(events_pos)
        x = caught(flags, events_pos)
        p = stats.hypergeom.sf(x - 1, n, k, m)
        data[fs] = {"flagged": k, "events": m, "caught": x, "caught_pm1": caught(flags, events_pos, 1), "p_value": p}
        rows.append([fs, k, f"{x}/{m}", f"{caught(flags, events_pos, 1)}/{m}",
                     f"{x / m / (k / n):.1f}x", f"{p:.1e}"])
    lines += table(["Feature set", "Flagged days", "Events (exact day)", "Events (±1 day)",
                    "Lift over random", "p-value (random 5% flags)"], rows)
    return lines, data


def sensitivity(comparisons: dict, events_pos: list[int]) -> tuple[list[str], dict]:
    lines, data = [], {}
    for fs in SETS:
        score = comparisons[fs]["ensemble_score"].to_numpy()
        rows, d = [], []
        for rate in SENSITIVITY_RATES:
            flags = top_share(score, rate)
            x, x1 = caught(flags, events_pos), caught(flags, events_pos, 1)
            rows.append([pct(rate, 1), int(flags.sum()), f"{x}/{len(events_pos)}", f"{x1}/{len(events_pos)}"])
            d.append({"rate": rate, "flagged": int(flags.sum()), "caught": x, "caught_pm1": x1})
        data[fs] = d
        lines += [f"**{fs.capitalize()} ensemble**", ""]
        lines += table(["Contamination", "Flagged days", "Events (exact)", "Events (±1 day)"], rows) + [""]
    return lines, data


# ============================================================================
# 3. Walk-forward (expanding window, yearly refit)
# ============================================================================

def walk_forward(features: pd.DataFrame, comparisons: dict, events_pos: list[int]) -> tuple[list[str], dict]:
    years = features["date"].dt.year
    test_years = [y for y in sorted(years.unique()) if (years < y).sum() >= MIN_TRAIN_ROWS]
    lines, data = [], {}

    for fs in SETS:
        cols = tm.FEATURE_SETS[fs]
        oos_flags = np.zeros(len(features), dtype=bool)
        oos_score = np.full(len(features), np.nan)
        tested = np.zeros(len(features), dtype=bool)
        per_year = []

        for y in test_years:
            tr, te = (years < y).to_numpy(), (years == y).to_numpy()
            Xtr, Xte = features.loc[tr, cols].to_numpy(), features.loc[te, cols].to_numpy()
            train_scores = fit_scores(Xtr)
            test_scores = fit_scores(Xtr, Xte)
            # Rank each test day against the *training* score distribution, so the
            # threshold is fixed before seeing the test year.
            test_pct = np.mean([np.searchsorted(np.sort(train_scores[m]), test_scores[m], side="right")
                                / len(train_scores[m]) for m in MODELS], axis=0)
            threshold = np.quantile(rank_ensemble(train_scores), 1 - CONTAMINATION)
            flags = test_pct >= threshold
            oos_flags[te], oos_score[te], tested[te] = flags, test_pct, True
            in_sample = comparisons[fs]["ensemble_anomaly"].to_numpy()[te]
            per_year.append([y, int(tr.sum()), int(te.sum()), int(flags.sum()), pct(flags.mean(), 1),
                             int(in_sample.sum()), f"{jaccard(flags, in_sample):.2f}"])

        test_events = [p for p in events_pos if tested[p]]
        x_oos = caught(oos_flags, test_events)
        in_flags = comparisons[fs]["ensemble_anomaly"].to_numpy()
        x_in = caught(in_flags, test_events)
        rho = stats.spearmanr(oos_score[tested], comparisons[fs]["ensemble_score"].to_numpy()[tested]).statistic
        agree = jaccard(oos_flags[tested], in_flags[tested])
        prec = (oos_flags & in_flags)[tested].sum() / max(oos_flags[tested].sum(), 1)
        rec = (oos_flags & in_flags)[tested].sum() / max(in_flags[tested].sum(), 1)
        data[fs] = {"test_years": test_years, "events_tested": len(test_events), "caught_oos": x_oos,
                    "caught_in_sample": x_in, "spearman": rho, "jaccard": agree,
                    "oos_flag_rate": float(oos_flags[tested].mean()),
                    "share_of_oos_flags_also_in_sample": float(prec), "share_of_in_sample_flags_found_oos": float(rec)}
        lines += [
            f"**{fs.capitalize()} ensemble** -- tested {test_years[0]}-{test_years[-1]}, "
            f"{int(tested.sum())} days",
            "",
            *table(["Test year", "Train days", "Test days", "Flagged (OOS)", "Flag rate",
                    "Flagged (in-sample)", "Jaccard vs in-sample"], per_year),
            "",
            f"* Known events in test years caught out-of-sample: **{x_oos}/{len(test_events)}** "
            f"(in-sample model: {x_in}/{len(test_events)})",
            f"* Rank correlation of out-of-sample vs in-sample scores: **{rho:.2f}** (Spearman)",
            f"* Out-of-sample flags that the full-history model also flags: {pct(prec)}; "
            f"full-history flags recovered out-of-sample: {pct(rec)}; overall flag rate {pct(oos_flags[tested].mean(), 1)} "
            f"(target {pct(CONTAMINATION)})",
            "",
        ]
    return lines, data


# ============================================================================
# 4. Stability
# ============================================================================

def stability(features: pd.DataFrame, comparisons: dict, n_seeds: int, n_boot: int,
              rng: np.random.Generator) -> tuple[list[str], dict]:
    lines, data = [], {}
    for fs in SETS:
        X = features[tm.FEATURE_SETS[fs]].to_numpy()
        base_scores = fit_scores(X)
        base_ens = top_share(rank_ensemble(base_scores))

        # Seeds: only the randomised models change.
        seed_j = {"ensemble": [], "isoforest": [], "elliptic": []}
        for seed in range(n_seeds):
            scores = dict(base_scores)
            scores.update(fit_scores(X, seed=seed, models=("isoforest", "elliptic")))
            seed_j["ensemble"].append(jaccard(top_share(rank_ensemble(scores)), base_ens))
            for m in ("isoforest", "elliptic"):
                seed_j[m].append(jaccard(top_share(scores[m]), top_share(base_scores[m])))

        # Subsamples: fit on a random 80% of days, score every day.
        boot_j = {m: [] for m in [*MODELS, "ensemble"]}
        for _ in range(n_boot):
            idx = np.sort(rng.choice(len(X), int(0.8 * len(X)), replace=False))
            scores = fit_scores(X[idx], X)
            for m in MODELS:
                boot_j[m].append(jaccard(top_share(scores[m]), top_share(base_scores[m])))
            boot_j["ensemble"].append(jaccard(top_share(rank_ensemble(scores)), base_ens))

        fmt = lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}"  # noqa: E731
        rows = [[m, fmt(seed_j[m]) if m in seed_j else "deterministic", fmt(boot_j[m])]
                for m in [*MODELS, "ensemble"]]
        data[fs] = {"seed_jaccard": {k: float(np.mean(v)) for k, v in seed_j.items()},
                    "subsample_jaccard": {k: float(np.mean(v)) for k, v in boot_j.items()}}
        lines += [f"**{fs.capitalize()}**", "",
                  *table(["Model", f"Seed change ({n_seeds} seeds)", f"80% subsample ({n_boot} runs)"], rows), ""]
    lines.append("Values are the Jaccard overlap of flagged days with the production run "
                 "(1.00 = identical set of days).")
    return lines, data


# ============================================================================
# 5. Synthetic anomaly injection
# ============================================================================

def spaced_choice(candidates: np.ndarray, k: int, spacing: int, rng: np.random.Generator) -> list[int]:
    chosen = []
    for c in rng.permutation(candidates):
        if all(abs(c - o) >= spacing for o in chosen):
            chosen.append(int(c))
            if len(chosen) == 2 * k:   # k to inject + k untouched controls
                break
    return chosen


def ensemble_flags(features: pd.DataFrame, fs: str) -> tuple[np.ndarray, dict]:
    scores = fit_scores(features[tm.FEATURE_SETS[fs]].to_numpy())
    return top_share(rank_ensemble(scores)), {m: top_share(s) for m, s in scores.items()}


def inject_shocks(prices: pd.DataFrame, features: pd.DataFrame, flags: np.ndarray, trials: int,
                  rng: np.random.Generator) -> tuple[list[str], dict]:
    calm = np.flatnonzero(~flags & (features["return_zscore"].abs() < 1).to_numpy())
    calm = calm[(calm > 60) & (calm < len(features) - 5)]
    price_pos = {d: i for i, d in enumerate(prices["date"])}
    rows, data = [], []
    for k in SHOCK_SIGMAS:
        hit = {m: [] for m in [*MODELS, "ensemble"]}
        control = []
        for _ in range(trials):
            picks = spaced_choice(calm, SHOCK_DAYS_PER_TRIAL, 30, rng)
            inject, ctrl = picks[:SHOCK_DAYS_PER_TRIAL], picks[SHOCK_DAYS_PER_TRIAL:]
            p = prices.copy()
            for i in inject:
                # Permanent jump of k x (that day's 20-day volatility), up or down, from
                # the open onward -- like a gap on news. Later returns are unchanged.
                sigma = features["volatility_20d"].iloc[i]
                factor = np.exp(rng.choice([-1, 1]) * k * sigma)
                start = price_pos[features["date"].iloc[i]]
                p.loc[start:, ["open", "high", "low", "close"]] *= factor
            f = build_features(p)
            ens, per_model = ensemble_flags(f, "shock")
            hit["ensemble"].append(ens[inject].mean())
            for m in MODELS:
                hit[m].append(per_model[m][inject].mean())
            control.append(ens[ctrl].mean())
        rows.append([f"{k} σ", *[pct(np.mean(hit[m])) for m in MODELS], f"**{pct(np.mean(hit['ensemble']))}**",
                     pct(np.mean(control))])
        data.append({"sigma": k, "detected": float(np.mean(hit["ensemble"])), "control": float(np.mean(control)),
                     "per_model": {m: float(np.mean(hit[m])) for m in MODELS}})
    lines = table(["Jump size", *MODELS, "Ensemble", "Untouched control days"], rows)
    return lines, data


def inject_regimes(prices: pd.DataFrame, features: pd.DataFrame, flags: np.ndarray, trials: int,
                   rng: np.random.Generator) -> tuple[list[str], dict]:
    # Calm = no regime flag anywhere near the window.
    near_flag = np.convolve(flags.astype(int), np.ones(2 * REGIME_WINDOW + REGIME_LAG), mode="same") > 0
    starts = np.flatnonzero(~near_flag)
    starts = starts[(starts > 80) & (starts < len(features) - REGIME_WINDOW - REGIME_LAG - 1)]
    price_pos = {d: i for i, d in enumerate(prices["date"])}
    log_close = np.log(prices["close"].to_numpy())
    rows, data = [], []

    for label, kind, size in REGIME_SCENARIOS:
        hit, share, control = [], [], []
        for _ in range(trials):
            picks = spaced_choice(starts, REGIME_WINDOWS_PER_TRIAL, 2 * REGIME_WINDOW + 40, rng)
            inject, ctrl = picks[:REGIME_WINDOWS_PER_TRIAL], picks[REGIME_WINDOWS_PER_TRIAL:]
            new_log = log_close.copy()
            for i in inject:
                a = price_pos[features["date"].iloc[i]]
                b = a + REGIME_WINDOW
                r = np.diff(log_close[a - 1:b])               # the window's daily log returns
                r = r * size if kind == "burst" else r + rng.choice([-1, 1]) * size
                shift = (new_log[a - 1] + np.cumsum(r)) - new_log[a:b]
                new_log[a:b] += shift
                new_log[b:] += shift[-1]                      # later prices move with the new level
            p = prices.copy()
            ratio = np.exp(new_log - log_close)
            for c in ("open", "high", "low", "close"):        # keep each day's intraday shape
                p[c] = p[c] * ratio
            f = build_features(p)
            ens, _ = ensemble_flags(f, "regime")

            def detected(i):
                return ens[i: i + REGIME_WINDOW + REGIME_LAG]
            hit.append(np.mean([detected(i).any() for i in inject]))
            share.append(np.mean([detected(i).mean() for i in inject]))
            control.append(np.mean([detected(i).any() for i in ctrl]))
        rows.append([label, f"**{pct(np.mean(hit))}**", pct(np.mean(share)), pct(np.mean(control))])
        data.append({"scenario": label, "detected": float(np.mean(hit)), "day_share": float(np.mean(share)),
                     "control": float(np.mean(control))})
    lines = table(["Injected period (20 days)", "Periods detected", "Share of days flagged",
                   "Untouched control periods"], rows)
    return lines, data


# ============================================================================
# 6. Separation and predictive validity
# ============================================================================

def separation(features: pd.DataFrame, comparisons: dict) -> tuple[list[str], dict]:
    r = features["log_return"].to_numpy()
    fwd5 = pd.Series(r).rolling(5).std().shift(-5).to_numpy()
    fwd20 = pd.Series(r).rolling(20).std().shift(-20).to_numpy()
    measures = {
        "|daily return|": np.abs(r),
        "intraday range": features["range_vol"].to_numpy(),
        "|opening gap|": features["gap"].abs().to_numpy(),
        "volatility, next 5 days": fwd5,
        "volatility, next 20 days": fwd20,
    }
    lines, data = [], {}
    for fs in SETS:
        flags = comparisons[fs]["ensemble_anomaly"].to_numpy()
        rows, d = [], {}
        for name, v in measures.items():
            ok = ~np.isnan(v)
            a, b = v[ok & flags], v[ok & ~flags]
            u, p = stats.mannwhitneyu(a, b, alternative="greater")
            effect = 2 * u / (len(a) * len(b)) - 1     # rank-biserial correlation
            rows.append([name, f"{np.median(a) * 100:.2f}%", f"{np.median(b) * 100:.2f}%",
                         f"{np.median(a) / np.median(b):.1f}x", f"{effect:.2f}", f"{p:.1e}"])
            d[name] = {"median_flagged": float(np.median(a)), "median_other": float(np.median(b)),
                       "effect": float(effect), "p": float(p)}
        data[fs] = d
        lines += [f"**{fs.capitalize()} ensemble**", "",
                  *table(["Measure", "Median, flagged", "Median, other days", "Ratio",
                          "Effect size (rank-biserial)", "p (Mann-Whitney, one-sided)"], rows), ""]
    lines.append("Forward volatility is measured over the days *after* each day, which the models never see. "
                 "Daily observations overlap and are autocorrelated, so p-values are optimistic; "
                 "the effect sizes and ratios are the more reliable read.")
    return lines, data


# ============================================================================
# 7. Agreement between models
# ============================================================================

def agreement(comparisons: dict) -> tuple[list[str], dict]:
    lines, data = [], {}
    for fs in SETS:
        df = comparisons[fs]
        corr = df[[f"{m}_score" for m in MODELS]].corr(method="spearman").to_numpy()
        rows = [[a, *[f"{corr[i, j]:.2f}" for j in range(len(MODELS))]] for i, a in enumerate(MODELS)]
        off = corr[np.triu_indices(len(MODELS), 1)]
        data[fs] = {"mean_pairwise_spearman": float(off.mean())}
        lines += [f"**{fs.capitalize()}** (mean pairwise ρ = {off.mean():.2f})", "",
                  *table(["", *MODELS], rows), ""]
    return lines, data


# ============================================================================
# Report
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate the anomaly ensembles.")
    parser.add_argument("sector", nargs="?", default="NIFTY50")
    parser.add_argument("--quick", action="store_true", help="fewer seeds/subsamples/trials")
    args = parser.parse_args()
    sector = args.sector
    n_seeds, n_boot, trials = (3, 3, 1) if args.quick else (10, 20, 3)
    rng = np.random.default_rng(7)
    t0 = time.time()

    features = pd.read_csv(tm.PROCESSED_DIR / f"{sector}_features.csv", parse_dates=["date"])
    prices = pd.read_csv(tm.PROCESSED_DIR / f"{sector}_merged.csv", parse_dates=["date"])
    comparisons = {fs: pd.read_csv(tm.PROCESSED_DIR / f"{sector}_{fs}_model_comparison.csv", parse_dates=["date"])
                   for fs in ("shock", "regime", "all")}
    events = pd.read_csv(tm.EVENTS_DIR / f"{sector}_known_events.csv")
    events_pos = event_positions(features, events)

    rebuilt = build_features(prices)
    assert len(rebuilt) == len(features), "features file is out of date -- rerun build_features.py"

    def step(name, fn, *a):
        t = time.time()
        print(f"{name}...", end=" ", flush=True)
        out = fn(*a)
        print(f"{time.time() - t:.0f}s")
        return out

    base_lines, base = step("Baseline", baseline, comparisons, events_pos)
    sens_lines, sens = step("Sensitivity", sensitivity, comparisons, events_pos)
    wf_lines, wf = step("Walk-forward", walk_forward, features, comparisons, events_pos)
    stab_lines, stab = step("Stability", stability, features, comparisons, n_seeds, n_boot, rng)
    shock_lines, shock_inj = step("Shock injection", inject_shocks, prices, features,
                                  comparisons["shock"]["ensemble_anomaly"].to_numpy(), trials, rng)
    regime_lines, regime_inj = step("Regime injection", inject_regimes, prices, features,
                                    comparisons["regime"]["ensemble_anomaly"].to_numpy(), trials, rng)
    sep_lines, sep = step("Separation", separation, features, comparisons)
    agr_lines, agr = step("Agreement", agreement, comparisons)

    s3 = next(d for d in shock_inj if d["sigma"] == 3)
    s5 = next(d for d in shock_inj if d["sigma"] == 5)
    summary = [
        f"* **Known events:** the shock ensemble flags {base['shock']['caught']}/{base['shock']['events']} on the exact day "
        f"(p = {base['shock']['p_value']:.0e} vs. random 5% flags); the all-features model {base['all']['caught']}/{base['all']['events']}.",
        f"* **No look-ahead (walk-forward):** trained only on earlier years, the shock ensemble still catches "
        f"{wf['shock']['caught_oos']}/{wf['shock']['events_tested']} events in its test years; out-of-sample and "
        f"full-history scores have rank correlation {wf['shock']['spearman']:.2f} (shock) / {wf['regime']['spearman']:.2f} (regime).",
        f"* **Stability:** across seeds the ensemble's flagged days overlap {stab['shock']['seed_jaccard']['ensemble']:.2f} (shock) "
        f"/ {stab['regime']['seed_jaccard']['ensemble']:.2f} (regime) with production; on 80% subsamples "
        f"{stab['shock']['subsample_jaccard']['ensemble']:.2f} / {stab['regime']['subsample_jaccard']['ensemble']:.2f}.",
        f"* **Synthetic shocks:** {pct(s3['detected'])} of injected 3σ jumps and {pct(s5['detected'])} of 5σ jumps are flagged "
        f"(untouched calm days: {pct(s3['control'])}).",
        "* **Synthetic regimes:** " + "; ".join(f"{d['scenario']} {pct(d['detected'])}" for d in regime_inj)
        + f" detected (control periods ~{pct(np.mean([d['control'] for d in regime_inj]))}).",
        f"* **Forward volatility:** after shock-flagged days, the next 20 days are "
        f"{sep['shock']['volatility, next 20 days']['median_flagged'] / sep['shock']['volatility, next 20 days']['median_other']:.1f}x "
        f"as volatile as after other days (regime: "
        f"{sep['regime']['volatility, next 20 days']['median_flagged'] / sep['regime']['volatility, next 20 days']['median_other']:.1f}x).",
    ]

    report = [
        f"# Model validation -- {sector}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `backend/src/models/validate_models.py`"
        f"{' (quick mode)' if args.quick else ''}. Ensembles: rank-average of {', '.join(MODELS)}; "
        f"top {pct(CONTAMINATION)} of days flagged. {len(events_pos)} reference events from "
        f"`db/events/{sector}_known_events.csv`.",
        "",
        "## Summary", "", *summary, "",
        "## 1. Known events and chance level", "",
        "How many reference events fall on a flagged day, and the probability of doing at least that well "
        "by flagging the same number of days at random (hypergeometric test).", "",
        *base_lines, "",
        "The reference list is small (13 events) and was used when choosing the feature split, so treat it "
        "as a sanity check rather than an unbiased accuracy estimate -- the walk-forward and injection tests below "
        "don't depend on it.", "",
        "## 2. Sensitivity to the contamination threshold", "",
        *sens_lines,
        "## 3. Walk-forward (out-of-sample) test", "",
        "For each year, every model is trained only on the years before it, and the test year's days are ranked "
        "against the training scores with the threshold fixed in advance -- exactly what a live system could have done. "
        "The production model instead fits all history at once.", "",
        *wf_lines,
        "## 4. Stability", "",
        *stab_lines, "",
        "## 5. Synthetic anomaly injection", "",
        "Known-size anomalies are planted in calm stretches of the real NIFTY 50 price series; features are rebuilt "
        f"and the ensemble refitted ({trials} trial(s) per row). Control stretches are equally calm but left untouched, "
        "so their rate is the false-alarm baseline.", "",
        f"**Shocks** -- a permanent jump of k × the day's 20-day volatility from the open ({SHOCK_DAYS_PER_TRIAL} days per trial):", "",
        *shock_lines, "",
        f"**Regimes** -- {REGIME_WINDOW} trading days with returns scaled (volatility burst) or shifted (drift); "
        f"a period counts as detected if any day in it or the {REGIME_LAG} days after is flagged "
        f"({REGIME_WINDOWS_PER_TRIAL} periods per trial):", "",
        *regime_lines, "",
        "## 6. Separation and forward-looking validity", "",
        *sep_lines, "",
        "## 7. Agreement between models (Spearman rank correlation of scores)", "",
        *agr_lines,
        f"Runtime {time.time() - t0:.0f}s.",
    ]

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    md = VALIDATION_DIR / f"{sector}_model_validation.md"
    md.write_text("\n".join(report) + "\n", encoding="utf-8")
    numbers = {"baseline": base, "sensitivity": sens, "walk_forward": wf, "stability": stab,
               "shock_injection": shock_inj, "regime_injection": regime_inj, "separation": sep, "agreement": agr}
    (VALIDATION_DIR / f"{sector}_model_validation.json").write_text(
        json.dumps(numbers, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nWrote {md}")


if __name__ == "__main__":
    main()
