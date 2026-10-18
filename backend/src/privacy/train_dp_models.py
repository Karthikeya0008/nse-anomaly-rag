"""
Train the differentially private anomaly ensembles and evaluate the privacy cost.

    python train_dp_models.py                    # NIFTY50, epsilon 2 per feature set
    python train_dp_models.py NIFTY50 --epsilon 1

Writes:
    db/models/dp/<SECTOR>_{shock,regime}_dp.joblib     shareable DP models (no training rows)
    db/models/dp/<SECTOR>_privacy_ledger.json          epsilon spent, per component
    db/processed/<SECTOR>_{shock,regime}_dp_comparison.csv
    db/validation/<SECTOR>_dp_report.md                leakage demo + privacy-utility curve
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import train_multi_model as tm  # noqa: E402
from dp_detectors import BUDGET_SPLIT, PUBLIC_BOUNDS, PUBLIC_SCALES, DPEnsemble  # noqa: E402

DP_MODELS_DIR = tm.MODELS_DIR / "dp"
VALIDATION_DIR = tm.PROCESSED_DIR.parent / "validation"
SETS = ("shock", "regime")
DEFAULT_EPSILON = 2.0
SWEEP_EPSILONS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, None)   # None = same detectors, no noise
SWEEP_REPEATS = 10
# A raw closing price enters the features of that day and the next 60 trading
# days (60-day windows), so it influences at most this many feature rows.
ROWS_PER_PRICE = 61


def jaccard(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(str(c) for c in r) + " |" for r in rows]]


# ----------------------------------------------------------------------------
# Leakage demonstration: what the non-DP model files reveal
# ----------------------------------------------------------------------------

def leakage_demo(sector: str, features: pd.DataFrame) -> list[str]:
    rows = []
    for fs in SETS:
        X = features[tm.FEATURE_SETS[fs]].to_numpy()
        for name, step in (("knn", "nearestneighbors"), ("lof", "localoutlierfactor")):
            path = tm.MULTI_MODELS_DIR / f"{sector}_{fs}_{name}.joblib"
            if not path.exists():
                continue
            pipe = joblib.load(path)
            stored = pipe.named_steps[step]._fit_X            # training rows, as fitted
            recovered = pipe.named_steps["robustscaler"].inverse_transform(stored)
            exact = np.isclose(recovered, X, rtol=0, atol=1e-9).all(axis=1).mean()
            rows.append([f"{fs} / {name}", f"{len(stored)} of {len(X)}", f"{exact:.0%}"])
        ocsvm = joblib.load(tm.MULTI_MODELS_DIR / f"{sector}_{fs}_ocsvm.joblib")
        svs = ocsvm.named_steps["oneclasssvm"].support_vectors_
        rows.append([f"{fs} / ocsvm", f"{len(svs)} support vectors", "these rows, exactly"])
    return [
        "Fitted non-DP models were loaded from `db/models/multi/` and inverted with their own saved scaler:", "",
        *table(["Model file", "Training rows stored", "Days whose features are recovered exactly"], rows), "",
        "LOF and kNN score new days by comparing them with stored training days, so the file *is* the "
        "training data; One-Class SVM keeps its support vectors, which are training days -- the most "
        "unusual ones. Isolation Forest stores split thresholds drawn between the minimum and maximum of "
        "sub-samples, which leak value ranges. The DP models below store only noisy counts, noisy means "
        "and spreads, random data-independent splits and noisy score distributions.", "",
    ]


# ----------------------------------------------------------------------------
# Privacy-utility sweep
# ----------------------------------------------------------------------------

def sweep(features: pd.DataFrame, production: dict, events_pos: list[int]) -> tuple[list[str], dict]:
    lines, data = [], {}
    for fs in SETS:
        X = features[tm.FEATURE_SETS[fs]].to_numpy()
        prod = production[fs]
        rows, d = [], []
        for eps in SWEEP_EPSILONS:
            stats = []
            for seed in range(SWEEP_REPEATS):
                model = DPEnsemble(tm.FEATURE_SETS[fs], eps if eps else 1e9, seed=1000 + seed).fit(X)
                flags, _, _ = model.predict(X)
                stats.append((flags.mean(), jaccard(flags, prod), flags[events_pos].sum()))
            s = np.array(stats)
            label = f"{eps:g}" if eps else "∞ (no noise)"
            rows.append([label, f"{s[:, 0].mean():.1%}", f"{s[:, 1].mean():.2f} ± {s[:, 1].std():.2f}",
                         f"{s[:, 2].mean():.1f} / {len(events_pos)}"])
            d.append({"epsilon": eps, "flag_rate": s[:, 0].mean(), "jaccard": s[:, 1].mean(),
                      "jaccard_sd": s[:, 1].std(), "events": s[:, 2].mean()})
        data[fs] = d
        lines += [f"**{fs.capitalize()} ensemble** ({SWEEP_REPEATS} independent noise draws per row)", "",
                  *table(["ε (per feature set)", "Days flagged", "Overlap with production flags (Jaccard)",
                          "Reference events caught"], rows), ""]
    return lines, data


def main():
    parser = argparse.ArgumentParser(description="Train DP anomaly ensembles.")
    parser.add_argument("sector", nargs="?", default="NIFTY50")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="budget per feature set")
    parser.add_argument("--no-sweep", action="store_true", help="skip the privacy-utility sweep")
    args = parser.parse_args()
    sector, eps = args.sector, args.epsilon

    features = pd.read_csv(tm.PROCESSED_DIR / f"{sector}_features.csv", parse_dates=["date"])
    production = {fs: pd.read_csv(tm.PROCESSED_DIR / f"{sector}_{fs}_model_comparison.csv")["ensemble_anomaly"].to_numpy()
                  for fs in SETS}
    events = pd.read_csv(tm.EVENTS_DIR / f"{sector}_known_events.csv")
    pos = {d: i for i, d in enumerate(features["date"].dt.strftime("%Y-%m-%d"))}
    events_pos = [pos[d] for d in events["date"] if d in pos]
    DP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -- deployed models (fresh randomness) -----------------------------------
    ledger = {"sector": sector, "unit_of_privacy": "one trading day's feature row (add/remove)",
              "epsilon_per_feature_set": eps, "feature_sets": {}, "generated": datetime.now().isoformat(timespec="seconds"),
              "budget_split": BUDGET_SPLIT,
              "public_bounds": {f: PUBLIC_BOUNDS[f] for fs in SETS for f in tm.FEATURE_SETS[fs]},
              "public_scales": {f: PUBLIC_SCALES[f] for fs in SETS for f in tm.FEATURE_SETS[fs]}}
    deployed_lines = []
    for fs in SETS:
        cols = tm.FEATURE_SETS[fs]
        X = features[cols].to_numpy()
        model = DPEnsemble(cols, eps).fit(X)
        joblib.dump(model, DP_MODELS_DIR / f"{sector}_{fs}_dp.joblib")
        flags, score, votes = model.predict(X)
        out = features[["date"]].copy()
        for name, s in model.detector_scores(X).items():
            out[f"{name}_score"] = s
        out["dp_score"], out["dp_votes"], out["dp_anomaly"] = score, votes, flags
        out.to_csv(tm.PROCESSED_DIR / f"{sector}_{fs}_dp_comparison.csv", index=False, date_format="%Y-%m-%d")
        ledger["feature_sets"][fs] = model.ledger.as_dict()

        clip = model.transform.clip_rate(X)
        deployed_lines.append([fs, f"{eps:g}", int(flags.sum()), f"{flags.mean():.1%}",
                               f"{jaccard(flags, production[fs]):.2f}",
                               f"{flags[events_pos].sum()} / {len(events_pos)}",
                               f"{max(clip.values()):.2%}"])
        print(f"{fs}: {flags.sum()} days flagged by the DP ensemble (eps={eps}); "
              f"Jaccard vs production {jaccard(flags, production[fs]):.2f}")

    total = round(sum(v["spent"] for v in ledger["feature_sets"].values()), 6)
    ledger["total_epsilon_per_trading_day"] = total
    ledger["total_epsilon_per_raw_price"] = total * ROWS_PER_PRICE
    (DP_MODELS_DIR / f"{sector}_privacy_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    # -- report ----------------------------------------------------------------
    ledger_rows = [[fs, e["component"], f"{e['epsilon']:.3f}", e["composition"], e["note"]]
                   for fs in SETS for e in ledger["feature_sets"][fs]["entries"]]
    if args.no_sweep:
        sweep_lines = ["(skipped)", ""]
    else:
        print("Running privacy-utility sweep...")
        sweep_lines, _ = sweep(features, production, events_pos)

    report = [
        f"# Differential privacy report -- {sector}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `backend/src/privacy/train_dp_models.py`.",
        "",
        "## What is protected",
        "",
        "* **Unit of privacy:** one trading day's feature row. A model is ε-DP if adding or removing any one "
        "day changes the probability of every possible fitted model by at most a factor e^ε.",
        "* **What it covers:** the fitted model files and everything derived from them (detector parameters, "
        "score distributions, thresholds). They can be shared, published or run by third parties without "
        "revealing the days they were trained on.",
        "* **What it does not cover:** the per-day flags. A day's own flag is computed from that day's own "
        "features, which is the point of anomaly detection; DP limits what the *model* reveals about the "
        "*other* days.",
        f"* **Raw prices:** one closing price feeds up to {ROWS_PER_PRICE} feature rows (60-day windows), so by "
        f"group privacy the guarantee for an individual price is ε × {ROWS_PER_PRICE}. Rolling features make "
        "row-level DP the meaningful unit here; for client- or trade-level data (see README) the unit would be "
        "the client.",
        "",
        "## 1. Why: the non-DP models leak their training data",
        "",
        *leakage_demo(sector, features),
        "## 2. Deployed DP models",
        "",
        *table(["Feature set", "ε", "Days flagged", "Flag rate", "Overlap with production (Jaccard)",
                "Reference events caught", "Max share of values clipped to public bounds"], deployed_lines),
        "",
        f"Shock and regime models are trained on the same days, so by sequential composition a trading day's "
        f"total privacy loss is **ε = {total:g}** ({total:g} × {ROWS_PER_PRICE} = {total * ROWS_PER_PRICE:g} "
        f"for a single raw price). These models were fitted with fresh random noise; re-running gives "
        f"slightly different flags.",
        "",
        "## 3. Privacy ledger",
        "",
        *table(["Set", "Component", "ε", "Composition", "Mechanism"], ledger_rows),
        "",
        "## 4. Privacy-utility trade-off",
        "",
        "Smaller ε = stronger privacy = more noise. \"∞\" runs the same three detectors without noise, which "
        "separates the cost of using DP-compatible detectors from the cost of the noise itself.",
        "",
        *sweep_lines,
        "## 5. Caveats",
        "",
        "* The feature bounds and scales are fixed constants (market rules, maths, typical magnitudes), but "
        "they were chosen by people who had seen this data. In a real deployment they must be fixed before "
        "looking at the private data.",
        "* `dp_mechanisms.py` is a transparent teaching implementation on numpy's floating-point Laplace "
        "sampler; production use should rely on a hardened library such as OpenDP.",
        "* The DP guarantee requires the noise to be secret: the deployed models use fresh OS randomness; "
        "fixed seeds are used only in the evaluation sweep above.",
    ]
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out = VALIDATION_DIR / f"{sector}_dp_report.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Total epsilon per trading day: {total:g}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
