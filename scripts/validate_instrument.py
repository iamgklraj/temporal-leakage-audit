"""Validate the leakage-response instrument on synthetic data with known leakage.

The synthetic generator (temporal_leakage_audit/data/synth.py) posts "literature" evidence
mostly after each program's decision time. ``leak_strength`` sets how strongly those
post-decision scores track the label; with ``leak_strength = 0`` and ``leak_count = 0`` the
post-decision evidence is independent of the label, so there is no leakage to find.

For each leak level we draw independent replicates (new data, new split, new models) and
compute LAP = AUPRC(naive) - AUPRC(deployable) with its target-clustered, paired percentile
bootstrap CI, exactly as in the paper. Reported per level:
  * mean LAP and the 2.5-97.5% range across replicates (does LAP grow with true leakage?);
  * detection rate = fraction of replicates whose CI lies above 0 (false-positive rate at
    level 0, power elsewhere);
  * coverage = fraction of replicate CIs containing the Monte Carlo mean LAP of that level
    (calibration of the test-set bootstrap, which conditions on the fitted models).
An example full leakage-response curve (default synthetic settings) is also stored.

Usage:  python scripts/validate_instrument.py [--reps 100] [--workers 6]
Writes outputs/instrument_validation.json.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

# One thread per worker process: the gradient-boosted learner is OpenMP-parallel, and
# N workers x all cores oversubscribes the machine badly.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from temporal_leakage_audit import features, leakage, metrics, models, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402

LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
N_BOOT = 200


def _cfg(leak_strength, leak_count, seed, n_programs=2000):
    cfg = get_config()
    cfg["seed"] = seed
    cfg["synth"].update({"n_programs": n_programs, "n_targets": int(n_programs * 0.75),
                         "leak_strength": leak_strength, "leak_count": leak_count})
    return cfg


def replicate(args):
    level, rep = args
    cfg = _cfg(level, 0, seed=10_000 + 1_000 * LEVELS.index(level) + rep)
    programs, evidence = synth.generate(cfg)
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    tr, te, info = splits.temporal_split(meta, 2016, (2017, 2020), enforce_target_disjoint=True)
    ytr, yte = y.loc[tr].values, y.loc[te].values
    p = {}
    for name, X in (("naive", Xn), ("cens", Xc)):
        clf = models.fit_gbdt(X.loc[tr], ytr, seed=0)
        p[name] = models.predict_gbdt(clf, X.loc[te])
    s = leakage._clustered_auprc_samples(yte, p, meta.loc[te, "target_id"].values,
                                         n_boot=N_BOOT, seed=rep)
    draws = np.array(s["naive"]) - np.array(s["cens"])
    return {"level": level, "rep": rep, "n_test": int(len(te)),
            "lap": float(metrics.auprc(yte, p["naive"]) - metrics.auprc(yte, p["cens"])),
            "lap_auroc": float(metrics.auroc(yte, p["naive"]) - metrics.auroc(yte, p["cens"])),
            "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975))}


def summarize(rows):
    out = []
    for level in LEVELS:
        r = [x for x in rows if x["level"] == level]
        lap = np.array([x["lap"] for x in r])
        lo, hi = np.array([x["lo"] for x in r]), np.array([x["hi"] for x in r])
        mean = float(lap.mean())
        out.append({
            "leak_strength": level, "replicates": len(r),
            "mean_lap": round(mean, 4),
            "lap_range_95": [round(float(np.quantile(lap, .025)), 4), round(float(np.quantile(lap, .975)), 4)],
            "mean_lap_auroc": round(float(np.mean([x["lap_auroc"] for x in r])), 4),
            "detection_rate": round(float(np.mean(lo > 0)), 3),
            "ci_below_zero_rate": round(float(np.mean(hi < 0)), 3),
            "coverage_of_mean_lap": round(float(np.mean((lo <= mean) & (mean <= hi))), 3),
            "mean_n_test": round(float(np.mean([x["n_test"] for x in r])), 1),
        })
    return out


def example_curve():
    cfg = _cfg(get_config()["synth"]["leak_strength"], 2, seed=7, n_programs=4000)
    programs, evidence = synth.generate(cfg)
    _, _, _, meta, _ = features.build(programs, evidence)
    tr, te, info = splits.temporal_split(meta, 2016, (2017, 2020), enforce_target_disjoint=True)
    c = leakage.leakage_response_curve(programs, evidence, tr, te,
                                       clusters=meta.loc[te, "target_id"].values, n_boot=500)
    return {"n_train": info["n_train"], "n_test": info["n_test"],
            **{k: c[k] for k in ("curve", "test_base_rate", "deployable_auprc", "naive_auprc",
                                 "total_LAP_auprc", "total_LAP_auprc_ci", "leakage_half_life_years")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="outputs/instrument_validation.json")
    args = ap.parse_args()
    jobs = [(lv, r) for lv in LEVELS for r in range(args.reps)]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(replicate, jobs, chunksize=4))
    summary = summarize(rows)
    out = {"design": {"levels_leak_strength": LEVELS, "leak_count": 0,
                      "replicates_per_level": args.reps, "n_programs": 2000,
                      "split": "train <= 2016, test 2017-2020, target-disjoint",
                      "bootstrap": f"target-clustered paired percentile, {N_BOOT} resamples"},
           "levels": summary, "example_curve": example_curve()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for s in summary:
        print(f"leak {s['leak_strength']:>4}: mean LAP {s['mean_lap']:+.3f} {s['lap_range_95']}  "
              f"detect {s['detection_rate']:.2f}  CI<0 {s['ci_below_zero_rate']:.2f}  "
              f"coverage {s['coverage_of_mean_lap']:.2f}  (n_test ~{s['mean_n_test']:.0f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
