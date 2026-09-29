"""Validate the leakage-response instrument at the size and prevalence of the real benchmark.

scripts/validate_instrument.py validates the instrument on synthetic test blocks of ~245
programs at prevalence ~0.5 with almost no target clustering (~1.2 programs per target). The
20-disease censored drug-program benchmark it is used to interpret is much harder: 200 test
programs, base rate 0.16 (32 positives), 96 target clusters (~2.1 programs per target), 2,425
training programs, train <= 2013. A detection limit measured in the first setting does not
transport to the second, so this script re-runs the validation in a matched setting.

Matched setting (existing generator arguments only; temporal_leakage_audit/data/synth.py):
  * years 2005-2020, split train <= 2013 / test 2014-2020, target-disjoint (as in the
    benchmark's main split, which also trains on <= 2013);
  * n_programs and n_targets chosen so the target-disjoint test block has ~200 programs in
    ~96 target clusters and the training block ~2,400 programs;
  * base_intercept tuned so the test prevalence is ~0.16.
Everything else (the instrument, the GBDT, the paired target-clustered percentile bootstrap
with 200 resamples) is exactly as in validate_instrument.py.

Arms (independent replicates: new data, new split, new models, new bootstrap):
  * null     : leak_strength = 0, leak_count = 0 (no leakage). Two-sided error of the 95% CI.
  * score    : mechanism 1, post-decision literature scores track the label (leak_strength).
  * count    : mechanism 2, successful programs attract extra post-decision papers (leak_count).
  * re_null / re_score : the null and three score levels with a target-level random intercept
    (SD 0.5) on the outcome logit (synth ``target_effect_sd``), which makes outcomes correlated
    within target clusters and so stresses the cluster bootstrap.

Per arm and level: mean LAP (AUPRC naive - deployable), detection rate (CI above 0), rate of
CI below 0, coverage of the Monte Carlo mean LAP, mean CI width, each with Monte Carlo SEs;
and, per mechanism, the mean LAP at which detection reaches 50% and 80% (linear interpolation
on the isotonic detection curve, MC SE by resampling replicates within level).

Usage:
  python scripts/validate_matched.py --pilot            # timing + level scan, prints only
  python scripts/validate_matched.py [--workers 4] [--checkpoint rows.jsonl]
Writes outputs/instrument_validation_matched.json.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

# One thread per worker process (the gradient-boosted learner is OpenMP-parallel).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from temporal_leakage_audit import features, leakage, metrics, models, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402

# The real benchmark this validation is matched to (outputs/censored_benchmark_20disease.json).
REAL = {"benchmark": "20-disease censored drug-program benchmark",
        "source": "outputs/censored_benchmark_20disease.json",
        "n_test": 200, "test_positives": 32, "test_base_rate": 0.16, "test_target_clusters": 96,
        "n_train": 2425, "split": "train <= 2013, test 2014-2027, target-disjoint",
        "observed_LAP_auprc": 0.0743, "observed_LAP_auprc_ci": [-0.0741, 0.1602],
        "observed_bootstrap": "target-clustered paired percentile, 500 resamples"}

MATCHED = {"n_programs": 4180, "n_targets": 1075, "year_min": 2005, "year_max": 2020}
INTERCEPT = -2.40          # base_intercept giving test prevalence ~0.16 (no target effect)
INTERCEPT_RE = -2.45       # same target prevalence with target_effect_sd = RE_SD
RE_SD = 0.5                # target-level random intercept SD (logit scale) for the stress arms
TRAIN_MAX, TEST_YEARS = 2013, (2014, 2020)
N_BOOT = 200               # as in validate_instrument.py

SCORE_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
COUNT_LEVELS = [1, 2, 3]
RE_SCORE_LEVELS = [1.0, 1.5, 2.0]

SEED_BASE = {"null": 3_000_000, "score": 4_000_000, "count": 5_000_000,
             "re_null": 6_000_000, "re_score": 7_000_000}


def job_seed(arm, level_index, rep):
    # Seeds spaced by 10 so each replicate's generator streams (seed, seed+1, seed+2) are unique.
    return SEED_BASE[arm] + 50_000 * level_index + 10 * rep


def make_cfg(seed, leak_strength, leak_count, intercept, target_sd):
    cfg = get_config()
    cfg["seed"] = seed
    cfg["synth"].update(dict(MATCHED, base_intercept=intercept, leak_strength=float(leak_strength),
                             leak_count=int(leak_count)))
    if target_sd:
        cfg["synth"]["target_effect_sd"] = float(target_sd)
    return cfg


def replicate(job):
    t0 = time.time()
    arm, level, li, rep = job["arm"], job["level"], job["level_index"], job["rep"]
    seed = job_seed(arm, li, rep)
    mech = job["mechanism"]
    cfg = make_cfg(seed,
                   leak_strength=level if mech == "score" else 0.0,
                   leak_count=level if mech == "count" else 0,
                   intercept=job["intercept"], target_sd=job["target_sd"])
    programs, evidence = synth.generate(cfg)
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    tr, te, info = splits.temporal_split(meta, TRAIN_MAX, TEST_YEARS, enforce_target_disjoint=True)
    ytr, yte = y.loc[tr].values, y.loc[te].values
    clusters = meta.loc[te, "target_id"].values
    p = {}
    for name, X in (("naive", Xn), ("cens", Xc)):
        clf = models.fit_gbdt(X.loc[tr], ytr, seed=0)
        p[name] = models.predict_gbdt(clf, X.loc[te])
    s = leakage._clustered_auprc_samples(yte, p, clusters, n_boot=job["n_boot"], seed=seed)
    draws = np.array(s["naive"]) - np.array(s["cens"])
    a_n, a_c = metrics.auprc(yte, p["naive"]), metrics.auprc(yte, p["cens"])
    return {"arm": arm, "mechanism": mech, "level": level, "rep": rep, "seed": seed,
            "target_sd": job["target_sd"], "n_test": int(len(te)), "n_pos": int(yte.sum()),
            "n_clusters": int(len(np.unique(clusters))), "n_train": int(len(tr)),
            "train_prev": float(ytr.mean()),
            "naive_auprc": float(a_n), "cens_auprc": float(a_c), "lap": float(a_n - a_c),
            "lap_auroc": float(metrics.auroc(yte, p["naive"]) - metrics.auroc(yte, p["cens"])),
            "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
            "n_boot_eff": int(len(draws)), "sec": round(time.time() - t0, 3)}


# ----------------------------------------------------------------------------- summaries
def _rate(x):
    x = np.asarray(x, float)
    p = float(x.mean())
    return round(p, 4), round(float(np.sqrt(p * (1 - p) / len(x))), 4)


def _mean(x):
    x = np.asarray(x, float)
    return round(float(x.mean()), 4), round(float(x.std(ddof=1) / np.sqrt(len(x))), 4)


def summarize_level(r):
    lap = np.array([x["lap"] for x in r])
    lo, hi = np.array([x["lo"] for x in r]), np.array([x["hi"] for x in r])
    width = hi - lo
    mean = float(lap.mean())
    sd = float(lap.std(ddof=1))
    out = {"replicates": len(r),
           "mean_n_test": round(float(np.mean([x["n_test"] for x in r])), 1),
           "mean_test_positives": round(float(np.mean([x["n_pos"] for x in r])), 2),
           "mean_test_prevalence": round(float(np.mean([x["n_pos"] / x["n_test"] for x in r])), 4),
           "mean_test_clusters": round(float(np.mean([x["n_clusters"] for x in r])), 1),
           "mean_n_train": round(float(np.mean([x["n_train"] for x in r])), 1),
           "mean_train_prevalence": round(float(np.mean([x["train_prev"] for x in r])), 4),
           "mean_deployable_auprc": round(float(np.mean([x["cens_auprc"] for x in r])), 4),
           "mean_naive_auprc": round(float(np.mean([x["naive_auprc"] for x in r])), 4)}
    out["mean_lap"], out["mean_lap_mcse"] = _mean(lap)
    out["sd_lap"] = round(sd, 4)
    out["lap_range_95"] = [round(float(np.quantile(lap, .025)), 4), round(float(np.quantile(lap, .975)), 4)]
    out["mean_lap_auroc"] = round(float(np.mean([x["lap_auroc"] for x in r])), 4)
    out["detection_rate"], out["detection_rate_mcse"] = _rate(lo > 0)
    out["ci_below_zero_rate"], out["ci_below_zero_rate_mcse"] = _rate(hi < 0)
    out["two_sided_exclusion_rate"], out["two_sided_exclusion_rate_mcse"] = _rate((lo > 0) | (hi < 0))
    out["coverage_of_mean_lap"], out["coverage_of_mean_lap_mcse"] = _rate((lo <= mean) & (mean <= hi))
    out["mean_ci_width"], out["mean_ci_width_mcse"] = _mean(width)
    # Calibration diagnostic: bootstrap CI width vs. the width implied by the Monte Carlo SD.
    out["ci_width_over_mc_width"] = round(float(width.mean() / (2 * 1.959964 * sd)), 3) if sd > 0 else None
    out["min_bootstrap_draws"] = int(min(x["n_boot_eff"] for x in r))
    return out


def _iso(y):
    """Pool-adjacent-violators: non-decreasing fit to y (equal weights)."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, w in blocks:
        out += [v] * w
    return np.array(out)


def _threshold(xs, ys, target):
    """Smallest x at which the isotonic, linearly interpolated curve reaches ``target``."""
    order = np.argsort(xs)
    x, yv = np.asarray(xs)[order], _iso(np.asarray(ys)[order])
    if yv[-1] < target:
        return None
    if yv[0] >= target:
        return float(x[0])
    j = int(np.argmax(yv >= target))
    x0, x1, y0, y1 = x[j - 1], x[j], yv[j - 1], yv[j]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)


def _interp_detect(xs, ys, x_at):
    order = np.argsort(xs)
    x, yv = np.asarray(xs)[order], _iso(np.asarray(ys)[order])
    return float(np.interp(x_at, x, yv))


def detection_thresholds(level_rows, targets=(0.5, 0.8), lap_at=0.0743, n_resample=2000, seed=0):
    """level_rows: list of per-level replicate lists (null first). Returns thresholds in mean-LAP
    units with MC SEs from resampling replicates within each level."""
    def stats(rows_by_level):
        xs = [np.mean([x["lap"] for x in r]) for r in rows_by_level]
        ys = [np.mean([x["lo"] > 0 for x in r]) for r in rows_by_level]
        return xs, ys
    xs, ys = stats(level_rows)
    point = {f"{int(t * 100)}pct": _threshold(xs, ys, t) for t in targets}
    point["detection_at_observed_lap"] = _interp_detect(xs, ys, lap_at)
    rng = np.random.default_rng(seed)
    arrs = [(np.array([x["lap"] for x in r]), np.array([x["lo"] > 0 for x in r], float)) for r in level_rows]
    boot = {k: [] for k in point}
    for _ in range(n_resample):
        bx, by = [], []
        for lap, det in arrs:
            idx = rng.integers(0, len(lap), len(lap))
            bx.append(lap[idx].mean())
            by.append(det[idx].mean())
        for t in targets:
            v = _threshold(bx, by, t)
            boot[f"{int(t * 100)}pct"].append(np.nan if v is None else v)
        boot["detection_at_observed_lap"].append(_interp_detect(bx, by, lap_at))
    out = {}
    for k, v in point.items():
        b = np.array(boot[k], float)
        ok = b[np.isfinite(b)]
        out[k] = {"value": None if v is None else round(v, 4),
                  "mcse": round(float(ok.std(ddof=1)), 4) if len(ok) > 1 else None,
                  "mc_interval_95": ([round(float(np.quantile(ok, .025)), 4), round(float(np.quantile(ok, .975)), 4)]
                                     if len(ok) > 1 else None),
                  "resamples_reaching_target": int(len(ok))}
    out["curve_points"] = [{"mean_lap": round(float(a), 4), "detection_rate": round(float(b), 4)}
                           for a, b in sorted(zip(xs, ys))]
    return out


# ----------------------------------------------------------------------------- driver
def build_jobs(args):
    jobs = []

    def add(arm, mech, levels, reps, intercept, target_sd):
        for li, lv in enumerate(levels):
            for rep in range(reps):
                jobs.append({"arm": arm, "mechanism": mech, "level": lv, "level_index": li,
                             "rep": rep, "intercept": intercept, "target_sd": target_sd,
                             "n_boot": args.n_boot})
    arms = set(args.arms.split(","))
    if "null" in arms:
        add("null", "none", [0], args.null_reps, INTERCEPT, 0.0)
    if "score" in arms:
        add("score", "score", args.score_levels, args.power_reps, INTERCEPT, 0.0)
    if "count" in arms:
        add("count", "count", args.count_levels, args.power_reps, INTERCEPT, 0.0)
    if "re_null" in arms:
        add("re_null", "none", [0], args.re_null_reps, INTERCEPT_RE, RE_SD)
    if "re_score" in arms:
        add("re_score", "score", args.re_levels, args.re_power_reps, INTERCEPT_RE, RE_SD)
    return jobs


def key(d):
    return (d["arm"], float(d["level"]), int(d["rep"]))


def run_jobs(jobs, workers, checkpoint):
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done[key(row)] = row
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs; {len(jobs) - len(todo)} already in checkpoint; running {len(todo)} "
          f"with {workers} workers", flush=True)
    t0 = time.time()
    fh = open(checkpoint, "a") if checkpoint else None
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(replicate, j) for j in todo]
            for i, f in enumerate(cf.as_completed(futs), 1):
                row = f.result()
                done[key(row)] = row
                if fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                if i % 100 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} done, {el / 60:.1f} min elapsed, "
                          f"ETA {el / i * (len(todo) - i) / 60:.1f} min", flush=True)
    finally:
        if fh:
            fh.close()
    wanted = {key(j) for j in jobs}
    return [r for k, r in done.items() if k in wanted], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--null-reps", type=int, default=3000)
    ap.add_argument("--power-reps", type=int, default=500)
    ap.add_argument("--re-null-reps", type=int, default=2000)
    ap.add_argument("--re-power-reps", type=int, default=400)
    ap.add_argument("--score-levels", type=float, nargs="+", default=SCORE_LEVELS)
    ap.add_argument("--count-levels", type=int, nargs="+", default=COUNT_LEVELS)
    ap.add_argument("--re-levels", type=float, nargs="+", default=RE_SCORE_LEVELS)
    ap.add_argument("--arms", default="null,score,count,re_null,re_score")
    ap.add_argument("--checkpoint", default=None, help="JSONL of finished replicates (resumable)")
    ap.add_argument("--pilot", action="store_true", help="few replicates per level; print only")
    ap.add_argument("--out", default="outputs/instrument_validation_matched.json")
    args = ap.parse_args()
    if args.pilot:
        args.null_reps = args.power_reps = args.re_null_reps = args.re_power_reps = min(args.power_reps, 12)

    jobs = build_jobs(args)
    rows, wall = run_jobs(jobs, args.workers, args.checkpoint)
    print(f"wall time {wall / 60:.1f} min; mean replicate {np.mean([r['sec'] for r in rows]):.2f} s "
          f"(single thread)", flush=True)

    def rows_of(arm, level=None):
        return [r for r in rows if r["arm"] == arm and (level is None or r["level"] == level)]

    results = {}
    for arm, levels, name in (("null", [0], "leak"), ("score", args.score_levels, "leak_strength"),
                              ("count", args.count_levels, "leak_count"), ("re_null", [0], "leak"),
                              ("re_score", args.re_levels, "leak_strength")):
        lv = [dict({name: l}, **summarize_level(rows_of(arm, l))) for l in levels if rows_of(arm, l)]
        if lv:
            results[arm] = lv
    for arm, lv in results.items():
        for s in lv:
            lab = [k for k in s if k.startswith("leak")][0]
            print(f"{arm:>8} {lab}={s[lab]:<4}: n_te {s['mean_n_test']:.0f} pos {s['mean_test_positives']:.1f} "
                  f"prev {s['mean_test_prevalence']:.3f} clus {s['mean_test_clusters']:.0f} | "
                  f"LAP {s['mean_lap']:+.4f}±{s['mean_lap_mcse']:.4f} (sd {s['sd_lap']:.3f}) "
                  f"detect {s['detection_rate']:.3f}±{s['detection_rate_mcse']:.3f} "
                  f"CI<0 {s['ci_below_zero_rate']:.3f}±{s['ci_below_zero_rate_mcse']:.3f} "
                  f"cov {s['coverage_of_mean_lap']:.3f} width {s['mean_ci_width']:.3f} "
                  f"(w/mc {s['ci_width_over_mc_width']}) dep {s['mean_deployable_auprc']:.3f}", flush=True)
    if args.pilot:
        return

    thresholds = {}
    null_rows = rows_of("null")
    if null_rows:
        for arm, levels in (("score", args.score_levels), ("count", args.count_levels)):
            per = [null_rows] + [rows_of(arm, l) for l in levels if rows_of(arm, l)]
            if len(per) > 1:
                thresholds[arm] = detection_thresholds(per, lap_at=REAL["observed_LAP_auprc"])
        if len(thresholds) == 2:
            pooled = [null_rows] + [rows_of(a, l) for a, ls in (("score", args.score_levels),
                                                                 ("count", args.count_levels))
                                    for l in ls if rows_of(a, l)]
            thresholds["pooled_score_and_count"] = detection_thresholds(pooled, lap_at=REAL["observed_LAP_auprc"])
    re_null_rows = rows_of("re_null")
    re_per = [re_null_rows] + [rows_of("re_score", l) for l in args.re_levels if rows_of("re_score", l)]
    if re_null_rows and len(re_per) > 1:
        thresholds["re_score"] = detection_thresholds(re_per, lap_at=REAL["observed_LAP_auprc"])
    for k, v in thresholds.items():
        print(f"threshold [{k}]: 50% at LAP {v['50pct']['value']} (MCSE {v['50pct']['mcse']}), "
              f"80% at {v['80pct']['value']} (MCSE {v['80pct']['mcse']}); detection at observed "
              f"LAP {REAL['observed_LAP_auprc']}: {v['detection_at_observed_lap']['value']}", flush=True)

    out = {
        "real_benchmark": REAL,
        "design": {
            "generator": "temporal_leakage_audit/data/synth.py (existing arguments; target_effect_sd "
                         "only in the re_* arms, default 0 reproduces the original generator exactly)",
            "synth": dict(MATCHED, base_intercept=INTERCEPT, base_intercept_re_arms=INTERCEPT_RE,
                          target_effect_sd_re_arms=RE_SD),
            "split": f"train <= {TRAIN_MAX}, test {TEST_YEARS[0]}-{TEST_YEARS[1]}, target-disjoint",
            "model": "models.fit_gbdt (HistGradientBoosting, seed 0), naive vs as-of-time censored (h=0)",
            "bootstrap": f"target-clustered paired percentile (leakage._clustered_auprc_samples), "
                         f"{args.n_boot} resamples, 95% CI",
            "score_levels": args.score_levels, "count_levels": args.count_levels,
            "re_score_levels": args.re_levels,
            "replicates": {"null": args.null_reps, "power_per_level": args.power_reps,
                           "re_null": args.re_null_reps, "re_power": args.re_power_reps},
            "seeds": "generator seed = arm base (3e6..7e6) + 50000*level_index + 10*rep; bootstrap seed = generator seed",
            "mcse": "rates: sqrt(p(1-p)/R); means: SD/sqrt(R); thresholds: SD over 2000 resamples of "
                    "replicates within level",
            "detection": "95% CI lower limit > 0; ci_below_zero_rate: upper limit < 0; coverage: CI "
                         "contains the Monte Carlo mean LAP of that level",
            "thresholds": "mean LAP at which the isotonic, linearly interpolated detection-vs-mean-LAP "
                          "curve (null level included) reaches 50% / 80%",
            "runtime_minutes": round(wall / 60, 1), "workers": args.workers,
            "mean_replicate_seconds_single_thread": round(float(np.mean([r["sec"] for r in rows])), 2),
        },
        "results": results,
        "detection_thresholds": thresholds,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
