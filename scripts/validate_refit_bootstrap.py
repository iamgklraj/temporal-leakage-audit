"""Does a refit bootstrap restore nominal coverage of the LAP confidence interval?

The instrument's LAP CI (scripts/validate_instrument.py) is a target-clustered, paired
percentile bootstrap of the TEST block with the two fitted learners held fixed. It therefore
ignores training-set variability, and its coverage of the Monte Carlo mean LAP falls below
95% at some leakage levels. This script compares, on the SAME synthetic replicates as
validate_instrument.py (score mechanism: ``leak_strength`` = lambda, ``leak_count`` = 0; same
seeds, split, learner and bootstrap seed):

  (a) fixed   : the existing fixed-model test-set bootstrap (200 target-clustered paired
                resamples of the test block; percentile 95% CI). Reproduces the existing method.
  (b) refit   : for each resample, draw target clusters with replacement separately within the
                training block and within the test block, refit BOTH learners (h=0, deployable;
                h=inf, naive) on the resampled training rows, evaluate both on the resampled test
                rows and take the paired difference LAP*_b. Percentile 95% CI of {LAP*_b}.
                Secondary intervals from the same draws: basic (2*LAP - q97.5, 2*LAP - q2.5) and
                normal (LAP +/- 1.96 sd*).
  (c) inflate : variance inflation. Each refit pair from (b) is also evaluated on the FULL
                original test block, giving the training-resample variance of LAP with the test
                set fixed (no extra fitting). CI = LAP +/- 1.96 sqrt(var_fixed_test + var_train),
                where var_fixed_test is the variance of the (a) draws. (Seed variance is zero:
                the learner has no subsampling or early stopping at this n, so random_state has
                no effect.)
  (d) half    : variance inflation with a half-sampling estimate of the training variance: both
                learners refit on half of the training clusters drawn WITHOUT replacement
                (``--n-half`` times), evaluated on the full test block; var_train =
                var(LAP_half) * m/(K-m); CI = LAP +/- 1.96 sqrt(var_fixed_test + var_train).
                Unlike (b)/(c) no training row is duplicated.

For each level we report, per method, coverage of the mean LAP (the existing definition: mean
over the same replicates; also against a high-precision mean over ``--point-reps`` replicates),
detection rate (CI above 0; the false-positive rate at lambda = 0), the rate of CIs below 0,
mean CI width and mean bootstrap SD, all with Monte Carlo standard errors. The (a) summary on
replicates 0..99 is compared field-by-field with outputs/instrument_validation.json.

Usage:  python scripts/validate_refit_bootstrap.py [--reps 120] [--n-refit 200] [--n-half 100]
                                                  [--point-reps 500] [--workers 6]
Writes outputs/instrument_validation_refit.json. Per-replicate results are appended to a
JSONL checkpoint as they finish (``--checkpoint``), so an interrupted run can be resumed and
``--summarize-only`` rebuilds the JSON from the checkpoint.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import validate_instrument as vi  # noqa: E402  (reuse its config, seeds, replicate())
from temporal_leakage_audit import features, leakage, metrics, models, splits  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402

LEVELS = [0.0, 0.5, 1.5]          # subset of vi.LEVELS (score mechanism)
Z = 1.959963984540054
METHODS = ("a_fixed", "b_refit", "b_refit_basic", "b_refit_normal", "c_inflate", "d_halfsample")
REFIT_SEED = 20_260_928


def _data(level, rep):
    """Exactly the data, split and seeds of validate_instrument.replicate(("score", level, rep))."""
    cfg = vi._cfg(level, 0, seed=10_000 + 1_000 * vi.LEVELS.index(level) + rep)
    programs, evidence = synth.generate(cfg)
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    tr, te, info = splits.temporal_split(meta, 2016, (2017, 2020), enforce_target_disjoint=True)
    return Xc, Xn, y, meta, tr, te


def _cluster_rows(clusters):
    uniq = np.unique(clusters)
    return uniq, {c: np.where(clusters == c)[0] for c in uniq}


def _resample(rng, uniq, cl_to_rows):
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([cl_to_rows[c] for c in chosen])


def _q(x, a):
    return float(np.quantile(x, a))


def run_replicate(args):
    level, rep, n_refit = args[:3]
    n_half = args[3] if len(args) > 3 else 0
    t0 = time.time()
    Xc, Xn, y, meta, tr, te = _data(level, rep)
    ytr, yte = y.loc[tr].values, y.loc[te].values
    X_tr = {"naive": Xn.loc[tr], "cens": Xc.loc[tr]}
    X_te = {"naive": Xn.loc[te], "cens": Xc.loc[te]}
    c_te = meta.loc[te, "target_id"].values

    # (a) fixed-model test-set bootstrap -- identical to validate_instrument.replicate().
    p = {}
    for name in ("naive", "cens"):
        clf = models.fit_gbdt(X_tr[name], ytr, seed=0)
        p[name] = models.predict_gbdt(clf, X_te[name])
    s = leakage._clustered_auprc_samples(yte, p, c_te, n_boot=vi.N_BOOT, seed=rep)
    a_draws = np.array(s["naive"]) - np.array(s["cens"])
    lap = float(metrics.auprc(yte, p["naive"]) - metrics.auprc(yte, p["cens"]))
    row = {"level": level, "rep": rep, "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_train_clusters": int(meta.loc[tr, "target_id"].nunique()),
           "n_test_clusters": int(len(np.unique(c_te))),
           "lap": lap,
           "lap_auroc": float(metrics.auroc(yte, p["naive"]) - metrics.auroc(yte, p["cens"])),
           "a_fixed": {"lo": _q(a_draws, .025), "hi": _q(a_draws, .975),
                       "sd": float(np.std(a_draws, ddof=1)), "n_draws": int(len(a_draws))}}
    u_tr, rows_tr_of = _cluster_rows(meta.loc[tr, "target_id"].values)
    if n_refit > 0:
        row.update(_refit_bootstrap(level, rep, n_refit, lap, row["a_fixed"]["sd"], X_tr, X_te,
                                    ytr, yte, u_tr, rows_tr_of, c_te))
    if n_half > 0:
        row.update(_half_sample(level, rep, n_half, lap, row["a_fixed"]["sd"], X_tr, X_te,
                                ytr, yte, u_tr, rows_tr_of))
    row["seconds"] = round(time.time() - t0, 2)
    return row


def _refit_bootstrap(level, rep, n_refit, lap, a_sd, X_tr, X_te, ytr, yte, u_tr, rows_tr_of, c_te):
    """(b) refit bootstrap: resample train and test clusters independently, refit both learners.
    Also returns (c), which uses the same refits evaluated on the full original test block."""
    t0 = time.time()
    rng = np.random.default_rng([REFIT_SEED, vi.LEVELS.index(level), rep])
    u_te, rows_te_of = _cluster_rows(c_te)
    two_way, train_only, skipped = [], [], 0
    for _ in range(n_refit):
        r_tr = _resample(rng, u_tr, rows_tr_of)
        r_te = _resample(rng, u_te, rows_te_of)
        pb = {}
        for name in ("naive", "cens"):
            clf = models.fit_gbdt(X_tr[name].iloc[r_tr], ytr[r_tr], seed=0)
            pb[name] = models.predict_gbdt(clf, X_te[name])   # predictions on the full test block
        # train-resample-only LAP (full original test block) -> used by (c)
        train_only.append(metrics.auprc(yte, pb["naive"]) - metrics.auprc(yte, pb["cens"]))
        yb = yte[r_te]
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        two_way.append(metrics.auprc(yb, pb["naive"][r_te]) - metrics.auprc(yb, pb["cens"][r_te]))
    two_way, train_only = np.array(two_way), np.array(train_only)
    b_lo, b_hi, b_sd = _q(two_way, .025), _q(two_way, .975), float(np.std(two_way, ddof=1))
    var_train = float(np.var(train_only, ddof=1))
    c_sd = float(np.sqrt(a_sd ** 2 + var_train))
    return {
        "b_refit": {"lo": b_lo, "hi": b_hi, "sd": b_sd, "mean": float(two_way.mean()),
                    "n_draws": int(len(two_way)), "skipped_one_class": skipped,
                    "seconds": round(time.time() - t0, 2)},
        "b_refit_basic": {"lo": 2 * lap - b_hi, "hi": 2 * lap - b_lo, "sd": b_sd},
        "b_refit_normal": {"lo": lap - Z * b_sd, "hi": lap + Z * b_sd, "sd": b_sd},
        "train_only": {"sd": float(np.sqrt(var_train)), "mean": float(train_only.mean()),
                       "n_draws": int(len(train_only))},
        "c_inflate": {"lo": lap - Z * c_sd, "hi": lap + Z * c_sd, "sd": c_sd},
        "n_refit": int(n_refit),
    }


def _half_sample(level, rep, n_half, lap, a_sd, X_tr, X_te, ytr, yte, u_tr, rows_tr_of):
    """(d) variance inflation with a half-sampling estimate of the training variance: refit both
    learners on m = K//2 training clusters drawn WITHOUT replacement, evaluate on the full test
    block; var_train = var(LAP_half) * m / (K - m) (= 1 for m = K/2; the delete-d jackknife /
    half-sampling scaling). CI = LAP +/- 1.96 sqrt(var(a draws) + var_train). Unlike (b)/(c),
    no training row is duplicated."""
    t0 = time.time()
    rng = np.random.default_rng([REFIT_SEED + 1, vi.LEVELS.index(level), rep])
    K = len(u_tr)
    m = K // 2
    half = []
    for _ in range(n_half):
        chosen = rng.choice(u_tr, size=m, replace=False)
        r_tr = np.concatenate([rows_tr_of[c] for c in chosen])
        ph = {}
        for name in ("naive", "cens"):
            clf = models.fit_gbdt(X_tr[name].iloc[r_tr], ytr[r_tr], seed=0)
            ph[name] = models.predict_gbdt(clf, X_te[name])
        half.append(metrics.auprc(yte, ph["naive"]) - metrics.auprc(yte, ph["cens"]))
    half = np.array(half)
    var_half = float(np.var(half, ddof=1)) * m / (K - m)
    d_sd = float(np.sqrt(a_sd ** 2 + var_half))
    return {
        "train_half": {"sd_raw": float(np.std(half, ddof=1)), "sd_scaled": float(np.sqrt(var_half)),
                       "mean": float(half.mean()), "n_draws": int(len(half)), "m_clusters": int(m),
                       "seconds": round(time.time() - t0, 2)},
        "d_halfsample": {"lo": lap - Z * d_sd, "hi": lap + Z * d_sd, "sd": d_sd},
        "n_half": int(n_half),
    }


# ----------------------------------------------------------------------------- summaries
def _rate(mask):
    m = np.asarray(mask, dtype=float)
    p = float(m.mean())
    return round(p, 3), round(float(np.sqrt(p * (1 - p) / len(m))), 3)


def _method_summary(rows, method, target_same, target_precise):
    lo = np.array([r[method]["lo"] for r in rows])
    hi = np.array([r[method]["hi"] for r in rows])
    sd = np.array([r[method]["sd"] for r in rows])
    cov, cov_se = _rate((lo <= target_same) & (target_same <= hi))
    covp, covp_se = _rate((lo <= target_precise) & (target_precise <= hi))
    det, det_se = _rate(lo > 0)
    below, below_se = _rate(hi < 0)
    w = hi - lo
    return {"coverage_of_mean_lap": cov, "coverage_mcse": cov_se,
            "coverage_of_precise_mean_lap": covp, "coverage_precise_mcse": covp_se,
            "detection_rate": det, "detection_mcse": det_se,
            "ci_below_zero_rate": below, "ci_below_zero_mcse": below_se,
            "mean_ci_width": round(float(w.mean()), 4),
            "mean_ci_width_mcse": round(float(w.std(ddof=1) / np.sqrt(len(w))), 4),
            "mean_boot_sd": round(float(sd.mean()), 4)}


def summarize(rows, reps, levels):
    out = []
    for level in levels:
        all_rows = sorted([r for r in rows if r["level"] == level], key=lambda r: r["rep"])
        refit = [r for r in all_rows if "b_refit" in r and r["rep"] < reps]
        lap_all = np.array([r["lap"] for r in all_rows])
        precise_mean = float(lap_all.mean())
        entry = {
            "leak_strength": level,
            "point_replicates": len(all_rows),
            "precise_mean_lap": round(precise_mean, 4),
            "precise_mean_lap_mcse": round(float(lap_all.std(ddof=1) / np.sqrt(len(lap_all))), 4),
            "empirical_sd_lap": round(float(lap_all.std(ddof=1)), 4),
            # (a) on every point replicate (more replicates -> tighter MC error for (a))
            "a_fixed_all_point_replicates": _method_summary(all_rows, "a_fixed", precise_mean,
                                                            precise_mean),
        }
        if refit:
            lap_r = np.array([r["lap"] for r in refit])
            same_mean = float(lap_r.mean())
            entry.update({
                "refit_replicates": len(refit),
                "refit_rep_ids": [int(refit[0]["rep"]), int(refit[-1]["rep"])],
                "mean_lap": round(same_mean, 4),
                "empirical_sd_lap_refit_reps": round(float(lap_r.std(ddof=1)), 4),
                "mean_n_test": round(float(np.mean([r["n_test"] for r in refit])), 1),
                "mean_n_train": round(float(np.mean([r["n_train"] for r in refit])), 1),
                "refit_center_bias": round(float(np.mean([r["b_refit"]["mean"] - r["lap"]
                                                          for r in refit])), 4),
                "train_only_sd_mean": round(float(np.mean([r["train_only"]["sd"] for r in refit])), 4),
                # training-induced SD implied by the simulation: sqrt(emp. var - mean fixed-boot var)
                "implied_train_sd": round(float(np.sqrt(max(lap_all.var(ddof=1) - np.mean(
                    [r["a_fixed"]["sd"] ** 2 for r in all_rows]), 0.0))), 4),
                "methods": {m: _method_summary(refit, m, same_mean, precise_mean) for m in METHODS
                            if all(m in r for r in refit)},
                "mean_seconds_per_refit_replicate": round(float(np.mean(
                    [r["b_refit"].get("seconds", r["seconds"]) for r in refit])), 1),
            })
            if all("train_half" in r for r in refit):
                entry["train_half_sd_scaled_mean"] = round(float(np.mean(
                    [r["train_half"]["sd_scaled"] for r in refit])), 4)
        out.append(entry)
    return out


def reproduction_check(rows, levels, ref_path):
    """Summarise (a) on replicates 0..99 with validate_instrument.summarize and compare."""
    if not os.path.exists(ref_path):
        return {"reference": ref_path, "available": False}
    ref = {r["leak_strength"]: r for r in json.load(open(ref_path))["levels"]}
    vi_rows = [{"mechanism": "score", "level": r["level"], "rep": r["rep"], "n_test": r["n_test"],
                "lap": r["lap"], "lap_auroc": r["lap_auroc"],
                "lo": r["a_fixed"]["lo"], "hi": r["a_fixed"]["hi"]}
               for r in rows if r["rep"] < 100]
    levels = [lv for lv in levels if any(r["level"] == lv for r in vi_rows)]
    ours = {s["leak_strength"]: s for s in vi.summarize(vi_rows, "score", levels)}
    out = {"reference": ref_path, "available": True, "levels": []}
    keys = ("replicates", "mean_lap", "lap_range_95", "mean_lap_auroc", "detection_rate",
            "ci_below_zero_rate", "coverage_of_mean_lap", "mean_n_test")
    for lv in levels:
        if lv not in ref or ours[lv]["replicates"] == 0:
            continue
        diffs = {k: {"reproduced": ours[lv][k], "reference": ref[lv][k]} for k in keys}
        out["levels"].append({"leak_strength": lv,
                              "exact_match": all(ours[lv][k] == ref[lv][k] for k in keys),
                              "fields": diffs})
    return out


# ----------------------------------------------------------------------------- main
def _load_checkpoint(path):
    rows = []
    if path and os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=120, help="replicates per level with the refit bootstrap")
    ap.add_argument("--n-refit", type=int, default=200, help="refit resamples per replicate")
    ap.add_argument("--n-half", type=int, default=100,
                    help="half-sample refits per replicate for (d); 0 disables")
    ap.add_argument("--point-reps", type=int, default=500,
                    help="replicates per level for (a) and the precise mean LAP (>= --reps)")
    ap.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint", default=None, help="JSONL of per-replicate rows (resumable)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--ref", default="outputs/instrument_validation.json")
    ap.add_argument("--out", default="outputs/instrument_validation_refit.json")
    args = ap.parse_args()
    levels = [float(x) for x in args.levels]
    assert all(lv in vi.LEVELS for lv in levels), "levels must be in validate_instrument.LEVELS"
    point_reps = max(args.point_reps, args.reps)
    assert point_reps <= 1000, "seeds of different levels would collide beyond 1000 replicates"

    rows = _load_checkpoint(args.checkpoint)
    t_start = time.time()
    if not args.summarize_only:
        # Self-check: our (a) is bit-identical to validate_instrument.replicate().
        for lv in levels[:2]:
            ref_row, ours = vi.replicate(("score", lv, 0)), run_replicate((lv, 0, 0))
            assert all(ref_row[k] == ours[k] for k in ("lap", "lap_auroc", "n_test")), (ref_row, ours)
            assert ref_row["lo"] == ours["a_fixed"]["lo"] and ref_row["hi"] == ours["a_fixed"]["hi"]
        have = {(r["level"], r["rep"]) for r in rows}
        have_b = {(r["level"], r["rep"]) for r in rows if "b_refit" in r}
        have_d = {(r["level"], r["rep"]) for r in rows if "d_halfsample" in r}
        # cheap point-only replicates first, then refit replicates interleaved across levels
        # (so an interrupted run still has balanced levels for --summarize-only)
        jobs = [(lv, rep, 0, 0) for rep in range(args.reps, point_reps) for lv in levels
                if (lv, rep) not in have]
        for rep in range(args.reps):
            for lv in levels:
                nb = args.n_refit if (lv, rep) not in have_b else 0
                nd = args.n_half if (lv, rep) not in have_d else 0
                if nb or nd:
                    jobs.append((lv, rep, nb, nd))
        print(f"{len(rows)} rows from checkpoint; {len(jobs)} jobs to run on {args.workers} workers",
              flush=True)
        ck = open(args.checkpoint, "a") if args.checkpoint else None
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_replicate, j) for j in jobs]
            for i, f in enumerate(cf.as_completed(futs), 1):
                r = f.result()
                rows.append(r)
                if ck:
                    ck.write(json.dumps(r) + "\n")
                    ck.flush()
                if "b_refit" in r or "d_halfsample" in r or i % 150 == 0:
                    print(f"[{time.time() - t_start:7.0f}s] {i}/{len(jobs)} level={r['level']} "
                          f"rep={r['rep']} refit={'b_refit' in r} half={'d_halfsample' in r} "
                          f"({r['seconds']}s)", flush=True)
        if ck:
            ck.close()
    # one row per (level, rep): merge rows from separate runs (the deterministic fields -- data,
    # LAP, (a) -- are identical; each run adds its method keys; the first run's timing is kept)
    merged = {}
    for r in rows:
        tgt = merged.setdefault((r["level"], r["rep"]), {})
        for key, val in r.items():
            tgt.setdefault(key, val)
    rows = [merged[k] for k in sorted(merged) if k[0] in levels and k[1] < point_reps]

    summary = summarize(rows, args.reps, levels)
    out = {
        "design": {
            "mechanism": "score (leak_strength = lambda, leak_count = 0), as validate_instrument.py",
            "levels_leak_strength": levels,
            "refit_replicates_per_level": args.reps,
            "point_replicates_per_level": point_reps,
            "n_programs": 2000,
            "split": "train <= 2016, test 2017-2020, target-disjoint",
            "seeds": "data seed 10000 + 1000*index(lambda in validate_instrument.LEVELS) + rep "
                     "(identical replicates to validate_instrument.py); (a) bootstrap seed = rep",
            "a_fixed": f"target-clustered paired percentile bootstrap of the test block, models "
                       f"fixed, {vi.N_BOOT} resamples (existing method)",
            "b_refit": f"{args.n_refit} resamples: target clusters drawn with replacement "
                       f"separately in train and test blocks; both learners (h=0, h=inf) refit on "
                       f"the resampled training rows; paired LAP on the resampled test rows; "
                       f"percentile 95% CI (b_refit), basic (b_refit_basic) and normal "
                       f"LAP +/- 1.96 sd (b_refit_normal) from the same draws",
            "d_halfsample": f"LAP +/- 1.96 sqrt(var(a draws) + var_train), var_train from "
                            f"{args.n_half} refits on half of the training clusters drawn without "
                            f"replacement (evaluated on the full test block), scaled by m/(K-m)",
            "c_inflate": "LAP +/- 1.96 sqrt(var(a draws) + var(train-resample-only LAP)), the "
                         "latter from the (b) refits evaluated on the full original test block",
            "coverage": "coverage_of_mean_lap: CI contains the mean LAP over the same (refit) "
                        "replicates (definition of validate_instrument.py); "
                        "coverage_of_precise_mean_lap: CI contains the mean over all point "
                        "replicates of that level",
            "mcse": "binomial sqrt(p(1-p)/R) for rates; sd/sqrt(R) for mean width",
            "learner": "HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, "
                       "max_iter=300, l2=1.0), seed 0; OMP_NUM_THREADS=1 per worker",
            "workers": args.workers,
        },
        "levels": summary,
        "reproduction_of_existing_fixed_bootstrap": reproduction_check(rows, levels, args.ref),
        "wall_seconds_this_run": round(time.time() - t_start, 1),
        # approximate: point-only rows + refit bootstraps + half-sample refits
        "worker_seconds_total": round(float(
            sum(r["seconds"] for r in rows if "b_refit" not in r)
            + sum(r["b_refit"].get("seconds", r["seconds"]) for r in rows if "b_refit" in r)
            + sum(r["train_half"]["seconds"] for r in rows if "train_half" in r)), 1),
        "replicates": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    for s in summary:
        print(f"\nlambda={s['leak_strength']}: precise mean LAP {s['precise_mean_lap']:+.4f} "
              f"(n={s['point_replicates']}, sd {s['empirical_sd_lap']:.4f})")
        a = s["a_fixed_all_point_replicates"]
        print(f"  a_fixed (all {s['point_replicates']} reps): cov {a['coverage_of_precise_mean_lap']:.3f}"
              f" +/- {a['coverage_precise_mcse']:.3f}  detect {a['detection_rate']:.3f}  "
              f"width {a['mean_ci_width']:.4f}")
        if "methods" in s:
            print(f"  refit reps {s['refit_replicates']}: mean LAP {s['mean_lap']:+.4f}, "
                  f"refit center bias {s['refit_center_bias']:+.4f}")
            for m, v in s["methods"].items():
                print(f"  {m:15s} cov {v['coverage_of_mean_lap']:.3f} +/- {v['coverage_mcse']:.3f}"
                      f"  covP {v['coverage_of_precise_mean_lap']:.3f}  detect "
                      f"{v['detection_rate']:.3f} +/- {v['detection_mcse']:.3f}  "
                      f"width {v['mean_ci_width']:.4f}  sd {v['mean_boot_sd']:.4f}")
    rc = out["reproduction_of_existing_fixed_bootstrap"]
    for lv in rc.get("levels", []):
        print(f"reproduction lambda={lv['leak_strength']}: exact_match={lv['exact_match']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
