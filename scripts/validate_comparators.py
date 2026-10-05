"""Validate LAP and the comparator checks against a KNOWN true LAP on synthetic data.

Reviewer: "On real data, 'mis-sized' treats LAP as the truth. Rerun every check in the synthetic
generator where the true LAP is known." The design below was fixed before the run; the script was
run once.

Design
------
Levels: score mechanism leak_strength in {0, 1.0, 2.0} (leak_count 0); count mechanism leak_count
in {0, 1, 2} (leak_strength 0); 40 replicates per level. Replicate r of a level is generated and
split exactly as replicate r of scripts/validate_instrument.py (its _cfg and seeds: 10_000 +
1_000 * LEVELS.index(level) + r for the score mechanism, 50_000 + 1_000 * COUNT_LEVELS.index(level)
+ r for the count mechanism; 2,000 programs on 1,500 targets; train info_time <= 2016, test
2017-2020, target-disjoint).

Representations (features.build): deployable X(0) and aggregate naive X(inf); and a separated naive
X_sep = [X(0), X_post], where X_post holds the same 14 evidence aggregates
(features._aggregate_evidence) computed from post-decision evidence only (evidence_date >
info_time), so X(0) + X_post = X(inf) column by column. In this generator only literature is ever
post-dated, so only the two post-literature columns of X_post vary. Each representation gets
models.fit_gbdt (seed 0) on the training block.

Ground truth: the learner-specific population LAP of the fitted models, AP_pop(naive model) -
AP_pop(deployable model), for X_sep and X(inf), on a large independent sample from the replicate's
own generative process restricted like its test block. The generator draws programs one after
another from a stream seeded by the replicate seed and draws per-area outcome effects from
seed + 1, so every seed has its own outcome model. The population is therefore made by rerunning
synth.generate with the replicate's configuration but 2,000 + 66,000 programs (n_targets
unchanged): the first 2,000 programs and their evidence are the replicate's data (asserted for
every replicate) and the next 66,000 are fresh programs from the identical process, with the
replicate's area effects. Population = the fresh programs with info_time 2017-2020 (~20,300).
Why this is the test-block distribution: the test block is the 2017-2020 programs whose target is
absent from training; with target_effect_sd = 0 (the default) target ids are drawn independently
of every covariate, evidence item and label, so the target-disjoint filter is a random thinning
and the test block is a random sample of the 2017-2020 programs of that process. The truth's own
sampling error is estimated from even/odd half-samples of the population. Sensitivity: one
separate-seed population per level (66,000 programs, seed 990_000 + 1_000 * j), which carries
different area effects from every replicate.

Per replicate (test block of about 245 programs):
 1. LAP_hat for X_sep and X(inf) with the target-clustered paired percentile bootstrap CI
    (leakage._clustered_auprc_samples, 200 resamples, seed r, as in validate_instrument.py).
 2. compare_methods.permutation on the X_sep model over the 14 X_post columns (20 permutations,
    seed 0): joint drop, largest and sum of single-column drops; plus the population joint drop
    (3 joint permutations of X_post on the population sample).
 3. compare_methods.screen of X_sep on the training block (direction-free AUROC of each column;
    does any X_post / any X(0) column reach 0.90, 0.80); the same screen of X(inf).
 4. Drop the X_post column with the largest single drop, refit, and express its population LAP
    (residual) as a share of the true X_sep LAP.
Per level: means with Monte Carlo standard errors (sd / sqrt(R); proportions sqrt(p(1 - p) / R)).
At level 0 the detection rate (CI above 0) is the false-positive rate.

Runtime: about 1.3 s per replicate single-threaded; the 240-replicate run took 4.0 min with 2
worker processes and OMP_NUM_THREADS=1.

Usage:  OMP_NUM_THREADS=1 python scripts/validate_comparators.py [--reps 40] [--workers 2]
Writes outputs/comparator_validation.json.
"""
import argparse
import concurrent.futures as cf
import copy
import json
import os
import sys
import time
from collections import Counter

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import compare_methods as CM  # noqa: E402
import validate_instrument as VI  # noqa: E402
from temporal_leakage_audit import features, leakage, metrics, models, splits  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402

SCORE_LEVELS = [0.0, 1.0, 2.0]
COUNT_LEVELS = [0, 1, 2]
LEVEL_KEYS = [("score", lv) for lv in SCORE_LEVELS] + [("count", lv) for lv in COUNT_LEVELS]
TRAIN_MAX, TEST_YEARS = 2016, (2017, 2020)
N_POP = 66_000                 # fresh programs per population (about 20,300 in 2017-2020)
N_POP_PERM = 3                 # joint permutations of X_post on the population sample
LEVEL_POP_SEED0 = 990_000      # sensitivity: one separate-seed population per level
POST = "post_"


def _setting(mechanism, level, rep):
    """Seed and leak parameters of replicate `rep`, exactly as in validate_instrument.replicate."""
    if mechanism == "score":
        return 10_000 + 1_000 * VI.LEVELS.index(level) + rep, float(level), 0
    return 50_000 + 1_000 * VI.COUNT_LEVELS.index(level) + rep, 0.0, int(level)


def post_block(programs, evidence):
    """X_post: features.build's evidence aggregates computed from post-decision evidence only."""
    ev = evidence.merge(programs[["program_id", "info_time"]], on="program_id", how="left")
    ev = ev[ev["evidence_date"] > ev["info_time"]].drop(columns="info_time")
    return features._aggregate_evidence(ev, programs.set_index("program_id").index).add_prefix(POST)


def representations(programs, evidence):
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    return {"dep": Xc, "agg": Xn, "sep": Xc.join(post_block(programs, evidence))}, y, meta


def _test_years(programs, evidence):
    p = programs[programs["info_time"].between(*TEST_YEARS)]
    return p, evidence[evidence["program_id"].isin(p["program_id"])]


def matched_population(cfg, programs, evidence):
    """Fresh programs from the replicate's own stream (same area effects), test years only."""
    big = copy.deepcopy(cfg)
    big["synth"]["n_programs"] = len(programs) + N_POP      # n_targets unchanged: same stream
    P, E = synth.generate(big)
    if not (P.iloc[:len(programs)].reset_index(drop=True).equals(programs) and
            E[E["program_id"].isin(programs["program_id"])].reset_index(drop=True).equals(evidence)):
        raise RuntimeError("population stream does not reproduce the replicate")
    X, y, _ = representations(*_test_years(P.iloc[len(programs):], E))
    return X, y.values


_LEVEL_POP = {}


def level_population(mechanism, level):
    """Sensitivity: one separate-seed population per level (different area effects)."""
    key = (mechanism, level)
    if key not in _LEVEL_POP:
        _, ls, lc = _setting(mechanism, level, 0)
        cfg = VI._cfg(ls, lc, seed=LEVEL_POP_SEED0 + 1_000 * LEVEL_KEYS.index(key), n_programs=N_POP)
        X, y, _ = representations(*_test_years(*synth.generate(cfg)))
        _LEVEL_POP[key] = (X, y.values)
    return _LEVEL_POP[key]


def population_truth(fits, keep, X, y, cols, post_idx=None, rng=None):
    for k in ("dep", "agg"):
        assert list(X[k].columns) == list(X["dep"].columns)
    assert list(X["sep"].columns) == cols
    p = {k: models.predict_gbdt(fits[k], X[k]) for k in ("dep", "agg", "sep")}
    p["drop"] = models.predict_gbdt(fits["drop"], X["sep"], cols=keep)
    ap = {k: metrics.auprc(y, v) for k, v in p.items()}
    out = {"true_lap_sep": ap["sep"] - ap["dep"], "true_lap_agg": ap["agg"] - ap["dep"],
           "true_lap_drop": ap["drop"] - ap["dep"], "n_pop": int(len(y))}
    if post_idx is not None:
        half = np.arange(len(y)) % 2 == 0
        h = [{k: metrics.auprc(y[m], p[k][m]) for k in ("dep", "agg", "sep")} for m in (half, ~half)]
        for k in ("sep", "agg"):
            out[f"half_diff_{k}"] = (h[0][k] - h[0]["dep"]) - (h[1][k] - h[1]["dep"])
        Xs = X["sep"].values
        drops = []
        for _ in range(N_POP_PERM):
            Xp = Xs.copy()
            Xp[:, post_idx] = Xs[rng.permutation(len(Xs))][:, post_idx]
            drops.append(ap["sep"] - metrics.auprc(y, fits["sep"].predict_proba(Xp)[:, 1]))
        out["pop_joint_pi"] = float(np.mean(drops))
    return out


def replicate(job):
    mechanism, level, rep = job
    seed, ls, lc = _setting(mechanism, level, rep)
    cfg = VI._cfg(ls, lc, seed=seed)
    programs, evidence = synth.generate(cfg)
    X, y, meta = representations(programs, evidence)
    ev_cols = [c for c in X["dep"].columns if c.startswith("ev_")]
    assert np.allclose(X["dep"][ev_cols].values + X["sep"][[POST + c for c in ev_cols]].values,
                       X["agg"][ev_cols].values)                       # X(0) + X_post = X(inf)
    tr, te, _ = splits.temporal_split(meta, TRAIN_MAX, TEST_YEARS, enforce_target_disjoint=True)
    ytr, yte = y.loc[tr].values, y.loc[te].values
    cols = list(X["sep"].columns)
    post = [c for c in cols if c.startswith(POST)]
    fits = {k: models.fit_gbdt(X[k].loc[tr], ytr, seed=0) for k in ("dep", "agg", "sep")}
    pte = {k: models.predict_gbdt(fits[k], X[k].loc[te]) for k in fits}
    row = {"mechanism": mechanism, "level": level, "rep": rep, "seed": seed, "n_test": int(len(te))}

    # 1. LAP_hat with the paper's target-clustered paired percentile bootstrap CI
    s = leakage._clustered_auprc_samples(yte, pte, meta.loc[te, "target_id"].values,
                                         n_boot=VI.N_BOOT, seed=rep)
    for k in ("sep", "agg"):
        draws = np.array(s[k]) - np.array(s["dep"])
        row[f"lap_hat_{k}"] = metrics.auprc(yte, pte[k]) - metrics.auprc(yte, pte["dep"])
        row[f"lo_{k}"], row[f"hi_{k}"] = np.quantile(draws, 0.025), np.quantile(draws, 0.975)

    # 2. permutation importance of X_post in the separated naive model
    perm = CM.permutation(fits["sep"], X["sep"].loc[te].values, yte, cols, post)
    row.update(joint_pi=perm["joint_drop"], max_single_pi=perm["largest_single_drop"],
               sum_single_pi=perm["sum_of_single_drops"], top_post_column=perm["largest_single_feature"])

    # 3. univariate screen on the training block
    scr, scr_agg = CM.screen(X["sep"].loc[tr], ytr, set(post)), CM.screen(X["agg"].loc[tr], ytr)
    row.update(post_max_auroc=scr["post_decision_max_auroc"], pre_max_auroc=scr["deployable_max_auroc"],
               agg_max_auroc=scr_agg["max_auroc"])
    for t in (CM.FLAG, CM.FLAG_SENS):
        row[f"post_flag_{t}"] = scr[f"post_decision_n_flagged_ge_{t}"] > 0
        row[f"pre_flag_{t}"] = scr[f"deployable_n_flagged_ge_{t}"] > 0
        row[f"agg_flag_{t}"] = scr_agg[f"n_flagged_ge_{t}"] > 0

    # 4. drop the top-ranked X_post column and refit
    keep = [c for c in cols if c != perm["largest_single_feature"]]
    fits["drop"] = models.fit_gbdt(X["sep"].loc[tr], ytr, seed=0, cols=keep)

    # ground truth on the matched population; sensitivity on the per-level population
    Xp, yp = matched_population(cfg, programs, evidence)
    row.update(population_truth(fits, keep, Xp, yp, cols, [cols.index(c) for c in post],
                                np.random.default_rng(rep)))
    Xa, ya = level_population(mechanism, level)
    alt = population_truth(fits, keep, Xa, ya, cols)
    row.update({f"alt_{k}": alt[k] for k in ("true_lap_sep", "true_lap_agg", "true_lap_drop")})
    return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else
                bool(v) if isinstance(v, np.bool_) else v) for k, v in row.items()}


def _mean(x):
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), 4), "mcse": round(float(x.std(ddof=1) / np.sqrt(len(x))), 4)}


def _rate(b):
    p = float(np.mean(b))
    return {"rate": round(p, 3), "mcse": round(float(np.sqrt(p * (1 - p) / len(b))), 3)}


def summarize(rows, mechanism, level):
    r = [x for x in rows if x["mechanism"] == mechanism and x["level"] == level]

    def g(k):
        return np.array([x[k] for x in r], float)

    out = {"leak_strength" if mechanism == "score" else "leak_count": level, "replicates": len(r),
           "mean_n_test": round(float(g("n_test").mean()), 1),
           "mean_n_population": round(float(g("n_pop").mean()), 1)}
    for k, name in (("sep", "lap_separated"), ("agg", "lap_aggregate")):
        t, est, lo, hi = g(f"true_lap_{k}"), g(f"lap_hat_{k}"), g(f"lo_{k}"), g(f"hi_{k}")
        out[name] = {"true_lap": _mean(t),
                     "truth_population_se": round(float(np.sqrt(np.mean(g(f"half_diff_{k}") ** 2) / 4)), 4),
                     "lap_hat": _mean(est), "bias": _mean(est - t),
                     "coverage_of_true_lap": _rate((lo <= t) & (t <= hi)),
                     "detection_rate_ci_above_0": _rate(lo > 0),
                     "mean_ci_width": round(float(np.mean(hi - lo)), 4)}
    ts, jp, pjp = g("true_lap_sep"), g("joint_pi"), g("pop_joint_pi")
    out["permutation_importance"] = {
        "joint_pi": _mean(jp), "joint_pi_minus_true_lap": _mean(jp - ts),
        "share_joint_pi_gt_true_lap": _rate(jp > ts),
        "population_joint_pi": _mean(pjp), "population_joint_pi_minus_true_lap": _mean(pjp - ts),
        "max_single_pi": _mean(g("max_single_pi")), "sum_single_pi": _mean(g("sum_single_pi"))}
    scr = {}
    for t in (CM.FLAG, CM.FLAG_SENS):
        scr[f"any_post_column_ge_{t}"] = _rate(g(f"post_flag_{t}"))
        scr[f"any_x0_column_ge_{t}"] = _rate(g(f"pre_flag_{t}"))
        scr[f"any_aggregate_column_ge_{t}"] = _rate(g(f"agg_flag_{t}"))
    scr.update({f"mean_max_auroc_{k}": round(float(g(f"{k}_max_auroc").mean()), 4)
                for k in ("post", "pre", "agg")})
    out["univariate_screen"] = scr
    res, pos = g("true_lap_drop"), ts > 0
    drop = {"top_ranked_column": dict(Counter(x["top_post_column"] for x in r)),
            "residual_true_lap": _mean(res)}
    if level != 0:
        drop.update(residual_share_mean=_mean(res[pos] / ts[pos]), residual_share_n=int(pos.sum()),
                    residual_share_ratio_of_means=round(float(res.mean() / ts.mean()), 3))
    else:
        drop["residual_share"] = "undefined (true LAP is about 0 at level 0)"
    out["drop_top_ranked_post_column"] = drop
    # sensitivity: truth from one separate-seed population per level (different area effects)
    sens = {}
    for k in ("sep", "agg"):
        a, lo, hi, est = g(f"alt_true_lap_{k}"), g(f"lo_{k}"), g(f"hi_{k}"), g(f"lap_hat_{k}")
        sens[k] = {"true_lap": _mean(a), "bias": _mean(est - a), "coverage": _rate((lo <= a) & (a <= hi)),
                   "mean_abs_diff_from_matched_truth": round(float(np.mean(np.abs(a - g(f"true_lap_{k}")))), 4)}
    a = g("alt_true_lap_sep")
    sens["joint_pi_minus_true_lap"] = _mean(jp - a)
    sens["share_joint_pi_gt_true_lap"] = _rate(jp > a)
    if level != 0:
        sens["residual_share_ratio_of_means"] = round(float(g("alt_true_lap_drop").mean() / a.mean()), 3)
    out["sensitivity_single_population_per_level"] = sens
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/comparator_validation.json")
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(m, lv, r) for m, lv in LEVEL_KEYS for r in range(args.reps)]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(replicate, jobs, chunksize=2))
    # the aggregate arm must reproduce validate_instrument.py replicate for replicate
    repro = []
    for job in (("score", 1.0, 0), ("count", 2, 0)):
        v = VI.replicate(job)
        mine = next(x for x in rows if (x["mechanism"], x["level"], x["rep"]) == job)
        repro.append(all(abs(v[a] - mine[b]) < 1e-5 for a, b in
                         (("lap", "lap_hat_agg"), ("lo", "lo_agg"), ("hi", "hi_agg"))))
    out = {
        "script": "scripts/validate_comparators.py",
        "question": "Are LAP and the comparator checks right when the true LAP is known?",
        "design": {
            "levels": {"score_mechanism_leak_strength": SCORE_LEVELS, "count_mechanism_leak_count": COUNT_LEVELS},
            "replicates_per_level": args.reps,
            "data_and_split": "as scripts/validate_instrument.py (same _cfg and seeds): 2,000 programs, "
                              "1,500 targets; train info_time <= 2016, test 2017-2020, target-disjoint",
            "representations": "deployable X(0); aggregate naive X(inf); separated naive X_sep = [X(0), X_post], "
                               "X_post = the 14 evidence aggregates from evidence_date > info_time only",
            "learner": "models.fit_gbdt (HistGradientBoosting, depth 3, lr 0.06, 300 iter, l2 1.0, seed 0)",
            "truth": f"AP_pop(naive fit) - AP_pop(deployable fit) on {N_POP:,} fresh programs from the "
                     "replicate's own stream (synth.generate with n_programs = 2,000 + "
                     f"{N_POP:,}, n_targets unchanged; first 2,000 programs asserted identical to the "
                     "replicate, so area effects and leak settings match), restricted to info_time "
                     "2017-2020. target_effect_sd = 0, so target ids are independent of covariates, "
                     "evidence and labels and the target-disjoint filter is a random thinning of the "
                     "2017-2020 programs. Truth sampling error from even/odd half-samples.",
            "lap_ci": f"target-clustered paired percentile bootstrap, {VI.N_BOOT} resamples, seed = replicate",
            "permutation": f"compare_methods.permutation, {CM.N_PERM} permutations, seed {CM.SEED}, X_sep "
                           f"model, X_post columns; population joint PI from {N_POP_PERM} permutations",
            "univariate_screen": f"compare_methods.screen on the training block, flags >= {CM.FLAG} and "
                                 f">= {CM.FLAG_SENS}",
            "drop_and_refit": "drop the X_post column with the largest single-column permutation drop, refit; "
                              "residual = its population LAP; share = residual / true X_sep LAP",
            "sensitivity": f"truth from one separate-seed population per level ({N_POP:,} programs, seed "
                           f"{LEVEL_POP_SEED0} + 1,000 * level index), which has different area effects",
            "mcse": "sd / sqrt(R) for means; sqrt(p (1 - p) / R) for rates",
        },
        "checks": {"population_stream_reproduces_replicate": f"asserted for all {len(rows)} replicates",
                   "x0_plus_xpost_equals_xinf": f"asserted for all {len(rows)} replicates",
                   "aggregate_arm_reproduces_validate_instrument": repro},
        "score_mechanism": [summarize(rows, "score", lv) for lv in SCORE_LEVELS],
        "count_mechanism": [summarize(rows, "count", lv) for lv in COUNT_LEVELS],
        "runtime_seconds": round(time.time() - t0, 1),
        "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "workers": args.workers},
        "replicates": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for m, key in (("score", "score_mechanism"), ("count", "count_mechanism")):
        for s in out[key]:
            a, p = s["lap_separated"], s["permutation_importance"]
            print(f"{m} {list(s.values())[0]}: true {a['true_lap']['mean']:+.3f} bias {a['bias']['mean']:+.4f} "
                  f"cov {a['coverage_of_true_lap']['rate']:.2f} det {a['detection_rate_ci_above_0']['rate']:.2f} "
                  f"jointPI-true {p['joint_pi_minus_true_lap']['mean']:+.3f}", flush=True)
    print(f"reproduces validate_instrument: {repro}; {out['runtime_seconds']} s; wrote {args.out}")


if __name__ == "__main__":
    main()
