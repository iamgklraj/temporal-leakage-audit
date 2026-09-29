"""Robustness of the leakage audits to the choice of learner and of the CTO temporal cutoff.

A. Learner sensitivity. The three audits are re-run with two further learners besides the
   paper's gradient-boosted trees (HistGradientBoostingClassifier, max_depth 3, lr 0.06,
   300 iterations, l2 1.0, seed 0):
     * logreg_l2     : median imputation -> standardization -> L2 logistic regression (C = 1)
     * random_forest : 500 trees, min_samples_leaf = 5, seed 0 (native missing-value support)
   on (i) the CTO labeling-signal audit (start-time / +during-trial / all tiers, LAP and its
   during/post components; sponsor-clustered paired bootstrap, 1,000 resamples), (ii) the
   TrialBench provided-vs-temporal split comparison (trial-level bootstrap, 1,000), and
   (iii) the 20-disease as-of-time-censored drug-program benchmark (leakage-response curve,
   LAP = h=inf - h=0; target-clustered paired bootstrap, 500).
B. CTO temporal-cutoff sensitivity (default learner): training block = trials starting in or
   before the 50th, 60th (paper) and 70th percentile of start year. Start years are integers
   and the 60th and 70th percentiles are both 2020, so the next distinct later cutoff (80th
   percentile, 2021) is added.

No logic is re-implemented. The existing code paths are called unchanged:
audit_cto.load/_fit_predict/_clustered_samples, audit_trialbench.audit_phase and
leakage.leakage_response_curve (via models.fit_gbdt). Each of these builds its classifier by
looking the name ``HistGradientBoostingClassifier`` up in its own module; the ``learner``
context manager below rebinds that name, for the duration of a block only, to a shim that
returns the requested learner (all three paths only call ``.fit`` and ``.predict_proba``).
The default learner is itself run through the shim, and the script checks that it reproduces
the published outputs/cto_audit.json, outputs/trialbench_audit.json and
outputs/censored_benchmark_20disease.json exactly.

Usage:  python scripts/robustness.py [--out outputs/robustness.json]
        (run from the repository root; ~5-15 min with 4 threads)
"""
import os

# Limit native thread pools before numpy / scikit-learn are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import audit_cto  # noqa: E402
import audit_trialbench  # noqa: E402
from temporal_leakage_audit import features, leakage, models, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402

N_JOBS = 4
CTO_N_BOOT = 1000                      # as in audit_cto.py
BENCH_CONFIG = "config/benchmark_20disease.yaml"   # n_boot = 500, as published
CTO_QUANTILES = (0.5, 0.6, 0.7, 0.8)   # 0.6 = paper; 0.8 added (0.7 maps to the same year as 0.6)
REFS = {"cto": "outputs/cto_audit.json",
        "trialbench": "outputs/trialbench_audit.json",
        "censored_benchmark_20disease": "outputs/censored_benchmark_20disease.json"}


# --------------------------------------------------------------------------- #
# Learners
# --------------------------------------------------------------------------- #

def make_hgb(random_state=0):
    """The paper's learner, with exactly the hyper-parameters hard-coded in the audits."""
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300,
                                          l2_regularization=1.0, random_state=random_state)


def make_logreg(random_state=0):  # noqa: ARG001 -- lbfgs is deterministic
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=10000))])


def make_rf(random_state=0):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                  random_state=random_state, n_jobs=N_JOBS)


LEARNERS = {"hgb_default": make_hgb, "logreg_l2": make_logreg, "random_forest": make_rf}
LEARNER_SPECS = {
    "hgb_default": "HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300, "
                   "l2_regularization=1.0, random_state=0) -- the paper's learner",
    "logreg_l2": "Pipeline(SimpleImputer(median), StandardScaler(), LogisticRegression(C=1.0, "
                 "L2 penalty, lbfgs, max_iter=10000))",
    "random_forest": f"RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0, "
                     f"n_jobs={N_JOBS}); missing values handled natively by scikit-learn",
}

_PATCHED_MODULES = (audit_cto, audit_trialbench, models)


@contextlib.contextmanager
def learner(factory):
    """Run the unchanged audit code with ``factory`` in place of the hard-coded GBDT.

    audit_cto._fit_predict, audit_trialbench._fit and models.fit_gbdt each call
    ``HistGradientBoostingClassifier(max_depth=3, ..., random_state=s)`` through their module
    namespace. Within this block that name resolves to a shim that drops the GBDT
    hyper-parameters and returns ``factory(random_state=s)``; the originals are restored on exit.
    """
    saved = [m.HistGradientBoostingClassifier for m in _PATCHED_MODULES]

    def shim(**kw):
        return factory(random_state=kw.get("random_state", 0))

    try:
        for m in _PATCHED_MODULES:
            m.HistGradientBoostingClassifier = shim
        yield
    finally:
        for m, s in zip(_PATCHED_MODULES, saved):
            m.HistGradientBoostingClassifier = s


# --------------------------------------------------------------------------- #
# (i) CTO -- the computation of audit_cto.main(), with the split quantile as a parameter
# --------------------------------------------------------------------------- #

def cto_audit(j, quantile=0.6, n_boot=CTO_N_BOOT):
    """Same steps, seeds and rounding as audit_cto.main() (per-LF decomposition omitted)."""
    cut = int(j["start_year"].quantile(quantile))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    yte = te["labels"].values
    base = float(te["labels"].mean())

    preds, points = {}, {}
    for name, cols in audit_cto.TIERS:
        p = audit_cto._fit_predict(tr, te, cols)
        preds[name] = p
        points[name] = {"auprc": round(audit_cto._auprc(yte, p), 4),
                        "auroc": round(audit_cto._auroc(yte, p), 4), "n_lfs": len(cols)}
    samp = audit_cto._clustered_samples(yte, preds, te["source"].values, n_boot=n_boot, seed=0)

    def ci(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)]

    lap = np.array([n - s for n, s in zip(samp["h2_post_naive"], samp["h0_start"])
                    if np.isfinite(n) and np.isfinite(s)])
    lap_pt = points["h2_post_naive"]["auprc"] - points["h0_start"]["auprc"]
    frac = lap_pt / points["h2_post_naive"]["auprc"]

    def paired(a_key, b_key):
        draws = np.array([a - b for a, b in zip(samp[a_key], samp[b_key])
                          if np.isfinite(a) and np.isfinite(b)])
        return {"auprc": round(points[a_key]["auprc"] - points[b_key]["auprc"], 4),
                "ci": [round(float(np.quantile(draws, .025)), 4),
                       round(float(np.quantile(draws, .975)), 4)]}

    return {
        "split_quantile": quantile,
        "n": len(j), "n_train": len(tr), "n_test": len(te),
        "train_start_year_le": cut, "test_base_rate": round(base, 4),
        "n_sponsors": int(j["source"].nunique()),
        "n_test_sponsors": int(te["source"].nunique()),
        "tiers": {name: {**points[name], "auprc_ci": ci(samp[name])} for name, _ in audit_cto.TIERS},
        "LAP_auprc": round(lap_pt, 4),
        "LAP_auprc_ci": [round(float(np.quantile(lap, .025)), 4),
                         round(float(np.quantile(lap, .975)), 4)],
        "LAP_p_gt_0": round(float(np.mean(lap > 0)), 4),
        "leakage_fraction_of_auprc": round(float(frac), 4),
        "LAP_components": {"during_trial": paired("h1_during", "h0_start"),
                           "post_completion": paired("h2_post_naive", "h1_during")},
        "n_boot": n_boot, "n_boot_valid": int(len(lap)),
    }


# --------------------------------------------------------------------------- #
# (ii) TrialBench -- audit_trialbench.audit_phase() per phase, as in audit_trialbench.main()
# --------------------------------------------------------------------------- #

def trialbench_audit():
    audit_trialbench.ensure_data()  # no-op when the tables are present
    phases = [audit_trialbench.audit_phase(ph) for ph in audit_trialbench.PHASES]
    out = {"phases": [{k: v for k, v in r.items() if not k.startswith("_")} for r in phases]}
    for k in ("auprc", "auroc", "auprc_lift"):
        out[f"mean_provided_minus_temporal_{k}"] = round(float(np.mean(
            [r["provided_minus_temporal"][k]["point"] for r in phases])), 4)
    return out


# --------------------------------------------------------------------------- #
# (iii) Censored drug-program benchmark -- the main-split leakage-response curve, exactly as
# in audit_censored_benchmark.main()
# --------------------------------------------------------------------------- #

def load_benchmark(config=BENCH_CONFIG):
    cfg = get_config(config)
    programs = pd.read_csv(os.path.join(cfg["data_dir"], "programs.csv"))
    evidence = pd.read_csv(os.path.join(cfg["data_dir"], "evidence.csv"))
    _, _, _, meta, _ = features.build(programs, evidence)
    yrs = programs["info_time"]
    tmax, hi = int(yrs.quantile(0.60)), int(yrs.max())
    tr, te, info = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    return {"cfg": cfg, "programs": programs, "evidence": evidence, "meta": meta,
            "tr": tr, "te": te, "info": info}


def benchmark_audit(b):
    cfg = b["cfg"]
    lrc = leakage.leakage_response_curve(
        b["programs"], b["evidence"], b["tr"], b["te"],
        clusters=b["meta"].loc[b["te"], "target_id"].values,
        n_boot=cfg["bootstrap"]["n_boot"], budget_frac=cfg["decision"]["budget_frac"])
    return {"main_split": b["info"], "leakage_response_curve": lrc}


# --------------------------------------------------------------------------- #
# Reproduction check
# --------------------------------------------------------------------------- #

def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _diff(ref, new, path=""):
    """List every leaf where ``new`` differs from ``ref`` (exact equality; NaN == NaN)."""
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return [f"{path}: type {type(new).__name__} != dict"]
        out = []
        for k in ref:
            out += ([f"{path}/{k}: missing"] if k not in new else _diff(ref[k], new[k], f"{path}/{k}"))
        return out
    if isinstance(ref, list):
        if not isinstance(new, list) or len(new) != len(ref):
            return [f"{path}: {new!r} != {ref!r}"]
        return [m for i, (a, b) in enumerate(zip(ref, new)) for m in _diff(a, b, f"{path}[{i}]")]
    if isinstance(ref, float) and isinstance(new, float) and math.isnan(ref) and math.isnan(new):
        return []
    return [] if ref == new else [f"{path}: reproduced {new!r} != published {ref!r}"]


def reproduction_check(name, new):
    ref = json.load(open(REFS[name]))
    new = _jsonable(new)
    if name == "cto":
        keys = ["n", "n_train", "n_test", "train_start_year_le", "test_base_rate", "n_sponsors",
                "tiers", "LAP_auprc", "LAP_auprc_ci", "LAP_p_gt_0",
                "leakage_fraction_of_auprc", "LAP_components"]
    elif name == "trialbench":
        keys = ["phases", "mean_provided_minus_temporal_auprc",
                "mean_provided_minus_temporal_auroc", "mean_provided_minus_temporal_auprc_lift"]
    else:
        keys = ["main_split", "leakage_response_curve"]
    mism = [m for k in keys for m in _diff(ref[k], new[k], k)]
    n_leaves = sum(1 for k in keys for _ in _leaves(ref[k]))
    return {"reference": REFS[name], "compared_fields": keys, "n_leaf_values_compared": n_leaves,
            "exact_match": not mism, "mismatches": mism}


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


# --------------------------------------------------------------------------- #

def _run(label, fn, *args, **kw):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        res = fn(*args, **kw)
    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    dt = time.time() - t0
    print(f"  [{label}] {dt:.1f}s" + (f"  ({n_conv} ConvergenceWarnings)" if n_conv else ""), flush=True)
    return res, round(dt, 1), n_conv


def main():
    ap = argparse.ArgumentParser(description="Learner and temporal-cutoff robustness of the audits.")
    ap.add_argument("--out", default="outputs/robustness.json")
    args = ap.parse_args()
    out_path = os.path.abspath(args.out)
    for ref in REFS.values():
        if os.path.abspath(os.path.join(ROOT, ref)) == out_path:
            raise SystemExit(f"refusing to overwrite the reference output {ref}")
    os.chdir(ROOT)  # the audit modules use repository-relative data paths
    t_start = time.time()

    cto = audit_cto.load("data/cto")
    bench = load_benchmark()

    learner_res = {"cto": {}, "trialbench": {}, "censored_benchmark_20disease": {}}
    runtime, conv = {}, {}
    for lname, factory in LEARNERS.items():
        print(f"learner {lname}", flush=True)
        with learner(factory):
            for task, fn, a in (("cto", cto_audit, (cto,)),
                                ("trialbench", trialbench_audit, ()),
                                ("censored_benchmark_20disease", benchmark_audit, (bench,))):
                res, dt, nc = _run(f"{lname}/{task}", fn, *a)
                learner_res[task][lname] = res
                runtime[f"{lname}/{task}"] = dt
                conv[f"{lname}/{task}"] = nc

    check = {k: reproduction_check(k, learner_res[k]["hgb_default"]) for k in REFS}
    for k, v in check.items():
        print(f"reproduction check {k}: exact_match={v['exact_match']} "
              f"({v['n_leaf_values_compared']} values)" +
              ("" if v["exact_match"] else f"  {v['mismatches'][:5]}"), flush=True)

    # B. CTO temporal cutoff (default learner). Distinct cut years are computed once.
    print("CTO cutoff sensitivity (default learner)", flush=True)
    by_year, cutoff_res = {}, []
    with learner(make_hgb):
        for q in CTO_QUANTILES:
            year = int(cto["start_year"].quantile(q))
            if year not in by_year:
                by_year[year], dt, _ = _run(f"cto q={q} (<= {year})", cto_audit, cto, quantile=q)
                runtime[f"cto_cutoff/q{q}"] = dt
            r = dict(by_year[year], split_quantile=q)
            r["train_fraction"] = round(r["n_train"] / r["n"], 4)
            r["requested"] = q in (0.5, 0.6, 0.7)
            r["same_split_as_paper_default"] = year == int(cto["start_year"].quantile(0.6))
            cutoff_res.append(r)

    out = {
        "description": ("Learner sensitivity of the CTO, TrialBench and 20-disease censored-benchmark "
                        "audits, and temporal-cutoff sensitivity of the CTO audit. All intervals are "
                        "95% percentile bootstrap CIs with the clustering unit and resample count of "
                        "the original analysis."),
        "settings": {
            "learners": LEARNER_SPECS,
            "bootstrap": {"cto": f"sponsor-clustered, {CTO_N_BOOT} resamples, seed 0, paired across tiers",
                          "trialbench": f"trial-level, {audit_trialbench.N_BOOT} resamples per test set; "
                                        "provided-minus-temporal from independent bootstraps (seeds 0/1)",
                          "censored_benchmark_20disease":
                              f"target-clustered, {bench['cfg']['bootstrap']['n_boot']} resamples, seed 0, "
                              "paired across horizons"},
            "same_bootstrap_resamples_across_learners": True,
            "cto_split_rule": "train = start_year <= int(quantile(start_year, q)); test = later",
            "cto_quantile_note": ("start years are integers: q=0.6 and q=0.7 both give 2020 (identical "
                                  "split); q=0.8 (2021) is added as the next distinct later cutoff"),
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        },
        "reproduction_check": check,
        "learner_sensitivity": learner_res,
        "cto_cutoff_sensitivity": cutoff_res,
        "convergence_warnings": conv,
        "runtime_seconds": {**runtime, "total": round(time.time() - t_start, 1)},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # Summary.
    print("\n=== CTO (sponsor-clustered CIs) ===")
    for lname, r in learner_res["cto"].items():
        t = r["tiers"]
        print(f"  {lname:14s} start {t['h0_start']['auprc']:.3f}/{t['h0_start']['auroc']:.3f}  "
              f"+during {t['h1_during']['auprc']:.3f}/{t['h1_during']['auroc']:.3f}  "
              f"all {t['h2_post_naive']['auprc']:.3f}/{t['h2_post_naive']['auroc']:.3f}  "
              f"LAP {r['LAP_auprc']:+.3f} {r['LAP_auprc_ci']}  "
              f"during {r['LAP_components']['during_trial']['auprc']:+.3f} "
              f"post {r['LAP_components']['post_completion']['auprc']:+.3f}")
    print("=== TrialBench provided - temporal ===")
    for lname, r in learner_res["trialbench"].items():
        s = "  ".join(f"{p['phase']} dAUPRC {p['provided_minus_temporal']['auprc']['point']:+.3f} "
                      f"dAUROC {p['provided_minus_temporal']['auroc']['point']:+.3f}" for p in r["phases"])
        print(f"  {lname:14s} {s}  mean {r['mean_provided_minus_temporal_auprc']:+.3f}/"
              f"{r['mean_provided_minus_temporal_auroc']:+.3f}")
    print("=== Censored benchmark (target-clustered) ===")
    for lname, r in learner_res["censored_benchmark_20disease"].items():
        c = r["leakage_response_curve"]
        print(f"  {lname:14s} h0 {c['deployable_auprc']:.3f}  hinf {c['naive_auprc']:.3f}  "
              f"LAP {c['total_LAP_auprc']:+.3f} {c['total_LAP_auprc_ci']}")
    print("=== CTO cutoff (default learner) ===")
    for r in cutoff_res:
        print(f"  q={r['split_quantile']} <= {r['train_start_year_le']}  n_test {r['n_test']}  "
              f"base {r['test_base_rate']:.3f}  start {r['tiers']['h0_start']['auprc']:.3f}/"
              f"{r['tiers']['h0_start']['auroc']:.3f}  all {r['tiers']['h2_post_naive']['auprc']:.3f}  "
              f"LAP {r['LAP_auprc']:+.3f} {r['LAP_auprc_ci']}")
    print(f"wrote {out_path}  ({out['runtime_seconds']['total']:.0f}s)")


if __name__ == "__main__":
    main()
