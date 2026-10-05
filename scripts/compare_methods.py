"""Comparison of the leakage-response method with common alternative leakage checks.

Three checks that practitioners use to look for leakage are applied to the same data, learner
(the default gradient-boosted trees) and test sets as the main audits, and set against LAP:

 S  Split comparison: performance with all features under a non-temporal split versus the
    temporal split. A temporal split is often taken as evidence that an evaluation is free of
    hindsight. CTO: random splits of the same test size (seeds 0-4), with LAP recomputed under
    each; TrialBench: the provided (non-temporal) split, read from trialbench_audit_v2.json.
 U  Univariate screen ("too good to be true"): the direction-free AUROC, max(AUROC, 1 - AUROC), of
    each naive feature on its own in the training block, missing values ranked lowest. Features
    are flagged at >= 0.90 and, as a sensitivity threshold, >= 0.80.
 P  Permutation importance in the naive model: the mean drop in test AUPRC over 20 permutations
    (seed 0) when post-decision columns are permuted one at a time (largest single drop and the
    sum of single drops) and jointly (one row permutation applied to all of them together).

Settings: CTO's labeling signals (15 of 17 distinct signals are post-start columns); TrialBench
Phases I-III (four post-start columns: actual enrollment and three final-record city covariates);
the 20-disease censored benchmark and the ICU records, where post-decision information enters the
same aggregate columns as pre-decision information (evidence counts and scores; last, minimum,
maximum, mean and count of measurements), so no column can be permuted and only the univariate
screen applies. The deployable and naive predictions are refitted here and checked against the
published outputs (cto_audit_v2.json, trialbench_lap_poststart.json,
censored_benchmark_20disease.json, physionet2012_audit.json).

Usage:  python scripts/compare_methods.py [--out outputs/method_comparison.json]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import audit_cto as A  # noqa: E402
import audit_cto_v2 as C  # noqa: E402
import audit_physionet2012 as E  # noqa: E402
import audit_trialbench as T  # noqa: E402
import audit_trialbench_v2 as V  # noqa: E402
from temporal_leakage_audit import features, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402

N_PERM, SEED = 20, 0
RANDOM_SPLIT_SEEDS = (0, 1, 2, 3, 4)
FLAG, FLAG_SENS = 0.90, 0.80
BENCH20 = "config/benchmark_20disease.yaml"


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def gbdt():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300,
                                          l2_regularization=1.0, random_state=0)


def auprc(y, p):
    return float(average_precision_score(y, p))


def univariate_auroc(x, y):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return 0.5
    x = np.where(np.isnan(x), np.nanmin(x) - 1.0, x)
    if len(np.unique(x)) < 2:
        return 0.5
    a = roc_auc_score(y, x)
    return max(a, 1.0 - a)


def screen(Xtr, ytr, post_cols=None):
    """Univariate screen on the training block; post_cols marks known post-decision columns."""
    au = {c: univariate_auroc(Xtr[c].values, ytr) for c in Xtr.columns}
    top = sorted(au.items(), key=lambda kv: -kv[1])
    res = {"n_features": len(au),
           "top5": [{"feature": c, "auroc": r4(a)} for c, a in top[:5]],
           "max_auroc": r4(top[0][1]),
           f"n_flagged_ge_{FLAG}": int(sum(a >= FLAG for a in au.values())),
           f"n_flagged_ge_{FLAG_SENS}": int(sum(a >= FLAG_SENS for a in au.values()))}
    if post_cols is not None:
        post = [c for c in au if c in post_cols]
        pre = [c for c in au if c not in post_cols]
        for tag, cols in (("post_decision", post), ("deployable", pre)):
            res[f"{tag}_n"] = len(cols)
            res[f"{tag}_max_auroc"] = r4(max(au[c] for c in cols))
            res[f"{tag}_n_flagged_ge_{FLAG}"] = int(sum(au[c] >= FLAG for c in cols))
            res[f"{tag}_n_flagged_ge_{FLAG_SENS}"] = int(sum(au[c] >= FLAG_SENS for c in cols))
        res[f"post_decision_flagged_ge_{FLAG}"] = sorted(c for c in post if au[c] >= FLAG)
        res[f"post_decision_flagged_ge_{FLAG_SENS}"] = sorted(c for c in post if au[c] >= FLAG_SENS)
    return res


def permutation(clf, Xte, yte, cols, post_cols, n_perm=N_PERM, seed=SEED):
    """Drop in test AUPRC of the fitted naive model when post-decision columns are permuted."""
    rng = np.random.default_rng(seed)
    X = np.asarray(Xte, dtype=float)
    base = auprc(yte, clf.predict_proba(X)[:, 1])
    idx = [cols.index(c) for c in post_cols]
    single = {}
    for c, j in zip(post_cols, idx):
        drops = []
        for _ in range(n_perm):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            drops.append(base - auprc(yte, clf.predict_proba(Xp)[:, 1]))
        single[c] = float(np.mean(drops))
    joint = []
    for _ in range(n_perm):
        Xp = X.copy()
        Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
        joint.append(base - auprc(yte, clf.predict_proba(Xp)[:, 1]))
    top = max(single.items(), key=lambda kv: kv[1])
    return {"naive_auprc": r4(base),
            "joint_drop": r4(np.mean(joint)), "joint_drop_sd": r4(np.std(joint, ddof=1)),
            "largest_single_drop": r4(top[1]), "largest_single_feature": top[0],
            "sum_of_single_drops": r4(sum(single.values())),
            "single_drops": {c: r4(v) for c, v in sorted(single.items(), key=lambda kv: -kv[1])}}


# --------------------------------------------------------------------------- CTO
def cto():
    j = A.load("data/cto")
    cut = int(j["start_year"].quantile(0.6))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    start, post = list(C.START_V2), list(C.COMPL_V2) + list(C.EXT_V2)
    allc = start + post
    ytr, yte = tr["labels"].values, te["labels"].values
    naive = gbdt().fit(tr[allc].values, ytr)
    p_all = naive.predict_proba(te[allc].values)[:, 1]
    p_start = A._fit_predict(tr, te, start)
    lap = auprc(yte, p_all) - auprc(yte, p_start)
    pub = json.load(open("outputs/cto_audit_v2.json"))["A_corrected_tiers"]
    check = (r4(auprc(yte, p_all)) == pub["tiers"]["t2_all"]["auprc"]
             and r4(auprc(yte, p_start)) == pub["tiers"]["t0_start"]["auprc"])
    # S: random splits with the test size of the temporal split
    rnd = []
    n, n_te = len(j), len(te)
    for s in RANDOM_SPLIT_SEEDS:
        rng = np.random.default_rng(s)
        m = np.zeros(n, bool)
        m[rng.choice(n, n_te, replace=False)] = True
        rtr, rte = j[~m], j[m]
        pa = A._fit_predict(rtr, rte, allc)
        ps = A._fit_predict(rtr, rte, start)
        ry = rte["labels"].values
        rnd.append({"seed": s, "test_base_rate": r4(ry.mean()), "auprc_all": r4(auprc(ry, pa)),
                    "auroc_all": r4(roc_auc_score(ry, pa)), "auprc_start": r4(auprc(ry, ps)),
                    "LAP": r4(auprc(ry, pa) - auprc(ry, ps))})
    # the fix a screen suggests: drop the top-flagged feature and refit
    uni = screen(tr[allc], ytr, set(post))
    perm = permutation(naive, te[allc].values, yte, allc, post)
    dropped = perm["largest_single_feature"]
    assert dropped in uni[f"post_decision_flagged_ge_{FLAG}"]
    p_drop = A._fit_predict(tr, te, [c for c in allc if c != dropped])
    d = C.clustered_draws(yte, {"start": p_start, "drop": p_drop, "all": p_all}, te["source"].values)
    after_drop = {"dropped_feature": dropped, "auprc": r4(auprc(yte, p_drop)),
                  "residual_LAP": r4(auprc(yte, p_drop) - auprc(yte, p_start)),
                  "residual_LAP_ci": C.q(d["drop"]["auprc"] - d["start"]["auprc"]),
                  "LAP_ci_recomputed": C.q(d["all"]["auprc"] - d["start"]["auprc"])}
    split = {"temporal_split": {"test_base_rate": r4(yte.mean()), "auprc_all": r4(auprc(yte, p_all)),
                                "auroc_all": r4(roc_auc_score(yte, p_all)), "LAP": r4(lap)},
             "random_splits": rnd,
             "random_mean_auprc_all": r4(np.mean([r["auprc_all"] for r in rnd])),
             "random_mean_auroc_all": r4(np.mean([r["auroc_all"] for r in rnd])),
             "random_mean_LAP": r4(np.mean([r["LAP"] for r in rnd])),
             "random_minus_temporal_auprc_all": r4(np.mean([r["auprc_all"] for r in rnd]) - auprc(yte, p_all))}
    return {"n_train": int(len(tr)), "n_test": int(len(te)), "post_decision_columns": post,
            "LAP": r4(lap), "LAP_ci_published": pub["LAP_all_minus_start"]["auprc"]["ci"],
            "matches_cto_audit_v2": bool(check),
            "split_comparison": split,
            "univariate_screen": uni,
            "permutation_importance": perm,
            "after_dropping_the_flagged_feature": after_drop}


# --------------------------------------------------------------------------- TrialBench
def trialbench():
    T.ensure_data()
    v2 = json.load(open("outputs/trialbench_audit_v2.json"))["T2_provided_vs_temporal"]
    lapj = json.load(open("outputs/trialbench_lap_poststart.json"))["phases"]
    out = {}
    for ph in T.PHASES:
        P = V.load_phase(ph)
        fs = V.feature_sets(P["X"])
        allc = fs["all_features"]
        post = [c for c in allc if c in V.NON_START]
        mask = P["temp_test"]
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        X, y = P["X"], P["y"]
        naive = gbdt().fit(X[allc].iloc[tr].values, y[tr])
        p_all = naive.predict_proba(X[allc].iloc[te].values)[:, 1]
        p_start = V.fit_cols(P, fs["ablated_non_start_removed"], tr, te)
        lap = auprc(y[te], p_all) - auprc(y[te], p_start)
        ref = lapj[ph]["temporal_split"]
        check = (r4(auprc(y[te], p_all)) == ref["arms"]["all_features"]["auprc"]
                 and r4(auprc(y[te], p_start)) == ref["arms"]["ablated_non_start_removed"]["auprc"])
        a = v2[ph]["all_features"]
        split = {s: {"auprc_all": a[s]["auprc"], "auroc_all": a[s]["auroc"]}
                 for s in ("provided_split", "temporal_split")}
        split["provided_minus_temporal_auprc_all"] = r4(a["provided_split"]["auprc"] - a["temporal_split"]["auprc"])
        split["note"] = "provided split is non-temporal; see trialbench_audit_v2.json T2 and T3"
        out[ph] = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_base_rate": r4(y[te].mean()),
                   "post_decision_columns": post, "LAP": r4(lap),
                   "LAP_ci_published": ref["paired_differences"]["LAP_total_all_minus_ablated"]["auprc"]["ci"],
                   "matches_trialbench_lap_poststart": bool(check),
                   "split_comparison": split,
                   "univariate_screen": screen(X[allc].iloc[tr], y[tr], set(post)),
                   "permutation_importance": permutation(naive, X[allc].iloc[te].values, y[te], allc, post)}
        print(ph, "LAP", out[ph]["LAP"], "perm joint", out[ph]["permutation_importance"]["joint_drop"],
              "max post AUROC", out[ph]["univariate_screen"]["post_decision_max_auroc"], flush=True)
    return out


# --------------------------------------------------------------------------- censored benchmark
def censored():
    cfg = get_config(BENCH20)
    programs = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "programs.csv"))
    evidence = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "evidence.csv"))
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    yrs = programs["info_time"]
    tmax, hi = int(yrs.quantile(0.60)), int(yrs.max())
    tr, te, _ = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    ytr, yte = y.loc[tr].values, y.loc[te].values
    p_n = gbdt().fit(Xn.loc[tr].values, ytr).predict_proba(Xn.loc[te].values)[:, 1]
    p_c = gbdt().fit(Xc.loc[tr].values, ytr).predict_proba(Xc.loc[te].values)[:, 1]
    pub = json.load(open("outputs/censored_benchmark_20disease.json"))["leakage_response_curve"]
    lap = auprc(yte, p_n) - auprc(yte, p_c)
    changed = [c for c in Xn.columns if not np.allclose(Xn[c].fillna(-9).values, Xc[c].fillna(-9).values)]
    return {"n_train": int(len(tr)), "n_test": int(len(te)), "LAP": r4(lap),
            "LAP_ci_published": pub["total_LAP_auprc_ci"],
            "matches_censored_benchmark_20disease": bool(r4(auprc(yte, p_n)) == pub["naive_auprc"]
                                                         and r4(auprc(yte, p_c)) == pub["deployable_auprc"]),
            "columns_carrying_post_decision_information": len(changed),
            "columns_total": int(Xn.shape[1]),
            "note": "post-decision evidence enters the same count and score columns as pre-decision "
                    "evidence; no separate column to permute",
            "univariate_screen_naive": screen(Xn.loc[tr], ytr),
            "univariate_screen_deployable": screen(Xc.loc[tr], ytr)}


# --------------------------------------------------------------------------- ICU
def icu():
    E.download()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    y = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    X24 = E.features(long, static, 24 * 60, params)
    X48 = E.features(long, static, 48 * 60, params)
    p24 = E.fit_predict(X24[is_tr], y[is_tr], X24[~is_tr])
    p48 = E.fit_predict(X48[is_tr], y[is_tr], X48[~is_tr])
    lap = auprc(y[~is_tr], p48) - auprc(y[~is_tr], p24)
    pub = json.load(open("outputs/physionet2012_audit.json"))["t24"]["LAP"]
    changed = [c for c in X48.columns
               if not np.allclose(X48[c].fillna(-9).values, X24[c].fillna(-9).values)]
    return {"n_train": int(is_tr.sum()), "n_test": int((~is_tr).sum()), "LAP": r4(lap),
            "LAP_ci_published": pub["auprc_ci"], "matches_physionet2012_audit": bool(r4(lap) == pub["auprc"]),
            "columns_carrying_post_decision_information": len(changed), "columns_total": int(X48.shape[1]),
            "note": "measurements after 24 h enter the same last/min/max/mean/count columns as earlier "
                    "measurements; no separate column to permute; no calendar dates for a temporal split",
            "univariate_screen_naive_48h": screen(X48[is_tr], y[is_tr]),
            "univariate_screen_deployable_24h": screen(X24[is_tr], y[is_tr])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/method_comparison.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"script": "scripts/compare_methods.py",
           "settings": {"learner": "HistGradientBoosting (max_depth 3, lr 0.06, 300 iter, l2 1.0, seed 0)",
                        "univariate_screen": f"direction-free training-block AUROC of each naive feature; "
                                             f"flag >= {FLAG} (sensitivity >= {FLAG_SENS})",
                        "permutation": f"mean drop in test AUPRC of the naive model over {N_PERM} "
                                       f"permutations, seed {SEED}; single columns and joint",
                        "random_split_seeds": list(RANDOM_SPLIT_SEEDS)}}
    out["cto"] = cto()
    print("CTO", out["cto"]["LAP"], out["cto"]["permutation_importance"]["joint_drop"],
          out["cto"]["split_comparison"]["random_mean_auprc_all"], flush=True)
    out["trialbench"] = trialbench()
    out["censored_benchmark_20disease"] = censored()
    print("censored", out["censored_benchmark_20disease"]["LAP"],
          out["censored_benchmark_20disease"]["univariate_screen_naive"]["max_auroc"], flush=True)
    out["icu_physionet2012_decision_24h"] = icu()
    print("ICU", out["icu_physionet2012_decision_24h"]["LAP"],
          out["icu_physionet2012_decision_24h"]["univariate_screen_naive_48h"]["max_auroc"], flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
