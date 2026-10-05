"""Variance-inflated intervals for the real audits.

The paired cluster bootstrap of the main analyses holds the fitted models fixed and describes
uncertainty about LAP for those models (a conditional estimand). The variance-inflated interval
adds training variability, as validated on synthetic data (scripts/validate_refit_bootstrap.py):

    LAP +/- 1.96 * sqrt(v_test + v_train),

where v_test is the variance of the fixed-model bootstrap draws of LAP and v_train is the variance
of LAP across refits of both learners (deployable and naive) on training clusters resampled with
replacement, each evaluated on the full, fixed test set. Clusters as in the main analyses: sponsors
(CTO), trials (TrialBench), drug targets (censored drug-program benchmark) and patients (ICU).

Reuses the data loaders, splits, learners and fixed-model bootstraps of the audit scripts unchanged;
the point estimates and fixed-model intervals are checked against the published outputs.

Usage:  python scripts/variance_inflated_intervals.py [--out outputs/variance_inflated_intervals.json]
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
from sklearn.metrics import average_precision_score  # noqa: E402

import audit_cto as A  # noqa: E402
import audit_cto_v2 as C  # noqa: E402
import audit_physionet2012 as E  # noqa: E402
import audit_trialbench as T  # noqa: E402
import audit_trialbench_v2 as V  # noqa: E402
import audit_trialbench_v2_lap as L  # noqa: E402
from temporal_leakage_audit import features, leakage, models, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402

N_REFIT = {"cto": 100, "trialbench": 100, "censored": 100, "icu": 50}
SEED = 0
BENCH20 = "config/benchmark_20disease.yaml"


def r4(x):
    return round(float(x), 4)


def summary(lap, lap_draws, refit_laps, published_ci):
    lap_draws = np.asarray([x for x in lap_draws if np.isfinite(x)])
    refit_laps = np.asarray([x for x in refit_laps if np.isfinite(x)])
    v_test, v_train = float(np.var(lap_draws, ddof=1)), float(np.var(refit_laps, ddof=1))
    half = 1.96 * np.sqrt(v_test + v_train)
    return {"LAP": r4(lap), "fixed_model_ci_published": published_ci,
            "fixed_model_ci_recomputed": [r4(np.quantile(lap_draws, 0.025)), r4(np.quantile(lap_draws, 0.975))],
            "sd_test": r4(np.sqrt(v_test)), "sd_train": r4(np.sqrt(v_train)),
            "n_refits": int(len(refit_laps)), "refit_mean_LAP": r4(np.mean(refit_laps)),
            "variance_inflated_ci": [r4(lap - half), r4(lap + half)],
            "variance_inflated_excludes_zero": bool(lap - half > 0 or lap + half < 0)}


def resample_clusters(clusters, rng):
    uniq = np.unique(clusters)
    c2r = {c: np.where(clusters == c)[0] for c in uniq}
    return np.concatenate([c2r[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])


def cto():
    j = A.load("data/cto")
    cut = int(j["start_year"].quantile(0.6))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    start, allc = list(C.START_V2), list(C.START_V2) + list(C.COMPL_V2) + list(C.EXT_V2)
    y = te["labels"].values
    p0, p1 = A._fit_predict(tr, te, start), A._fit_predict(tr, te, allc)
    d = C.clustered_draws(y, {"s": p0, "a": p1}, te["source"].values)
    rng = np.random.default_rng(SEED)
    refits = []
    for _ in range(N_REFIT["cto"]):
        trb = tr.iloc[resample_clusters(tr["source"].values, rng)]
        if trb["labels"].nunique() < 2:
            continue
        refits.append(average_precision_score(y, A._fit_predict(trb, te, allc))
                      - average_precision_score(y, A._fit_predict(trb, te, start)))
    pub = json.load(open("outputs/cto_audit_v2.json"))["A_corrected_tiers"]["LAP_all_minus_start"]["auprc"]["ci"]
    lap = average_precision_score(y, p1) - average_precision_score(y, p0)
    return summary(lap, d["a"]["auprc"] - d["s"]["auprc"], refits, pub)


def trialbench():
    T.ensure_data()
    pubj = json.load(open("outputs/trialbench_lap_poststart.json"))["phases"]
    out = {}
    for ph in T.PHASES:
        P = V.load_phase(ph)
        fs = V.feature_sets(P["X"])
        mask = P["temp_test"]
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        y = P["y"][te]
        pa = V.fit_cols(P, fs["all_features"], tr, te)
        pb = V.fit_cols(P, fs["ablated_non_start_removed"], tr, te)
        d = L.paired_draws(y, {"a": pa, "b": pb})
        rng = np.random.default_rng(SEED)
        refits = []
        for _ in range(N_REFIT["trialbench"]):
            trb = tr[rng.integers(0, len(tr), len(tr))]
            refits.append(average_precision_score(y, V.fit_cols(P, fs["all_features"], trb, te))
                          - average_precision_score(y, V.fit_cols(P, fs["ablated_non_start_removed"], trb, te)))
        pub = pubj[ph]["temporal_split"]["paired_differences"]["LAP_total_all_minus_ablated"]["auprc"]["ci"]
        lap = average_precision_score(y, pa) - average_precision_score(y, pb)
        out[ph] = summary(lap, d["a"]["auprc"] - d["b"]["auprc"], refits, pub)
        print(ph, out[ph]["LAP"], out[ph]["variance_inflated_ci"], flush=True)
    return out


def censored():
    cfg = get_config(BENCH20)
    programs = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "programs.csv"))
    evidence = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "evidence.csv"))
    Xc, Xn, y, meta, _ = features.build(programs, evidence)
    yrs = programs["info_time"]
    tmax, hi = int(yrs.quantile(0.60)), int(yrs.max())
    tr, te, _ = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    yte = y.loc[te].values

    def fitp(X, rows):
        clf = models.fit_gbdt(X.loc[rows], y.loc[rows].values, seed=0)
        return models.predict_gbdt(clf, X.loc[te])

    pn, pc = fitp(Xn, tr), fitp(Xc, tr)
    clusters = meta.loc[te, "target_id"].values
    samples = leakage._clustered_auprc_samples(yte, {"n": pn, "c": pc}, clusters,
                                               n_boot=cfg["bootstrap"]["n_boot"], seed=0)
    draws = np.asarray(samples["n"]) - np.asarray(samples["c"])
    rng = np.random.default_rng(SEED)
    tr_idx = np.asarray(tr)
    refits = []
    for _ in range(N_REFIT["censored"]):
        rows = tr_idx[resample_clusters(meta.loc[tr, "target_id"].values, rng)]
        if y.loc[rows].nunique() < 2:
            continue
        refits.append(average_precision_score(yte, fitp(Xn, rows)) - average_precision_score(yte, fitp(Xc, rows)))
    pub = json.load(open("outputs/censored_benchmark_20disease.json"))["leakage_response_curve"]["total_LAP_auprc_ci"]
    lap = average_precision_score(yte, pn) - average_precision_score(yte, pc)
    return summary(lap, draws, refits, pub)


def icu():
    E.download()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    yy = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    X24 = E.features(long, static, 24 * 60, params)
    X48 = E.features(long, static, 48 * 60, params)
    ytr, yte = yy[is_tr], yy[~is_tr]
    X24tr, X48tr, X24te, X48te = X24[is_tr], X48[is_tr], X24[~is_tr], X48[~is_tr]
    p24, p48 = E.fit_predict(X24tr, ytr, X24te), E.fit_predict(X48tr, ytr, X48te)
    b = E.paired_boot(yte, {"24": p24, "48": p48})
    rng = np.random.default_rng(SEED)
    refits = []
    for _ in range(N_REFIT["icu"]):
        rows = rng.integers(0, len(ytr), len(ytr))
        refits.append(average_precision_score(yte, E.fit_predict(X48tr.iloc[rows], ytr[rows], X48te))
                      - average_precision_score(yte, E.fit_predict(X24tr.iloc[rows], ytr[rows], X24te)))
    pub = json.load(open("outputs/physionet2012_audit.json"))["t24"]["LAP"]["auprc_ci"]
    lap = average_precision_score(yte, p48) - average_precision_score(yte, p24)
    return summary(lap, b["48"]["auprc"] - b["24"]["auprc"], refits, pub)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/variance_inflated_intervals.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"script": "scripts/variance_inflated_intervals.py",
           "method": "LAP +/- 1.96 sqrt(v_test + v_train); v_test from the fixed-model paired cluster bootstrap of "
                     "the main analyses, v_train from refits of both learners on training clusters resampled with "
                     "replacement, evaluated on the full fixed test set",
           "n_refits": N_REFIT, "seed": SEED}
    out["cto"] = cto()
    print("CTO", out["cto"], flush=True)
    out["trialbench"] = trialbench()
    out["censored_benchmark_20disease"] = censored()
    print("censored", out["censored_benchmark_20disease"], flush=True)
    out["icu_physionet2012_decision_24h"] = icu()
    print("ICU", out["icu_physionet2012_decision_24h"], flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
