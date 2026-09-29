"""Reviewer-requested verifications and small sensitivity analyses.

1. Verification of specific claims against the data and the published outputs:
   a. CTO: identical labeling-function columns and the number of distinct signals per tier
      (5 start-time / 13 with during-trial / 22 all), plus a refit showing that dropping the
      duplicate columns leaves the published tier AUPRC/AUROC unchanged.
   b. Censored benchmark rolling-origin blocks (20 and 41 diseases), from the published JSON.
   c. Instrument validation: CI-above-zero and CI-below-zero rates at zero leakage.
   d. Target-disjointness of the 20-disease main test block (eligible, dropped, clusters, positives).
   e. Label maturity in the 20-disease test block (decision years >= 2022, labels, max info_time).
   f. TrialBench: provided/temporal test-set overlap; which difference CIs exclude zero.
   g. LLM study: named-intervention gain in the approved-after-start and never-approved strata;
      unadjusted identifier-recall CI for Opus 5.
2. Mature-label sensitivity (20-disease censored benchmark): the published main-split
   leakage-response curve (leakage.leakage_response_curve; default learner; target-clustered
   paired bootstrap, 500 resamples) re-evaluated after excluding test programs whose decision
   year (info_time) is too recent for advancement to be observed (keep info_time <= 2021, <= 2020).
   Training block unchanged. The unrestricted run is checked against the published output.
3. TrialBench with the test set held fixed. (i) As specified: the temporal test set; train on
   (a) the temporal (past-only) training set and (b) the provided training pool minus the test
   trials. By construction (the temporal test set is the latest block by NCT number) that pool is a
   subset of the temporal training set, so this design cannot contain trials that start after the
   test trials; it is reported with this caveat, with (a) also size-matched to (b). (ii) Variant
   that realises the intended contrast: a fixed middle block (NCT ranks centred, same size as the
   provided test set); train on (a) all earlier trials (past-only) versus (b) a same-size random
   sample of all non-test trials (earlier and later; a non-temporal mix) and (c) a same-size sample
   of later trials only. Paired trial-level bootstrap (1,000 resamples) on the fixed test set.
4. LLM study: the pre-specified primary contrasts (D+ID - D; D+T - D+Tm) with a bootstrap that
   resamples the 25-trial prompt batches (24 clusters) instead of trials, from the saved
   predictions (no model is queried). Holm adjustment via the study's own primary_tests/holm.

Existing functions are reused unchanged (audit_cto, audit_trialbench, leakage, splits,
llm_memorization_study). Nothing in outputs/ other than the new JSON is written.

Usage:  python scripts/sensitivity_reviewer.py [--out outputs/sensitivity_reviewer.json]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import audit_cto  # noqa: E402
import audit_trialbench as TB  # noqa: E402
import llm_memorization_study as L  # noqa: E402
from temporal_leakage_audit import features, leakage, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402

BENCH20 = "config/benchmark_20disease.yaml"
MATURE_MAX_YEARS = (2021, 2020)
TB_N_BOOT = 1000
TB_MIX_SEEDS = range(10)          # draw-to-draw variability of the random training mix
PROTECTED = ("censored_benchmark_20disease.json", "censored_benchmark_41disease.json",
             "cto_audit.json", "dating_rule_sensitivity.json", "instrument_validation.json",
             "llm_knowledge_dating.json", "llm_memorization_study.json", "llm_predictions.csv",
             "trialbench_audit.json", "robustness.json")


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


def _q(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [_r(np.quantile(a, .025)), _r(np.quantile(a, .975))] if len(a) else [None, None]


def _excludes_zero(ci):
    return bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))


def _diff(ref, new, path=""):
    """Leaves where ``new`` differs from ``ref`` (exact equality)."""
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return [f"{path}: not a dict"]
        return [m for k in ref for m in (_diff(ref[k], new[k], f"{path}/{k}") if k in new
                                         else [f"{path}/{k}: missing"])]
    if isinstance(ref, list):
        if not isinstance(new, list) or len(ref) != len(new):
            return [f"{path}: {new!r} != {ref!r}"]
        return [m for i, (a, b) in enumerate(zip(ref, new)) for m in _diff(a, b, f"{path}[{i}]")]
    return [] if ref == new else [f"{path}: reproduced {new!r} != published {ref!r}"]


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _load(path):
    with open(os.path.join(ROOT, path)) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- 1a CTO
def verify_cto():
    j = audit_cto.load("data/cto")
    tiers = {"start_time": audit_cto.START,
             "with_during_trial": audit_cto.START + audit_cto.DURING,
             "all": audit_cto.START + audit_cto.DURING + audit_cto.POST}

    def groups_of(cols, exact=True):
        g = collections.OrderedDict()
        for c in cols:
            s = j[c].astype(float).fillna(-1e9)
            k = tuple(s.values) if exact else tuple(s.rank(method="dense").values)
            g.setdefault(k, []).append(c)
        return list(g.values())

    raw_identity = {}
    pairs = [("hint_train", "hint_train2"), ("hint_train", "hint_train3"), ("linkage", "linkage2"),
             ("status", "status2"), ("gpt", "gpt2")]
    for f in ("phase1_CTO_rf.csv", "phase2_CTO_rf.csv", "phase3_CTO_rf.csv"):
        d = pd.read_csv(os.path.join(ROOT, "data/cto", f))
        raw_identity[f] = {"n_rows": int(len(d)),
                           **{f"{a}=={b}": bool(d[a].fillna(-9).equals(d[b].fillna(-9)))
                              for a, b in pairs}}

    per_tier = {}
    for name, cols in tiers.items():
        g = groups_of(cols)
        per_tier[name] = {"n_columns": len(cols), "n_distinct_exact": len(g),
                          "n_distinct_up_to_monotone_transform": len(groups_of(cols, exact=False)),
                          "identical_groups": [x for x in g if len(x) > 1]}

    # Refit each tier on de-duplicated columns (first of each identical group); same split,
    # learner and seed as audit_cto.main().
    cut = int(j["start_year"].quantile(0.6))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    yte = te["labels"].values
    pub = _load("outputs/cto_audit.json")["tiers"]
    refit = {}
    for (name, cols), key in zip(tiers.items(), ("h0_start", "h1_during", "h2_post_naive")):
        dedup = [x[0] for x in groups_of(cols)]
        p_full = audit_cto._fit_predict(tr, te, cols)
        p_dedup = audit_cto._fit_predict(tr, te, dedup)
        refit[name] = {"n_columns_dedup": len(dedup), "dedup_columns": dedup,
                       "full_auprc": _r(audit_cto._auprc(yte, p_full)),
                       "full_auroc": _r(audit_cto._auroc(yte, p_full)),
                       "dedup_auprc": _r(audit_cto._auprc(yte, p_dedup)),
                       "dedup_auroc": _r(audit_cto._auroc(yte, p_dedup)),
                       "max_abs_pred_diff": float(np.max(np.abs(p_full - p_dedup))),
                       "published_auprc": pub[key]["auprc"], "published_auroc": pub[key]["auroc"]}
    return {"n_trials": int(len(j)), "raw_file_identity": raw_identity, "per_tier": per_tier,
            "refit_dedup_same_split": refit,
            "note": ("LF values are ternary (-1/0/1; status and num_sponsors binary). num_sponsors "
                     "exists only in the phase-2 table, so it is missing for "
                     f"{_r(j['num_sponsors'].isna().mean())} of the audited trials.")}


# --------------------------------------------------------------------------- 1b, 1c
def verify_censored_json():
    out = {}
    for tag in ("20disease", "41disease"):
        d = _load(f"outputs/censored_benchmark_{tag}.json")
        lrc = d["leakage_response_curve"]
        blocks = [{"cutoff": r["cutoff"], "n_test": r["n_test"], "test_pos": r["test_pos"],
                   "test_base_rate": r["test_base_rate"], "naive_auprc": r["naive_auprc"],
                   "censored_auprc": r["censored_auprc"], "LAP": r["LAP_auprc"],
                   "LAP_ci": [r["LAP_ci"]["lo"], r["LAP_ci"]["hi"]],
                   "n_valid_boot": r["LAP_ci"]["n"], "p_gt_0": r["LAP_p_gt_0"],
                   "ci_excludes_zero": _excludes_zero([r["LAP_ci"]["lo"], r["LAP_ci"]["hi"]])}
                  for r in d["rolling_origin_LAP"]]
        out[tag] = {"data_dir": d["data_dir"], "n_programs": d["dataset"]["n"],
                    "main_split": {**d["main_split"], "LAP": lrc["total_LAP_auprc"],
                                   "LAP_ci": lrc["total_LAP_auprc_ci"],
                                   "test_base_rate": lrc["test_base_rate"]},
                    "rolling_origin_blocks": blocks,
                    "n_blocks_ci_excludes_zero": sum(b["ci_excludes_zero"] for b in blocks)}
    out["note"] = ("The 41-disease benchmark (data/benchmark_41disease_otyear) uses Open Targets "
                   "publication-year evidence dating, not the PubMed-year rule of the 20-disease set.")
    return out


def verify_instrument():
    d = _load("outputs/instrument_validation.json")
    lv0 = next(x for x in d["levels"] if x["leak_strength"] == 0.0)
    cn0 = next(x for x in d["count_levels"] if x["leak_count"] == 0)
    f = lambda x: {"replicates": x["replicates"], "ci_above_zero_rate": x["detection_rate"],  # noqa: E731
                   "ci_below_zero_rate": x["ci_below_zero_rate"],
                   "two_sided_rejection_rate": _r(x["detection_rate"] + x["ci_below_zero_rate"], 3),
                   "mean_lap": x["mean_lap"]}
    return {"score_mechanism_leak_strength_0": f(lv0), "count_mechanism_leak_count_0": f(cn0),
            "bootstrap": d["design"]["bootstrap"]}


# --------------------------------------------------------------------------- benchmark
def load_bench20():
    cfg = get_config(BENCH20)
    programs = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "programs.csv"))
    evidence = pd.read_csv(os.path.join(ROOT, cfg["data_dir"], "evidence.csv"))
    _, _, y, meta, _ = features.build(programs, evidence)
    yrs = programs["info_time"]
    tmax, hi = int(yrs.quantile(0.60)), int(yrs.max())
    tr, te, info = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    return {"cfg": cfg, "programs": programs, "evidence": evidence, "y": y, "meta": meta,
            "tr": tr, "te": te, "info": info, "tmax": tmax, "hi": hi}


def verify_test_block(b):
    meta, y, te, tmax = b["meta"], b["y"], b["te"], b["tmax"]
    eligible = meta.index[(meta["info_time"] > tmax) & (meta["info_time"] <= b["hi"])]
    t = pd.DataFrame({"yr": meta.loc[te, "info_time"], "y": y.loc[te]})
    allp = pd.DataFrame({"yr": meta["info_time"], "y": y})
    recent = t[t["yr"] >= 2022]
    return {
        "d_target_disjointness": {
            "train_max_year": tmax, "n_eligible_post_cutoff": int(len(eligible)),
            "n_dropped_shared_target": b["info"]["test_dropped_shared_target"],
            "n_test": int(len(te)), "n_target_clusters_test": int(meta.loc[te, "target_id"].nunique()),
            "n_positives_test": int(y.loc[te].sum()), "test_base_rate": _r(y.loc[te].mean())},
        "e_label_maturity": {
            "n_test_decision_year_ge_2022": int(len(recent)),
            "n_positive_among_them": int(recent["y"].sum()),
            "fraction_labelled_1": _r(recent["y"].mean()) if len(recent) else None,
            "max_info_time_test_block": int(t["yr"].max()),
            "max_info_time_dataset": int(allp["yr"].max()),
            "test_block_by_year": {int(k): {"n": int(v["size"]), "pos": int(v["sum"])}
                                   for k, v in t.groupby("yr")["y"].agg(["size", "sum"]).iterrows()},
            "all_programs_label_rate_by_year_2018_on": {
                int(k): {"n": int(v["size"]), "rate": _r(v["mean"])}
                for k, v in allp[allp["yr"] >= 2018].groupby("yr")["y"].agg(["size", "mean"]).iterrows()}},
    }


def mature_label_sensitivity(b):
    """Task 2. leakage_response_curve with the test block restricted to mature decision years."""
    cfg, meta, y = b["cfg"], b["meta"], b["y"]
    nb, bf = cfg["bootstrap"]["n_boot"], cfg["decision"]["budget_frac"]

    def run(te):
        return leakage.leakage_response_curve(b["programs"], b["evidence"], b["tr"], te,
                                              clusters=meta.loc[te, "target_id"].values,
                                              n_boot=nb, budget_frac=bf)

    def summ(te, lrc):
        c = {x["horizon"]: x for x in lrc["curve"]}
        return {"n_test": int(len(te)), "n_pos": int(y.loc[te].sum()),
                "n_target_clusters": int(meta.loc[te, "target_id"].nunique()),
                "test_base_rate": lrc["test_base_rate"],
                "h0_auprc": lrc["deployable_auprc"], "h0_auprc_ci": c[0].get("auprc_ci"),
                "hinf_auprc": lrc["naive_auprc"], "hinf_auprc_ci": c[None].get("auprc_ci"),
                "LAP": lrc["total_LAP_auprc"], "LAP_ci": lrc["total_LAP_auprc_ci"],
                "LAP_p_gt_0": lrc["total_LAP_auprc_p_gt_0"], "peak_LAP": lrc["peak_LAP_auprc"],
                "peak_horizon": lrc["peak_horizon"],
                "curve_auprc": {("inf" if x["horizon"] is None else str(x["horizon"])): x["auprc_mean"]
                                for x in lrc["curve"]}}

    full = run(b["te"])
    pub = _load("outputs/censored_benchmark_20disease.json")
    mism = _diff(pub["leakage_response_curve"], _jsonable(full), "leakage_response_curve") + \
        _diff(pub["main_split"], _jsonable(b["info"]), "main_split")
    out = {"settings": {"train": f"info_time <= {b['tmax']} (unchanged)",
                        "learner": "HistGradientBoosting default (models.fit_gbdt, seed 0)",
                        "bootstrap": f"target-clustered paired percentile, {nb} resamples, seed 0"},
           "reproduction_check": {"reference": "outputs/censored_benchmark_20disease.json",
                                  "exact_match": not mism, "mismatches": mism},
           "all_test_years": summ(b["te"], full)}
    for ymax in MATURE_MAX_YEARS:
        te = b["te"][meta.loc[b["te"], "info_time"].values <= ymax]
        out[f"decision_year_le_{ymax}"] = summ(te, run(te))
    return out


# --------------------------------------------------------------------------- 1f, 3 TrialBench
def verify_trialbench_json():
    d = _load("outputs/trialbench_audit.json")
    rows = {}
    for p in d["phases"]:
        dd = p["provided_minus_temporal"]
        rows[p["phase"]] = {"overlap": p["overlap_provided_temporal_test"], "n_test": p["n_test"],
                            **{f"{k}_diff": dd[k]["point"] for k in ("auprc", "auroc", "auprc_lift")},
                            **{f"{k}_diff_ci": dd[k]["ci"] for k in ("auprc", "auroc", "auprc_lift")},
                            **{f"{k}_ci_excludes_zero": _excludes_zero(dd[k]["ci"])
                               for k in ("auprc", "auroc", "auprc_lift")}}
    return {"per_phase": rows, "sign_convention": "provided minus temporal",
            "phases_auroc_ci_excludes_zero": [k for k, v in rows.items() if v["auroc_ci_excludes_zero"]],
            "phases_auprc_ci_excludes_zero": [k for k, v in rows.items() if v["auprc_ci_excludes_zero"]]}


def _tb_load(ph):
    """Same rows, features and splits as audit_trialbench.audit_phase()."""
    xtr = pd.read_csv(f"{TB.X_BASE}/{ph}/train_x.csv", low_memory=False)
    xte = pd.read_csv(f"{TB.X_BASE}/{ph}/test_x.csv", low_memory=False)
    ytr = pd.read_csv(f"{TB.Y_BASE}/{ph}/train_y.csv")
    yte = pd.read_csv(f"{TB.Y_BASE}/{ph}/test_y.csv")
    xtr = xtr.merge(ytr, on="Unnamed: 0", how="inner")
    xte = xte.merge(yte, on="Unnamed: 0", how="inner")
    allrows = pd.concat([xtr, xte], ignore_index=True)
    y = allrows["outcome"].astype(int).values
    X, ids = TB._featurize(allrows.drop(columns=["outcome"]))
    n, n_test = len(allrows), len(xte)
    prov_test = np.zeros(n, bool)
    prov_test[len(xtr):] = True
    nct = np.array([TB._nct_num(i) for i in ids])
    order = np.argsort(nct)
    temp_test = np.zeros(n, bool)
    temp_test[order[-n_test:]] = True
    return X, y, ids, nct, order, prov_test, temp_test


def _paired_boot(y, preds, n_boot=TB_N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: {"auprc": [], "auroc": []} for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        for k, p in preds.items():
            d[k]["auprc"].append(TB._auprc(y[idx], p[idx]))
            d[k]["auroc"].append(TB._auroc(y[idx], p[idx]))
    return {k: {m: np.array(v) for m, v in dd.items()} for k, dd in d.items()}


def _arm(y, p, draws, ytr):
    return {"n_train": int(len(ytr)), "train_base_rate": _r(ytr.mean()),
            "auprc": _r(TB._auprc(y, p)), "auprc_ci": _q(draws["auprc"]),
            "auroc": _r(TB._auroc(y, p)), "auroc_ci": _q(draws["auroc"])}


def _contrast(y, pa, pb, da, db):
    out = {}
    for m, f in (("auprc", TB._auprc), ("auroc", TB._auroc)):
        dd = db[m] - da[m]
        out[m] = {"point": _r(f(y, pb) - f(y, pa)), "ci": _q(dd), "p_gt_0": _r(np.mean(dd > 0), 3)}
    return out


def trialbench_fixed_test():
    TB.ensure_data()  # no-op when present
    sd = pd.read_csv(TB.START_DATES_CSV)  # cached ClinicalTrials.gov start years (read only)
    yrmap = dict(zip(sd["nct_id"], sd["start_year"]))
    pub = {p["phase"]: p for p in _load("outputs/trialbench_audit.json")["phases"]}
    res, repro = {}, {}
    for ph in TB.PHASES:
        X, y, ids, nct, order, prov_test, temp_test = _tb_load(ph)
        yr = np.array([yrmap.get(str(i), np.nan) for i in ids], float)
        n, n_test = len(y), int(temp_test.sum())

        # (i) As specified: fixed temporal test set.
        te = np.where(temp_test)[0]
        a_full = np.where(~temp_test)[0]
        pool = np.where(~prov_test & ~temp_test)[0]
        rng = np.random.default_rng(0)
        a_sub = np.sort(rng.choice(a_full, size=len(pool), replace=False))
        yte = y[te]
        P = {k: TB._fit(X.iloc[idx], y[idx], X.iloc[te]) for k, idx in
             (("a_temporal_full", a_full), ("a_temporal_size_matched", a_sub), ("b_provided_pool", pool))}
        ev = TB._evaluate(yte, P["a_temporal_full"], 0)
        ev["train_base_rate"] = _r(y[a_full].mean())
        mism = _diff(pub[ph]["temporal_split"], _jsonable(ev), f"{ph}/temporal_split")
        repro[ph] = {"exact_match": not mism, "mismatches": mism}
        D = _paired_boot(yte, P)
        spec = {
            "test_set": "temporal test set (latest n_test by NCT number)",
            "n_test": n_test, "test_base_rate": _r(yte.mean()),
            "pool_is_subset_of_temporal_train": bool(np.isin(pool, a_full).all()),
            "pool_trials_with_nct_after_earliest_test_nct": int((nct[pool] > nct[te].min()).sum()),
            "pool_frac_start_after_test_median_start": _r(np.nanmean(yr[pool] > np.nanmedian(yr[te]))),
            "test_median_start_year": _r(np.nanmedian(yr[te]), 1),
            "arms": {k: _arm(yte, P[k], D[k], y[idx]) for k, idx in
                     (("a_temporal_full", a_full), ("a_temporal_size_matched", a_sub),
                      ("b_provided_pool", pool))},
            "b_minus_a_size_matched": _contrast(yte, P["a_temporal_size_matched"], P["b_provided_pool"],
                                                D["a_temporal_size_matched"], D["b_provided_pool"]),
            "b_minus_a_full": _contrast(yte, P["a_temporal_full"], P["b_provided_pool"],
                                        D["a_temporal_full"], D["b_provided_pool"]),
        }

        # (ii) Variant: fixed middle block; past-only vs same-size non-temporal mix vs future-only.
        lo = (n - n_test) // 2
        te2 = np.sort(order[lo:lo + n_test])
        past, future = np.sort(order[:lo]), np.sort(order[lo + n_test:])
        nonte = np.concatenate([past, future])
        yte2 = y[te2]
        rng = np.random.default_rng(0)
        mix = np.sort(rng.choice(nonte, size=len(past), replace=False))
        fut = future if len(future) == len(past) else \
            np.sort(np.random.default_rng(0).choice(future, size=len(past), replace=False))
        P2 = {k: TB._fit(X.iloc[idx], y[idx], X.iloc[te2]) for k, idx in
              (("a_past_only", past), ("b_mix_past_future", mix), ("c_future_only", fut))}
        D2 = _paired_boot(yte2, P2)
        draws = []
        for s in TB_MIX_SEEDS:
            m = np.sort(np.random.default_rng(s).choice(nonte, size=len(past), replace=False))
            pm = P2["b_mix_past_future"] if s == 0 else TB._fit(X.iloc[m], y[m], X.iloc[te2])
            draws.append({"seed": s, "auprc_diff": TB._auprc(yte2, pm) - TB._auprc(yte2, P2["a_past_only"]),
                          "auroc_diff": TB._auroc(yte2, pm) - TB._auroc(yte2, P2["a_past_only"]),
                          "frac_future": float(np.isin(m, future).mean())})
        dd = pd.DataFrame(draws)
        variant = {
            "test_set": f"NCT ranks {lo}..{lo + n_test - 1} of {n} (middle block, size of provided test set)",
            "n_test": n_test, "test_base_rate": _r(yte2.mean()),
            "median_start_year": {"past": _r(np.nanmedian(yr[past]), 1), "test": _r(np.nanmedian(yr[te2]), 1),
                                  "future": _r(np.nanmedian(yr[future]), 1)},
            "frac_future_in_mix_seed0": _r(np.isin(mix, future).mean()),
            "arms": {k: _arm(yte2, P2[k], D2[k], y[idx]) for k, idx in
                     (("a_past_only", past), ("b_mix_past_future", mix), ("c_future_only", fut))},
            "b_mix_minus_a_past": _contrast(yte2, P2["a_past_only"], P2["b_mix_past_future"],
                                            D2["a_past_only"], D2["b_mix_past_future"]),
            "c_future_minus_a_past": _contrast(yte2, P2["a_past_only"], P2["c_future_only"],
                                               D2["a_past_only"], D2["c_future_only"]),
            "mix_draw_variability_10_seeds": {
                "auprc_diff_mean": _r(dd["auprc_diff"].mean()), "auprc_diff_range": [_r(dd["auprc_diff"].min()), _r(dd["auprc_diff"].max())],
                "auroc_diff_mean": _r(dd["auroc_diff"].mean()), "auroc_diff_range": [_r(dd["auroc_diff"].min()), _r(dd["auroc_diff"].max())]},
        }
        res[ph] = {"as_specified_fixed_temporal_test": spec, "variant_fixed_middle_block": variant}
        print(f"  TrialBench {ph} done", flush=True)
    return {"settings": {"learner": "audit_trialbench._fit (HGB default, seed 0)",
                         "bootstrap": f"trial-level paired percentile on the fixed test set, {TB_N_BOOT} resamples, seed 0",
                         "contrast_sign": "second arm minus first arm (b - a, c - a)",
                         "time_order": "NCT number (as in the audit); start years from the cached CT.gov table for context"},
            "reproduction_check_temporal_split": repro, "phases": res}


# --------------------------------------------------------------------------- 1g, 4 LLM
def verify_llm_json():
    kd = _load("outputs/llm_knowledge_dating.json")
    ms = _load("outputs/llm_memorization_study.json")
    strata = {}
    for m, r in kd["models"].items():
        if not m.startswith("claude"):
            continue
        strata[m] = {s: {k: r[s]["D+T - D+Tm"].get(k) for k in
                         ("n", "n_pos", "auprc", "auprc_ci", "auroc", "auroc_ci")}
                     for s in ("approved_after_start", "never_approved_code")}
    pt = next(x for x in ms["primary_tests"] if x["model"] == "claude-opus-5" and x["contrast"] == "D+ID - D")
    return {"named_intervention_gain_by_stratum": strata, "strata_counts": kd["strata_counts"],
            "opus5_identifier_recall": {**ms["models"]["claude-opus-5"]["all"]["contrasts"]["D+ID - D"],
                                        "p_two_sided": pt["p_two_sided"], "p_holm": pt["p_holm"]}}


def _batch_map():
    """nct_id -> prompt batch (start index), from the saved raw responses; checked for consistency."""
    m = collections.defaultdict(set)
    for model in L.MODELS:
        for (_cond, b), rec in L._done(model).items():
            for nct in rec["nct_ids"]:
                m[nct].add(b)
    bad = {k: v for k, v in m.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"trials in more than one batch: {list(bad)[:5]}")
    return {k: next(iter(v)) for k, v in m.items()}


def _batch_summary(w, batch_of, seed=L.SEED, n_boot=L.N_BOOT):
    """Primary contrasts with a bootstrap over prompt batches (mirrors L.summarize otherwise)."""
    w = w.dropna(subset=L.CONDITIONS)
    y = w["label"].values.astype(int)
    b = w["nct_id"].map(batch_of).values
    P = {c: w[c].values for c in L.CONDITIONS}
    uniq = np.unique(b)
    rows_of = {u: np.where(b == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    need = sorted({c for k in L.PRIMARY for c in k.split(" - ")})
    boot = {c: [] for c in need}
    boot_roc = {c: [] for c in need}
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(set(y[rows])) < 2:
            continue
        for c in need:
            boot[c].append(L._auprc(y[rows], P[c][rows]))
            boot_roc[c].append(L._auroc(y[rows], P[c][rows]))
    boot = {c: np.array(v) for c, v in boot.items()}
    boot_roc = {c: np.array(v) for c, v in boot_roc.items()}
    con = {}
    for key in L.PRIMARY:
        a, bb = key.split(" - ")
        d, dr = boot[a] - boot[bb], boot_roc[a] - boot_roc[bb]
        con[key] = {"auprc": round(L._auprc(y, P[a]) - L._auprc(y, P[bb]), 4), "auprc_ci": L._q(d),
                    "auprc_p_gt_0": round(float(np.mean(d > 0)), 3),
                    "auroc": round(L._auroc(y, P[a]) - L._auroc(y, P[bb]), 4), "auroc_ci": L._q(dr)}
    return {"n": int(len(w)), "n_batches": int(len(uniq)), "n_valid_boot": int(len(boot[need[0]])),
            "contrasts": con}


def llm_batch_bootstrap():
    preds = pd.read_csv(os.path.join(ROOT, L.PRED_PATH))
    batch_of = _batch_map()
    pub = _load(L.OUT_PATH)
    trial_out, batch_out, repro = {}, {}, {}
    for model in [m for m in L.MODELS if m in set(preds["model"])]:
        wide = preds[preds["model"] == model].pivot(
            index=["nct_id", "label", "completion_date"], columns="condition", values="p").reset_index()
        s = L.summarize(wide)
        trial_out[model] = {"all": s}
        mism = _diff(pub["models"][model]["all"], _jsonable(s), f"{model}/all")
        repro[model] = {"exact_match": not mism, "mismatches": mism[:5]}
        batch_out[model] = {"all": _batch_summary(wide, batch_of)}
    t_tests = L.primary_tests(trial_out)
    b_tests = L.primary_tests(batch_out)
    mism = _diff(pub["primary_tests"], _jsonable(t_tests), "primary_tests")
    repro["primary_tests"] = {"exact_match": not mism, "mismatches": mism[:5]}
    rows = []
    for t, bt in zip(t_tests, b_tests):
        assert (t["model"], t["contrast"]) == (bt["model"], bt["contrast"])
        c_t = trial_out[t["model"]]["all"]["contrasts"][t["contrast"]]
        c_b = batch_out[t["model"]]["all"]["contrasts"][t["contrast"]]
        rows.append({"model": t["model"], "contrast": t["contrast"], "auprc": t["auprc"],
                     "trial_auprc_ci": c_t["auprc_ci"], "batch_auprc_ci": c_b["auprc_ci"],
                     "auroc": c_t["auroc"], "trial_auroc_ci": c_t["auroc_ci"], "batch_auroc_ci": c_b["auroc_ci"],
                     "trial_p_two_sided": t["p_two_sided"], "batch_p_two_sided": bt["p_two_sided"],
                     "trial_p_holm": t["p_holm"], "batch_p_holm": bt["p_holm"],
                     "trial_sig_holm": t["significant_holm_0.05"], "batch_sig_holm": bt["significant_holm_0.05"],
                     "holm_conclusion_changes": t["significant_holm_0.05"] != bt["significant_holm_0.05"]})
    return {"settings": {"clusters": "prompt batches of 25 trials (batch start index; same 24 batches in "
                                     "every condition and model)",
                         "n_boot": L.N_BOOT, "seed": L.SEED,
                         "p_value": "two-sided bootstrap, floor 1/1000; Holm across all 10 primary tests "
                                    "(L.primary_tests / L.holm, unchanged)"},
            "reproduction_check_trial_level": repro, "primary_contrasts": rows,
            "n_holm_conclusions_changed": int(sum(r["holm_conclusion_changes"] for r in rows))}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Reviewer-requested verifications and sensitivity analyses.")
    ap.add_argument("--out", default="outputs/sensitivity_reviewer.json")
    args = ap.parse_args()
    out_path = os.path.abspath(args.out)
    if os.path.basename(out_path) in PROTECTED:
        raise SystemExit(f"refusing to overwrite {out_path}")
    os.chdir(ROOT)
    t0, rt = time.time(), {}

    def timed(label, fn, *a):
        t = time.time()
        r = fn(*a)
        rt[label] = round(time.time() - t, 1)
        print(f"[{label}] {rt[label]}s", flush=True)
        return r

    ver = {"a_cto_labeling_functions": timed("1a", verify_cto),
           "b_censored_benchmark_rolling_origin": timed("1b", verify_censored_json),
           "c_instrument_validation_zero_leakage": timed("1c", verify_instrument)}
    bench = timed("load_bench20", load_bench20)
    ver.update(timed("1de", verify_test_block, bench))
    ver["f_trialbench"] = timed("1f", verify_trialbench_json)
    ver["g_llm"] = timed("1g", verify_llm_json)
    mature = timed("2", mature_label_sensitivity, bench)
    tbfix = timed("3", trialbench_fixed_test)
    llmb = timed("4", llm_batch_bootstrap)

    out = {"description": __doc__.split("\n\n")[0],
           "verifications": ver,
           "mature_label_sensitivity_20disease": mature,
           "trialbench_fixed_test": tbfix,
           "llm_batch_clustered_bootstrap": llmb,
           "reproduces_published_defaults": {
               "censored_benchmark_20disease_main_curve": mature["reproduction_check"]["exact_match"],
               "trialbench_temporal_split": all(v["exact_match"] for v in
                                                tbfix["reproduction_check_temporal_split"].values()),
               "llm_trial_level_contrasts_and_primary_tests": all(
                   v["exact_match"] for v in llmb["reproduction_check_trial_level"].values()),
               "cto_tier_point_estimates": all(
                   v["full_auprc"] == v["published_auprc"] and v["full_auroc"] == v["published_auroc"]
                   for v in ver["a_cto_labeling_functions"]["refit_dedup_same_split"].values())},
           "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
           "runtime_seconds": {**rt, "total": round(time.time() - t0, 1)}}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out["reproduces_published_defaults"]))
    print(f"wrote {out_path} ({out['runtime_seconds']['total']}s)")


if __name__ == "__main__":
    main()
