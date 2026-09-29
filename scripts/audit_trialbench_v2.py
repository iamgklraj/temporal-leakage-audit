"""TrialBench split audit, v2 (reviewer response): feature timing, ablation, joint bootstrap,
fixed-test-set design.

Reuses scripts/audit_trialbench.py unchanged (ensure_data, _featurize, _fit, _evaluate, _nct_num,
_auprc, _auroc and the paths). The learner, featurisation, provided split and NCT-ordered temporal
split are the same as in the original.

 T1  Classify every feature column of the approval-forecasting tables by when it becomes public,
     and flag those the model uses that are not start-time. Evidence: ClinicalTrials.gov field
     semantics (42 CFR 11.10(b)(18), 11.64), the TrialBench paper (arXiv:2407.00631), and the
     current ClinicalTrials.gov record (enrollment type, overall status, location count), fetched
     once and cached to data/trialbench/ctgov_enrollment_status_v2.csv. Predictiveness of
     'enrollment' alone per phase.
 T2  Provided vs temporal split per phase, with (a) all features, (b) the non-start-time features
     removed and (c) only 'enrollment' removed. Difference CIs come from a JOINT bootstrap that
     resamples trial IDs from the union of the two test sets, so overlapping trials are shared.
 T3  Fixed test set, with the ablated features and with all features:
     (i)   temporal test set; train on the past-only (temporal) training set vs the provided
           training pool minus test trials (and a size-matched past-only subsample). The same
           design as scripts/sensitivity_reviewer.py; see its caveat that the pool lies entirely
           in the past.
     (ii)  fixed middle block by NCT rank; past-only vs a same-size random non-test mix vs
           future-only (same design as sensitivity_reviewer.py).
     (iii) temporal test set, 2-fold cross-fitted: for each half, past-only training vs a
           same-size random sample of (past + the other half of the test era). Predictions cover
           the whole temporal test set.
     All arms: paired trial-level bootstrap (1,000 resamples) of the AUROC/AUPRC difference.
 T4  Test prevalence and approval rate by start-year quartile, using the cached start dates
     (read only).

Usage:  python scripts/audit_trialbench_v2.py [--out outputs/trialbench_audit_v2.json] [--no-fetch]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)  # audit_trialbench uses repo-relative data paths

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import audit_trialbench as T  # noqa: E402

N_BOOT = 1000
SEED = 0
MIX_SEEDS = (0, 1, 2, 3, 4)
CTGOV_V2_CSV = f"{T.DATA_DIR}/ctgov_enrollment_status_v2.csv"
CITY = ["location/facility/address/city-Aging", "location/facility/address/city-GDP",
        "location/facility/address/city-Population"]
NON_START = ["enrollment"] + CITY

# --------------------------------------------------------------------------- #
# T1: timing classification of every raw feature column
# --------------------------------------------------------------------------- #
DESIGN = ("start-time (protocol design element entered at registration; can be amended but "
          "describes the planned design)")
RAW_TIMING = {
    "Unnamed: 0": ("id", "NCT identifier (not a feature)"),
    "enrollment": ("NOT start-time: final (actual) enrollment",
                   "42 CFR 11.10(b)(18): 'Enrollment means the estimated total number of human subjects "
                   "to be enrolled (target number) or the actual total number ... Once the trial has "
                   "reached the primary completion date, the responsible party must update the "
                   "Enrollment data element to reflect the actual number of human subjects enrolled'. "
                   "TrialBench parsed the XML records of trials registered before 16 Feb 2024, so a "
                   "completed or terminated trial carries its actual enrollment. The value matches the "
                   "current ACTUAL count for most trials (see ctgov_evidence). A terminated trial's "
                   "actual enrollment reflects early stopping."),
    "location/facility/address/city": ("NOT start-time: final-record facility list (text; not used by the model)",
                                       "Facility list is updated during recruitment (42 CFR 11.64 individual site status)."),
    "location/facility/address/city-Aging": ("NOT start-time: covariate of a final-record facility city",
                                             "City covariate joined to the facility list of the final record. It is "
                                             "missing (filled with 0 by _featurize) when no facility is listed, "
                                             "which is common for withdrawn trials."),
    "location/facility/address/city-GDP": ("NOT start-time: covariate of a final-record facility city", "as city-Aging"),
    "location/facility/address/city-Population": ("NOT start-time: covariate of a final-record facility city",
                                                  "as city-Aging"),
    "ipd_info_type-Analytic Code": ("ambiguous: IPD-sharing plan (not used: >50% missing)", "Plan-to-share-IPD fields can be updated at any time."),
    "ipd_info_type-Clinical Study Report (CSR)": ("ambiguous: IPD-sharing plan (not used: >50% missing)", "as above"),
    "ipd_info_type-Informed Consent Form (ICF)": ("ambiguous: IPD-sharing plan (not used: >50% missing)", "as above"),
    "ipd_info_type-Statistical Analysis Plan (SAP)": ("ambiguous: IPD-sharing plan (not used: >50% missing)", "as above"),
    "ipd_info_type-Study Protocol": ("ambiguous: IPD-sharing plan (not used: >50% missing)", "as above"),
    "patient_data/sharing_ipd": ("ambiguous: IPD-sharing plan (text; not used by the model)", "as above"),
    "has_expanded_access": ("start-time (not used by the model)", "registration element"),
    "oversight_info/has_dmc": ("start-time (not used by the model)", "registration element"),
    "oversight_info/is_fda_regulated_device": ("start-time (not used by the model)", "registration element"),
    "oversight_info/is_fda_regulated_drug": ("start-time (not used by the model)", "registration element"),
    "responsible_party/responsible_party_type": ("start-time (not used by the model)", "registration element"),
}


def raw_timing(col):
    if col in RAW_TIMING:
        return RAW_TIMING[col]
    if col.endswith("Arm Number") or col.endswith("intervention Number") or col.startswith("MaskingType") \
            or col in ("number_of_arms", "study_design_info/masking_num", "study_design_info/allocation",
                       "study_design_info/intervention_model", "study_design_info/primary_purpose",
                       "study_type", "phase", "eligibility/gender", "eligibility/healthy_volunteers",
                       "eligibility/minimum_age", "eligibility/maximum_age",
                       "eligibility/criteria/textblock", "sponsors/lead_sponsor/agency_class"):
        return (DESIGN, "Arms/interventions, masking, allocation, model, purpose, eligibility and "
                        "lead-sponsor class are protocol registration data elements (42 CFR 11.10(b)).")
    return ("start-time descriptive text/code (not used by the numeric model)",
            "title/summary/condition/MeSH/ICD/intervention names/SMILES/keywords")


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov evidence (cached)
# --------------------------------------------------------------------------- #

def fetch_ctgov(ncts, batch=200):
    """NCT -> overall status, enrollment count/type, #locations (API v2; cached, new file)."""
    cache = pd.read_csv(CTGOV_V2_CSV) if os.path.exists(CTGOV_V2_CSV) else pd.DataFrame(
        columns=["nct_id", "overall_status", "enrollment_count", "enrollment_type", "n_locations"])
    have = set(cache["nct_id"])
    todo = [x for x in dict.fromkeys(ncts) if x not in have]
    rows = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        params = {"filter.ids": ",".join(chunk),
                  "fields": "NCTId,OverallStatus,EnrollmentCount,EnrollmentType,LocationCity",
                  "pageSize": 1000, "format": "json"}
        url = T.CTGOV_URL + "?" + urllib.parse.urlencode(params)
        studies = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    studies = json.load(r).get("studies", [])
                break
            except Exception:  # noqa: BLE001 -- transient network error: back off and retry
                time.sleep(2 ** attempt)
        if studies is None:
            raise RuntimeError(f"ClinicalTrials.gov request failed: {url[:120]}")
        got = set()
        for s in studies:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            ei = ps.get("designModule", {}).get("enrollmentInfo", {}) or {}
            locs = ps.get("contactsLocationsModule", {}).get("locations", []) or []
            rows.append({"nct_id": nct, "overall_status": ps.get("statusModule", {}).get("overallStatus"),
                         "enrollment_count": ei.get("count"), "enrollment_type": ei.get("type"),
                         "n_locations": len(locs)})
            got.add(nct)
        for x in chunk:
            if x not in got:
                rows.append({"nct_id": x, "overall_status": None, "enrollment_count": None,
                             "enrollment_type": None, "n_locations": None})
        time.sleep(0.2)
        if (i // batch) % 20 == 0:
            print(f"    fetched {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(CTGOV_V2_CSV, index=False)
    return cache.drop_duplicates("nct_id").set_index("nct_id")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def q(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [r4(np.quantile(a, .025)), r4(np.quantile(a, .975))]


def load_phase(ph):
    """Same rows, features and splits as audit_trialbench.audit_phase()."""
    xtr = pd.read_csv(f"{T.X_BASE}/{ph}/train_x.csv", low_memory=False)
    xte = pd.read_csv(f"{T.X_BASE}/{ph}/test_x.csv", low_memory=False)
    ytr = pd.read_csv(f"{T.Y_BASE}/{ph}/train_y.csv")
    yte = pd.read_csv(f"{T.Y_BASE}/{ph}/test_y.csv")
    xtr = xtr.merge(ytr, on="Unnamed: 0", how="inner")
    xte = xte.merge(yte, on="Unnamed: 0", how="inner")
    allrows = pd.concat([xtr, xte], ignore_index=True)
    y = allrows["outcome"].astype(int).values
    raw = allrows.drop(columns=["outcome"])
    X, ids = T._featurize(raw)
    n, n_test = len(allrows), len(xte)
    prov_test = np.zeros(n, bool)
    prov_test[len(xtr):] = True
    nct = np.array([T._nct_num(i) for i in ids])
    order = np.argsort(nct)
    temp_test = np.zeros(n, bool)
    temp_test[order[-n_test:]] = True
    return dict(raw=raw, X=X, y=y, ids=np.array([str(i) for i in ids]), nct=nct, order=order,
                prov_test=prov_test, temp_test=temp_test, n=n, n_test=n_test)


def feature_sets(X):
    full = list(X.columns)
    return {"all_features": full,
            "ablated_non_start_removed": [c for c in full if c not in NON_START],
            "minus_enrollment_only": [c for c in full if c != "enrollment"]}


def trial_boot(y, preds, n_boot=N_BOOT, seed=SEED):
    """Paired trial-level bootstrap draws (AUPRC, AUROC) for every key (same test set)."""
    rng = np.random.default_rng(seed)
    d = {k: {"auprc": [], "auroc": []} for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        for k, p in preds.items():
            d[k]["auprc"].append(T._auprc(y[idx], p[idx]))
            d[k]["auroc"].append(T._auroc(y[idx], p[idx]))
    return {k: {m: np.array(v) for m, v in dd.items()} for k, dd in d.items()}


def contrast(y, pa, pb, da, db):
    out = {}
    for m, f in (("auprc", T._auprc), ("auroc", T._auroc)):
        dd = db[m] - da[m]
        out[m] = {"point": r4(f(y, pb) - f(y, pa)), "ci": q(dd), "p_gt_0": r4(np.mean(dd > 0))}
    return out


def arm(y, p, d, ytr):
    return {"n_train": int(len(ytr)), "train_base_rate": r4(ytr.mean()),
            "auprc": r4(T._auprc(y, p)), "auprc_ci": q(d["auprc"]),
            "auroc": r4(T._auroc(y, p)), "auroc_ci": q(d["auroc"])}


def joint_boot_diff(y, p_prov, p_temp, prov_mask, temp_mask, n_boot=N_BOOT, seed=SEED):
    """Provided-minus-temporal differences; resample trial IDs from the UNION of the two test sets.

    p_prov / p_temp are length-n arrays holding each arm's prediction at its own test rows. A
    resampled trial that lies in both test sets enters both arms (with each arm's prediction), so
    the ~20% overlap is respected.
    """
    U = np.where(prov_mask | temp_mask)[0]
    rng = np.random.default_rng(seed)
    dd = {"auprc": [], "auroc": [], "auprc_lift": []}
    for _ in range(n_boot):
        idx = U[rng.integers(0, len(U), len(U))]
        ip, it = idx[prov_mask[idx]], idx[temp_mask[idx]]
        yp, yt = y[ip], y[it]
        if len(set(yp)) < 2 or len(set(yt)) < 2:
            continue
        ap, at = T._auprc(yp, p_prov[ip]), T._auprc(yt, p_temp[it])
        dd["auprc"].append(ap - at)
        dd["auroc"].append(T._auroc(yp, p_prov[ip]) - T._auroc(yt, p_temp[it]))
        dd["auprc_lift"].append((ap - yp.mean()) - (at - yt.mean()))
    return {k: np.array(v) for k, v in dd.items()}


def fit_cols(P, cols, tr_idx, te_idx):
    X = P["X"][cols]
    return T._fit(X.iloc[tr_idx], P["y"][tr_idx], X.iloc[te_idx])


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def t1_evidence(P, ct):
    raw, y, ids = P["raw"], P["y"], P["ids"]
    enr = pd.to_numeric(raw["enrollment"], errors="coerce").values
    c = ct.reindex(ids)
    typ = c["enrollment_type"].values
    cnt = pd.to_numeric(c["enrollment_count"], errors="coerce").values
    ok = np.isfinite(cnt) & np.isfinite(enr)
    st = c["overall_status"].fillna("NOT_RETURNED").values
    by_status = {}
    for s in pd.Series(st).value_counts().index:
        m = st == s
        by_status[s] = {"n": int(m.sum()), "approval_rate": r4(y[m].mean()),
                        "median_trialbench_enrollment": r4(np.nanmedian(enr[m]))}
    city_missing = pd.to_numeric(raw[CITY[1]], errors="coerce").isna().values
    nloc = pd.to_numeric(c["n_locations"], errors="coerce").values
    return {
        "n_trials": int(len(y)),
        "ctgov_returned": int(pd.notna(c["overall_status"]).sum()),
        "current_enrollment_type_counts": {str(k): int(v) for k, v in
                                           pd.Series(typ).fillna("NA").value_counts().items()},
        "frac_current_enrollment_type_ACTUAL": r4(np.mean(typ == "ACTUAL")),
        "frac_trialbench_enrollment_equals_current_count": r4(np.mean(enr[ok] == cnt[ok])),
        "frac_equal_among_ACTUAL": r4(np.mean((enr == cnt)[ok & (typ == "ACTUAL")])),
        "by_current_overall_status": by_status,
        "median_enrollment_by_label": {"approved": r4(np.nanmedian(enr[y == 1])),
                                       "not_approved": r4(np.nanmedian(enr[y == 0]))},
        "city_covariates_missing": {"frac": r4(city_missing.mean()),
                                    "approval_rate_missing": r4(y[city_missing].mean()),
                                    "approval_rate_present": r4(y[~city_missing].mean()),
                                    "frac_missing_among_ctgov_zero_locations":
                                        r4(city_missing[nloc == 0].mean()) if np.any(nloc == 0) else None,
                                    "n_ctgov_zero_locations": int(np.sum(nloc == 0))},
    }


def enrollment_alone(P):
    y, raw = P["y"], P["raw"]
    enr = pd.to_numeric(raw["enrollment"], errors="coerce").fillna(0.0).values
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            draws.append(roc_auc_score(y[idx], enr[idx]))
    res = {"univariate_auroc_all_trials": r4(roc_auc_score(y, enr)), "univariate_auroc_all_trials_ci": q(draws)}
    for name, mask in (("provided_split", P["prov_test"]), ("temporal_split", P["temp_test"])):
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        for tag, cols in (("gbdt_enrollment_only", ["enrollment"]), ("gbdt_city_covariates_only", CITY)):
            p = fit_cols(P, cols, tr, te)
            ev = T._evaluate(y[te], p, SEED)
            res[f"{tag}_{name}"] = {k: ev[k] for k in ("test_base_rate", "auroc", "auroc_ci", "auprc",
                                                       "auprc_ci", "auprc_lift", "auprc_lift_ci")}
        res[f"univariate_auroc_{name}_test"] = r4(roc_auc_score(y[te], enr[te]))
    return res


def t2_compare(P, cols):
    y, prov, temp = P["y"], P["prov_test"], P["temp_test"]
    res, full = {}, {}
    for name, mask in (("provided_split", prov), ("temporal_split", temp)):
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        p = fit_cols(P, cols, tr, te)
        pf = np.full(P["n"], np.nan)
        pf[te] = p
        full[name] = pf
        res[name] = T._evaluate(y[te], p, SEED)
        res[name]["train_base_rate"] = r4(y[tr].mean())
    d = joint_boot_diff(y, full["provided_split"], full["temporal_split"], prov, temp)
    diff = {}
    for k in ("auprc", "auroc", "auprc_lift"):
        diff[k] = {"point": r4(res["provided_split"][k] - res["temporal_split"][k]), "ci_joint": q(d[k]),
                   "excludes_0": bool(q(d[k])[0] > 0 or q(d[k])[1] < 0)}
    res["provided_minus_temporal_joint_bootstrap"] = diff
    res["n_features"] = len(cols)
    return res


def t3_fixed(P, cols):
    y, X, nct, order, prov, temp = P["y"], P["X"], P["nct"], P["order"], P["prov_test"], P["temp_test"]
    n, n_test = P["n"], P["n_test"]
    out = {}
    # (i) as specified (mirrors sensitivity_reviewer.py)
    te = np.where(temp)[0]
    a_full = np.where(~temp)[0]
    pool = np.where(~prov & ~temp)[0]
    a_sub = np.sort(np.random.default_rng(SEED).choice(a_full, size=len(pool), replace=False))
    yte = y[te]
    arms = {"a_temporal_full": a_full, "a_temporal_size_matched": a_sub, "b_provided_pool": pool}
    Pr = {k: fit_cols(P, cols, idx, te) for k, idx in arms.items()}
    D = trial_boot(yte, Pr)
    out["i_fixed_temporal_test"] = {
        "n_test": int(len(te)), "test_base_rate": r4(yte.mean()),
        "pool_is_subset_of_temporal_train": bool(np.isin(pool, a_full).all()),
        "arms": {k: arm(yte, Pr[k], D[k], y[idx]) for k, idx in arms.items()},
        "b_minus_a_size_matched": contrast(yte, Pr["a_temporal_size_matched"], Pr["b_provided_pool"],
                                           D["a_temporal_size_matched"], D["b_provided_pool"]),
        "b_minus_a_full": contrast(yte, Pr["a_temporal_full"], Pr["b_provided_pool"],
                                   D["a_temporal_full"], D["b_provided_pool"])}
    # (ii) middle block (mirrors sensitivity_reviewer.py)
    lo = (n - n_test) // 2
    te2 = np.sort(order[lo:lo + n_test])
    past, future = np.sort(order[:lo]), np.sort(order[lo + n_test:])
    nonte = np.concatenate([past, future])
    mix = np.sort(np.random.default_rng(SEED).choice(nonte, size=len(past), replace=False))
    fut = future if len(future) == len(past) else \
        np.sort(np.random.default_rng(SEED).choice(future, size=len(past), replace=False))
    yte2 = y[te2]
    arms2 = {"a_past_only": past, "b_mix_past_future": mix, "c_future_only": fut}
    P2 = {k: fit_cols(P, cols, idx, te2) for k, idx in arms2.items()}
    D2 = trial_boot(yte2, P2)
    out["ii_fixed_middle_block"] = {
        "n_test": int(len(te2)), "test_base_rate": r4(yte2.mean()),
        "frac_future_in_mix": r4(np.isin(mix, future).mean()),
        "arms": {k: arm(yte2, P2[k], D2[k], y[idx]) for k, idx in arms2.items()},
        "b_mix_minus_a_past": contrast(yte2, P2["a_past_only"], P2["b_mix_past_future"],
                                       D2["a_past_only"], D2["b_mix_past_future"]),
        "c_future_minus_a_past": contrast(yte2, P2["a_past_only"], P2["c_future_only"],
                                          D2["a_past_only"], D2["c_future_only"])}
    # (iii) temporal test set, 2-fold cross-fitted non-temporal mix vs past-only
    p_past = fit_cols(P, cols, a_full, te)          # identical to the temporal-split model
    seeds = {}
    for s in MIX_SEEDS:
        rng = np.random.default_rng(s)
        perm = rng.permutation(te)
        folds = [np.sort(perm[:len(te) // 2]), np.sort(perm[len(te) // 2:])]
        pm = np.full(n, np.nan)
        frac_recent = []
        for k in (0, 1):
            ev, other = folds[k], folds[1 - k]
            mixk = np.sort(rng.choice(np.concatenate([a_full, other]), size=len(a_full), replace=False))
            frac_recent.append(float(np.isin(mixk, other).mean()))
            pm[ev] = fit_cols(P, cols, mixk, ev)
        seeds[s] = (pm[te], float(np.mean(frac_recent)))
    pm0 = seeds[0][0]
    D3 = trial_boot(yte, {"past": p_past, "mix": pm0})
    out["iii_temporal_test_crossfit"] = {
        "n_test": int(len(te)), "test_base_rate": r4(yte.mean()),
        "n_train_each_arm": int(len(a_full)),
        "frac_test_era_trials_in_mix_seed0": r4(seeds[0][1]),
        "arms": {"a_past_only": arm(yte, p_past, D3["past"], y[a_full]),
                 "b_mix_with_test_era": {"auprc": r4(T._auprc(yte, pm0)), "auprc_ci": q(D3["mix"]["auprc"]),
                                         "auroc": r4(T._auroc(yte, pm0)), "auroc_ci": q(D3["mix"]["auroc"])}},
        "b_minus_a": contrast(yte, p_past, pm0, D3["past"], D3["mix"]),
        "b_minus_a_over_5_mix_seeds": {
            "auprc": [r4(T._auprc(yte, seeds[s][0]) - T._auprc(yte, p_past)) for s in MIX_SEEDS],
            "auroc": [r4(T._auroc(yte, seeds[s][0]) - T._auroc(yte, p_past)) for s in MIX_SEEDS]}}
    return out


def t4_maturity(P, years):
    yr = np.array([years.get(i, np.nan) for i in P["ids"]], float)
    ok = np.isfinite(yr)
    y = P["y"]
    qq = pd.qcut(pd.Series(yr[ok]), 4, duplicates="drop")
    by_q = pd.Series(y[ok]).groupby(qq.values, observed=True).mean()
    res = {"approval_rate_by_start_year_quartile": {str(k): r4(v) for k, v in by_q.items()},
           "overall_approval_rate": r4(y.mean())}
    for name, mask in (("provided_test", P["prov_test"]), ("temporal_test", P["temp_test"])):
        m = mask & ok
        res[name] = {"n": int(mask.sum()), "prevalence": r4(y[mask].mean()),
                     "median_start_year": r4(np.median(yr[m])),
                     "frac_start_year_ge_2015": r4(np.mean(yr[m] >= 2015))}
    return res


def main():
    ap = argparse.ArgumentParser(description="TrialBench split audit v2.")
    ap.add_argument("--out", default="outputs/trialbench_audit_v2.json")
    ap.add_argument("--no-fetch", action="store_true", help="use only the cached CT.gov table")
    args = ap.parse_args()
    t0 = time.time()
    T.ensure_data()  # no-op when present
    pub = {p["phase"]: p for p in json.load(open("outputs/trialbench_audit.json"))["phases"]}
    sd = pd.read_csv(T.START_DATES_CSV)  # read only
    years = dict(zip(sd["nct_id"].astype(str), sd["start_year"]))

    phases = {ph: load_phase(ph) for ph in T.PHASES}
    all_ids = [i for P in phases.values() for i in P["ids"]]
    if args.no_fetch and os.path.exists(CTGOV_V2_CSV):
        ct = pd.read_csv(CTGOV_V2_CSV).drop_duplicates("nct_id").set_index("nct_id")
    else:
        print("fetching ClinicalTrials.gov enrollment/status (cached) ...", flush=True)
        ct = fetch_ctgov(all_ids)

    raw_cols = list(phases["Phase1"]["raw"].columns)
    model_cols = {ph: list(P["X"].columns) for ph, P in phases.items()}
    feature_table = {}
    for c in raw_cols:
        timing, why = raw_timing(c)
        used_numeric = c in model_cols["Phase1"] or c in model_cols["Phase2"]
        used_onehot = c in T.CATS
        feature_table[c] = {"used_by_model": bool(used_numeric or used_onehot),
                            "how": "numeric" if used_numeric else ("one-hot" if used_onehot else "not used"),
                            "timing": timing, "justification": why}

    out = {"benchmark": "TrialBench approval forecasting (arXiv:2407.00631; Zenodo 14975339)",
           "script": "scripts/audit_trialbench_v2.py (reuses scripts/audit_trialbench.py unchanged)",
           "settings": {"learner": "audit_trialbench._fit (HGB default, seed 0)",
                        "per_arm_ci": "audit_trialbench._evaluate: trial-level percentile bootstrap, 1,000, seed 0",
                        "difference_ci": "joint bootstrap over the union of provided and temporal test IDs, 1,000, seed 0",
                        "fixed_test_ci": "paired trial-level bootstrap on the fixed test set, 1,000, seed 0",
                        "non_start_time_features_removed": NON_START},
           "T1_feature_timing": {
               "trialbench_paper_quotes": [
                   "\"The data, representing clinical trials registered before February 16, 2024, were collected "
                   "from ClinicalTrials.gov. We extracted elements and attributes from the XML records of each "
                   "clinical trial and converted them into tabular data formats\"",
                   "\"we manually determined the prediction objectives for each task and selected variables "
                   "according to the timing of applying AI in real-world practice. For instance, we ensured that "
                   "trial result information was not included if the AI task is to be performed before trial "
                   "completion.\"",
                   "Approval ground truth from TrialTrove (\"trial approval information as groundtruth\")."],
               "regulatory_quotes": {
                   "42_CFR_11.10(b)(18)": "\"Enrollment means the estimated total number of human subjects to be "
                                          "enrolled (target number) or the actual total number of human subjects "
                                          "that are enrolled in the clinical trial. Once the trial has reached the "
                                          "primary completion date, the responsible party must update the "
                                          "Enrollment data element to reflect the actual number of human subjects "
                                          "enrolled in the clinical trial.\"",
                   "42_CFR_11.64": "\"Individual Site Status must be updated not later than 30 calendar days after a "
                                   "change in status for any individual site.\""},
               "columns": feature_table,
               "model_features_per_phase": model_cols,
               "non_start_time_model_features": NON_START,
               "phases": {}},
           "T2_provided_vs_temporal": {}, "T3_fixed_test_set": {}, "T4_prevalence_and_maturity": {},
           "reproduction_check": {}}

    for ph, P in phases.items():
        print(f"== {ph}", flush=True)
        out["T1_feature_timing"]["phases"][ph] = {"ctgov_evidence": t1_evidence(P, ct),
                                                  "enrollment_alone": enrollment_alone(P)}
        fs = feature_sets(P["X"])
        out["T2_provided_vs_temporal"][ph] = {
            "n": P["n"], "n_test": P["n_test"],
            "overlap_provided_temporal_test": int((P["prov_test"] & P["temp_test"]).sum()),
            "union_size": int((P["prov_test"] | P["temp_test"]).sum())}
        for name, cols in fs.items():
            out["T2_provided_vs_temporal"][ph][name] = t2_compare(P, cols)
        # reproduction check of the published all-feature numbers
        full = out["T2_provided_vs_temporal"][ph]["all_features"]
        mism = [f"{s}.{k}" for s in ("provided_split", "temporal_split")
                for k in ("test_base_rate", "auprc", "auroc", "auprc_lift", "auprc_ci", "auroc_ci", "auprc_lift_ci")
                if full[s][k] != pub[ph][s][k]]
        out["reproduction_check"][ph] = {"exact_match": not mism, "mismatches": mism}
        print(f"   T2 done; reproduction exact={not mism}", flush=True)
        out["T3_fixed_test_set"][ph] = {"ablated_non_start_removed": t3_fixed(P, fs["ablated_non_start_removed"]),
                                        "all_features": t3_fixed(P, fs["all_features"])}
        print("   T3 done", flush=True)
        out["T4_prevalence_and_maturity"][ph] = t4_maturity(P, years)

    for name in ("all_features", "ablated_non_start_removed", "minus_enrollment_only"):
        for k in ("auprc", "auroc", "auprc_lift"):
            out[f"mean_provided_minus_temporal_{k}_{name}"] = r4(np.mean(
                [out["T2_provided_vs_temporal"][ph][name]["provided_minus_temporal_joint_bootstrap"][k]["point"]
                 for ph in T.PHASES]))
    out["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out} ({out['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
