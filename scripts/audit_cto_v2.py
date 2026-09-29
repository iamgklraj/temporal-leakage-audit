"""Corrected temporal-leakage audit of CTO's labeling functions (reviewer response).

A domain reviewer raised four problems with scripts/audit_cto.py; this script checks each one
against the data and re-runs the audit with corrections. audit_cto.py is imported and
reused unchanged (load, _fit_predict, _clustered_samples, _auprc, _auroc). The default learner, the
temporal split (train: start year <= 60th percentile) and the sponsor-clustered percentile
bootstrap (1,000 resamples, seed 0, paired for contrasts) are the same as in the original.

 C1  Label construction. CTO built its curated labels by applying a rule first: stopped statuses
     -> failure; primary p < 0.05 -> success. Experts reviewed only the trials the rule left
     unlabeled. We tabulate the human label by overall_status and check how often the label
     agrees with the rule.
 C2  Duplicate LF columns (status2, gpt2, linkage2, hint_train2, hint_train3) and abstention.
 C3  When each LF becomes public (see LF_TIMING; sources are CTO's released code at
     github.com/sunlabuiuc/CTO, arXiv:2406.10292v3, 42 CFR 11).
 C4  'sites' abstention vs withdrawn trials; start-time model with withdrawn trials excluded.

 D   Reproduction of the published default numbers (outputs/cto_audit.json).
 A   Deduplicated LFs re-tiered by when they become public. For the start-time set, each
     cumulative tier and all LFs: AUPRC, AUROC and lift (AUPRC minus base rate), each with a CI.
     Also LAP (all minus start-time) and its components, with paired CIs.
 B   Sensitivity: withdrawn trials excluded; withdrawn and suspended excluded (both refits,
     and the A models evaluated on the restricted test set); 'sites' treated as post-start.
 C   Exploratory: marginal gain of each LF added alone to the corrected start-time set.

Usage:  python scripts/audit_cto_v2.py [--data-dir data/cto] [--out outputs/cto_audit_v2.json]
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import audit_cto as A  # noqa: E402

N_BOOT = 1000
SEED = 0
STOPPED = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
DUPLICATES = {"status2": "status", "gpt2": "gpt", "linkage2": "linkage",
              "hint_train2": "hint_train", "hint_train3": "hint_train"}

# --------------------------------------------------------------------------- #
# C3: public-availability classification of each (deduplicated) labeling function.
# Code references are to github.com/sunlabuiuc/CTO @ f8fb234 (labeling/lfs.py unless noted).
# --------------------------------------------------------------------------- #
LF_TIMING = {
    "num_sponsors": {
        "tier": "start",
        "public": "registration (protocol data element)",
        "definition": "count of sponsor + collaborator names in AACT sponsors.txt > phase-specific "
                      "quantile (lf_num_sponsors)",
        "justification": "Lead sponsor and collaborators are protocol-registration data elements "
                         "(42 CFR 11.10(b)). The value comes from the final record, so it can "
                         "include collaborators added later. The column exists only in CTO's "
                         "phase-2 table and is NaN for rows from the phase-1/3 tables.",
    },
    "sites": {
        "tier": "start (lenient; variant B3 treats it as post-start)",
        "public": "registration onward; value used is the FINAL record",
        "definition": "count of facility names in AACT facilities.txt > phase quantile (lf_sites); "
                      "paper: 'The number of total sites during the trial'",
        "justification": "Facilities are listed from registration, but the list changes during "
                         "recruitment: individual site status must be updated within 30 days "
                         "(42 CFR 11.64). CTO counts the final-record list, and the LF abstains "
                         "(-1) when no facility is listed, which happens mostly for withdrawn "
                         "trials (a post-registration outcome). Kept in the start tier only as a "
                         "lenient upper bound.",
    },
    "status": {
        "tier": "completion_registry",
        "public": "when the trial stops/ends (status update within 30 days, 42 CFR 11.64)",
        "definition": "overall_status in {terminated, withdrawn, suspended, withheld, no longer "
                      "available, temporarily not available} -> 0; 'approved for marketing' -> 1; "
                      "else abstain (lf_status)",
        "justification": "A terminal status exists only once the trial has stopped; for a stopped "
                         "trial that is its end. It is also the rule CTO applied before human "
                         "review when building the curated labels (C1).",
    },
    "results_reported": {
        "tier": "completion_registry",
        "public": "after primary completion (results due within 1 year, 42 CFR 11.44)",
        "definition": "AACT calculated_values.were_results_reported == 't'",
        "justification": "Results posting happens after primary completion.",
    },
    "pvalues": {
        "tier": "completion_registry",
        "public": "after primary completion (results section)",
        "definition": "any primary-outcome p < 0.05 in AACT outcome_analyses (lf_pvalues)",
        "justification": "Statistical analyses are part of the results section (42 CFR 11.48(a)(3)). "
                         "This is also the rule CTO applied before human review (C1).",
    },
    "num_patients": {
        "tier": "completion_registry",
        "public": "after primary completion (results section)",
        "definition": "sum of AACT outcome_counts.count (participants analysed per outcome), not "
                      "enrollment (lf_num_patients)",
        "justification": "outcome_counts is results-section data.",
    },
    "patient_drop": {
        "tier": "completion_registry",
        "public": "after primary completion (results: participant flow)",
        "definition": "sum of AACT drop_withdrawals.count < phase quantile (lf_patient_drop)",
        "justification": "Participant flow is results-section data (42 CFR 11.48(a)(1)).",
    },
    "serious_ae": {
        "tier": "completion_registry",
        "public": "after primary completion (results: adverse events)",
        "definition": "sum of subjects_affected for event_type 'serious' in AACT "
                      "reported_event_totals <= quantile",
        "justification": "Adverse-event tables are results-section data (42 CFR 11.48(a)(4)).",
    },
    "death_ae": {
        "tier": "completion_registry",
        "public": "after primary completion (results: adverse events)",
        "definition": "sum of subjects_affected for event_type 'deaths' in reported_event_totals",
        "justification": "Adverse-event tables are results-section data (42 CFR 11.48(a)(4)).",
    },
    "all_ae": {
        "tier": "completion_registry",
        "public": "after primary completion (results: adverse events)",
        "definition": "sum of subjects_affected over all event types in reported_event_totals",
        "justification": "Adverse-event tables are results-section data (42 CFR 11.48(a)(4)).",
    },
    "update_more_recent": {
        "tier": "completion_registry",
        "public": "after completion (by definition)",
        "definition": "last_update_submitted_date - completion_date < median (lf_update_more_recent)",
        "justification": "Defined relative to the completion date and the last record update.",
    },
    "amendments": {
        "tier": "completion_registry",
        "public": "as-of-scrape (2024) total of record versions",
        "definition": "number of versions on the ClinicalTrials.gov history tab "
                      "(stock_price/scrape_amendments.py: versions_df['Version'].iloc[-2]) > quantile",
        "justification": "A total over the whole record history at scrape time, including "
                         "versions posted during and after the trial (e.g. results posting). The "
                         "count at trial start is not what CTO uses.",
    },
    "linkage": {
        "tier": "external_post_completion",
        "public": "after completion (later-phase trials / FDA approval)",
        "definition": "Success if a later-phase trial or FDA approval is linked; Failure if none is "
                      "found and completion was > 6 months before the scrape "
                      "(clinical_trial_linkage/extract_outcome_from_trial_linkage.py)",
        "justification": "Paper: 'Is positive if a trial was found to have any later-stage trials "
                         "linked to it'. It is defined by what happened after the trial.",
    },
    "gpt": {
        "tier": "external_post_completion",
        "public": "after completion (publications)",
        "definition": "GPT-3.5 verdict on PubMed Derived/Results abstracts published up to 5 years "
                      "after completion (llm_prediction_on_pubmed/support_functions.py)",
        "justification": "Uses results publications.",
    },
    "stock_price": {
        "tier": "external_post_completion",
        "public": "after completion",
        "definition": "sign of the slope of the sponsor's 5-day SMA over the 7 days after "
                      "completion_date (stock_price/get_stocks.py)",
        "justification": "The window starts at completion by construction.",
    },
    "new_headlines": {
        "tier": "external_post_completion",
        "public": "after completion (news scraped in 2024)",
        "definition": "sentiment of Google-News headlines matched to COMPLETED trials "
                      "(news_headlines/get_news2.py; studies filtered to overall_status == "
                      "'COMPLETED')",
        "justification": "Computed only for completed trials, from news up to the 2024 scrape.",
    },
    "hint_train": {
        "tier": "external_post_completion",
        "public": "after completion (published outcome label)",
        "definition": "TOP/HINT benchmark outcome label (Fu et al. 2022), phase*_{train,valid,"
                      "test}.csv 'label' (hint_train_lf); triplicated as hint_train/2/3",
        "justification": "It is an outcome label from another benchmark, not a start-time "
                         "covariate (paper: 'We duplicate TOP labels 3 times to obtain high "
                         "agreement'). audit_cto.py wrongly put it in the start tier. It abstains "
                         "for every test trial.",
    },
}
START_V2 = [k for k, v in LF_TIMING.items() if v["tier"].startswith("start")]
COMPL_V2 = [k for k, v in LF_TIMING.items() if v["tier"] == "completion_registry"]
EXT_V2 = [k for k, v in LF_TIMING.items() if v["tier"] == "external_post_completion"]
ALL_V2 = START_V2 + COMPL_V2 + EXT_V2


def tiers_v2(start=START_V2, compl=COMPL_V2, ext=EXT_V2):
    return [("t0_start", list(start)),
            ("t1_start_plus_completion_registry", list(start) + list(compl)),
            ("t2_all", list(start) + list(compl) + list(ext))]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def q(a):
    a = np.asarray([x for x in a if np.isfinite(x)])
    return [r4(np.quantile(a, .025)), r4(np.quantile(a, .975))]


def clustered_draws(y, preds, clusters, n_boot=N_BOOT, seed=SEED):
    """Sponsor-clustered bootstrap draws of AUPRC, AUROC and base rate for every key.

    Same random stream and skip rule as audit_cto._clustered_samples, so the AUPRC draws (and
    hence the published AUPRC CIs) are identical; AUROC and the resample base rate are added.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    c2r = {c: np.where(clusters == c)[0] for c in uniq}
    out = {k: {"auprc": [], "auroc": [], "base": []} for k in preds}
    for _ in range(n_boot):
        rows = np.concatenate([c2r[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        yb = y[rows]
        if len(np.unique(yb)) < 2:
            continue
        b = yb.mean()
        for k, p in preds.items():
            out[k]["auprc"].append(A._auprc(yb, p[rows]))
            out[k]["auroc"].append(A._auroc(yb, p[rows]))
            out[k]["base"].append(b)
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def summarize(y, p, d, cols=None):
    base = float(y.mean())
    a = A._auprc(y, p)
    res = {"auprc": r4(a), "auprc_ci": q(d["auprc"]),
           "auroc": r4(A._auroc(y, p)), "auroc_ci": q(d["auroc"]),
           "lift": r4(a - base), "lift_ci": q(d["auprc"] - d["base"])}
    if cols is not None:
        res["n_lfs"] = len(cols)
        res["lfs"] = list(cols)
    return res


def paired(y, pa, pb, da, db):
    """b minus a, paired on the same resamples (AUPRC difference == lift difference)."""
    out = {}
    for m, f in (("auprc", A._auprc), ("auroc", A._auroc)):
        dd = db[m] - da[m]
        out[m] = {"point": r4(f(y, pb) - f(y, pa)), "ci": q(dd),
                  "p_gt_0": round(float(np.mean(dd > 0)), 4)}
    return out


def run_tiers(tr, te, tiers, n_boot=N_BOOT, preds=None):
    """Fit each tier (unless preds given), bootstrap on te, summarise tiers + LAP."""
    y = te["labels"].values
    if preds is None:
        preds = {name: A._fit_predict(tr, te, cols) for name, cols in tiers}
    d = clustered_draws(y, preds, te["source"].values, n_boot=n_boot)
    names = [n for n, _ in tiers]
    t0, t1, t2 = names
    base = float(y.mean())
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_base_rate": r4(base),
           "n_test_positive": int(y.sum()),
           "tiers": {n: summarize(y, preds[n], d[n], c) for n, c in tiers}}
    lap = paired(y, preds[t0], preds[t2], d[t0], d[t2])
    res["LAP_all_minus_start"] = lap
    res["LAP_components"] = {f"{t1}_minus_{t0}": paired(y, preds[t0], preds[t1], d[t0], d[t1]),
                             f"{t2}_minus_{t1}": paired(y, preds[t1], preds[t2], d[t1], d[t2])}
    a_all, a_st = A._auprc(y, preds[t2]), A._auprc(y, preds[t0])
    res["share_of_auprc_all_that_is_post_start"] = r4((a_all - a_st) / a_all)
    res["share_of_lift_all_that_is_post_start"] = r4((a_all - a_st) / (a_all - base))
    res["start_lift_over_base"] = r4(a_st - base)
    return res, preds, d


# --------------------------------------------------------------------------- #
# Claim checks
# --------------------------------------------------------------------------- #

def claim_c1(j, cut):
    j = j.copy()
    j["split"] = np.where(j["start_year"] <= cut, "train", "test")
    tab = {}
    for sp in ("train", "test", "all"):
        s = j if sp == "all" else j[j["split"] == sp]
        ct = pd.crosstab(s["overall_status"], s["labels"])
        tab[sp] = {st: {"label_0": int(ct.loc[st].get(0, 0)), "label_1": int(ct.loc[st].get(1, 0))}
                   for st in ct.index}
    comp = j[j["overall_status"] == "COMPLETED"]
    by_p = {}
    for sp in ("train", "test"):
        s = comp[comp["split"] == sp]
        by_p[sp] = {f"pvalues_lf={int(v)}": {"label_0": int(((s["pvalues"] == v) & (s["labels"] == 0)).sum()),
                                             "label_1": int(((s["pvalues"] == v) & (s["labels"] == 1)).sum())}
                    for v in (-1, 0, 1)}
    rule = j["overall_status"].isin(STOPPED) | ((j["overall_status"] == "COMPLETED") & (j["pvalues"] == 1))
    rule_label = np.where(j["overall_status"].isin(STOPPED), 0, 1)
    status_lf_matches = bool(((j["status"] == 0) == j["overall_status"].isin(STOPPED)).all()
                             and (j.loc[~j["overall_status"].isin(STOPPED), "status"] == -1).all())
    return {
        "verdict": "TRUE",
        "source_quote_arxiv_2406.10292v3": (
            "\"Automated labeling is not a substitute for human expertise, so we manually annotated "
            "over 2,500 challenging cases using our knowledge base to create a high-confidence gold "
            "standard set for evaluation. The process for developing this benchmark is illustrated "
            "in Figure 1 D. We applied the same rule-based termination criteria and p-value threshold "
            "of less than 0.05 as used in the automated labeling step to filter out the easier cases. "
            "For the remaining unlabeled trials, we engaged three clinical trial experts to review "
            "relevant publications, news articles, and other reported outcomes to accurately "
            "determine the trial outcomes.\""),
        "rule_quote_arxiv_2406.10292v3": (
            "\"First, trials with statuses indicating 'terminated', 'withdrawn', 'suspended', 'no "
            "longer available'} were labeled as failures. Trials labeled as 'approved for marketing' "
            "were marked as successes. For trials with 'completed' as status, we examined the "
            "reported results: if any reported p-value for a primary endpoint was < 0.05, the trial "
            "was labeled as a success.\""),
        "nature_health_version": (
            "Published as 'A large-scale database for clinical trial outcomes and features', Nature "
            "Health (04 March 2026), doi:10.1038/s44360-026-00081-6 (the DOI given with 's41360' does "
            "not resolve). Methods are paywalled; the public abstract says 'We additionally manually "
            "annotated a subset of recent trials from 2020 to 2024', and the author-contributions "
            "note says 'Data annotation of the manually curated set was performed by all authors "
            "involved in the paper in May 2024, under the advisement of S.T. and J.S.'"),
        "label_by_overall_status": tab,
        "completed_trials_label_by_pvalues_lf": by_p,
        "rule_applicable_share": {"all": r4(rule.mean()),
                                  "train": r4(rule[j["split"] == "train"].mean()),
                                  "test": r4(rule[j["split"] == "test"].mean())},
        "label_agrees_with_rule_where_rule_applies": r4(float((j.loc[rule, "labels"].values
                                                               == rule_label[rule.values]).mean())),
        "status_lf_equals_stopped_status_indicator": status_lf_matches,
        "all_positive_labels_are_COMPLETED": bool((j.loc[j["labels"] == 1, "overall_status"]
                                                   == "COMPLETED").all()),
    }


def claim_c2(j, cut):
    lfs = A.START + A.DURING + A.POST
    X = j[lfs].fillna(-999)
    groups = {}
    for c in lfs:
        groups.setdefault(tuple(X[c].values), []).append(c)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]

    def distinct(cols):
        return len({tuple(X[c].values) for c in cols})
    abst = {c: {"train_abstain": r4((tr[c] == -1).mean()), "test_abstain": r4((te[c] == -1).mean()),
                "train_nan": r4(tr[c].isna().mean()), "test_nan": r4(te[c].isna().mean())}
            for c in lfs}
    return {
        "verdict": "TRUE",
        "pairs_identical_on_all_rows": {f"{a}=={b}": bool(((j[a] == j[b]) | (j[a].isna() & j[b].isna())).all())
                                        for a, b in DUPLICATES.items()},
        "duplicate_groups": dup_groups,
        "n_columns_used": len(lfs), "n_distinct_signals": len(groups),
        "distinct_per_original_tier": {"start (5 cols)": distinct(A.START),
                                       "start+during (13 cols)": distinct(A.START + A.DURING),
                                       "all (22 cols)": distinct(lfs)},
        "hint_train_test_abstain_fraction": r4((te["hint_train"] == -1).mean()),
        "hint_train_train_abstain_fraction": r4((tr["hint_train"] == -1).mean()),
        "abstention_by_lf": abst,
        "source": "CTO labeling/lfs.py get_lfs(): known_lfs_list = [hint_lf,hint_lf,hint_lf, "
                  "status_lf,status_lf, gpt_lf,gpt_lf, linkage_lf,linkage_lf, ...]",
    }


def claim_c3_evidence(j):
    """Empirical timing check: results-derived LFs vs results posting dates."""
    out = {}
    for lf in ("serious_ae", "death_ae", "all_ae", "patient_drop", "num_patients", "pvalues"):
        m = j[lf] != -1
        rp = pd.to_datetime(j.loc[m, "results_first_posted_date"], errors="coerce")
        pcd = pd.to_datetime(j.loc[m, "primary_completion_date"], errors="coerce")
        cd = pd.to_datetime(j.loc[m, "completion_date"], errors="coerce")
        out[lf] = {"n_non_abstaining": int(m.sum()),
                   "frac_with_results_posted": r4(rp.notna().mean()),
                   "frac_results_posted_after_primary_completion": r4(((rp - pcd).dt.days > 0).mean()),
                   "median_days_results_posted_after_primary_completion": r4((rp - pcd).dt.days.median()),
                   "frac_results_posted_on_or_after_study_completion": r4(((rp - cd).dt.days >= 0).mean())}
    return out


def claim_c4(j, cut, preds_orig_start):
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    ab = te[te["sites"] == -1]
    y = te["labels"].values
    keep_w = (te["overall_status"] != "WITHDRAWN").values
    keep_ws = (~te["overall_status"].isin(["WITHDRAWN", "SUSPENDED"])).values
    res = {"sites_abstain_test": int(len(ab)),
           "sites_abstain_test_by_status": {k: int(v) for k, v in ab["overall_status"].value_counts().items()},
           "sites_abstain_test_label_1": int(ab["labels"].sum()),
           "sites_abstain_train": int((tr["sites"] == -1).sum()),
           "sites_abstain_train_by_status": {k: int(v) for k, v in
                                             tr.loc[tr["sites"] == -1, "overall_status"].value_counts().items()},
           "withdrawn_test_n": int((te["overall_status"] == "WITHDRAWN").sum()),
           "withdrawn_test_sites_values": {str(int(k)): int(v) for k, v in
                                           te.loc[te["overall_status"] == "WITHDRAWN", "sites"].value_counts().items()}}
    for tag, keep in (("excl_withdrawn", keep_w), ("excl_withdrawn_suspended", keep_ws)):
        yk, pk = y[keep], preds_orig_start[keep]
        d = clustered_draws(yk, {"s": pk}, te["source"].values[keep])["s"]
        res[f"original_start_model_eval_only_{tag}"] = {"n_test": int(keep.sum()), "base_rate": r4(yk.mean()),
                                                        **summarize(yk, pk, d)}
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Corrected CTO temporal-leakage audit (v2).")
    ap.add_argument("--data-dir", default="data/cto")
    ap.add_argument("--out", default="outputs/cto_audit_v2.json")
    args = ap.parse_args()
    t_start = time.time()

    j = A.load(args.data_dir)
    hl = pd.read_csv(os.path.join(args.data_dir, "human_labels_2020_2024.csv"), low_memory=False)
    extra = hl[["nct_id", "overall_status", "results_first_posted_date", "primary_completion_date"]]
    j = j.merge(extra, on="nct_id", how="left")
    assert j["overall_status"].notna().all()
    cut = int(j["start_year"].quantile(0.6))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    yte = te["labels"].values

    # ---- D. Reproduce the published default run (original tiers, original bootstrap) ----
    pub = json.load(open(os.path.join(ROOT, "outputs", "cto_audit.json")))
    preds_orig = {name: A._fit_predict(tr, te, cols) for name, cols in A.TIERS}
    samp = A._clustered_samples(yte, preds_orig, te["source"].values, n_boot=N_BOOT, seed=SEED)
    repro = {"n": len(j), "n_train": len(tr), "n_test": len(te), "train_start_year_le": cut,
             "test_base_rate": r4(yte.mean())}
    for name, _ in A.TIERS:
        repro[name] = {"auprc": r4(A._auprc(yte, preds_orig[name])),
                       "auroc": r4(A._auroc(yte, preds_orig[name])), "auprc_ci": q(samp[name])}
    lap = np.array([a - b for a, b in zip(samp["h2_post_naive"], samp["h0_start"])])
    repro["LAP_auprc"] = r4(repro["h2_post_naive"]["auprc"] - repro["h0_start"]["auprc"])
    repro["LAP_auprc_ci"] = q(lap)
    checks = [repro["n"] == pub["n"], repro["n_test"] == pub["n_test"],
              repro["test_base_rate"] == pub["test_base_rate"], repro["LAP_auprc_ci"] == pub["LAP_auprc_ci"]]
    for name, _ in A.TIERS:
        for k in ("auprc", "auroc", "auprc_ci"):
            checks.append(repro[name][k] == pub["tiers"][name][k])
    repro["exact_match_to_published"] = bool(all(checks))
    print(f"[D] reproduction exact match: {repro['exact_match_to_published']}  "
          f"(start {repro['h0_start']['auprc']}, all {repro['h2_post_naive']['auprc']}, LAP {repro['LAP_auprc']})",
          flush=True)

    # Original tiering with AUROC / lift CIs (22 cols and deduplicated 17 cols).
    orig_enriched, _, _ = run_tiers(tr, te, A.TIERS, preds=preds_orig)
    dedup = lambda cols: [c for c in cols if c not in DUPLICATES]  # noqa: E731
    tiers_orig_dedup = [(n, dedup(c)) for n, c in A.TIERS]
    orig_dedup, _, _ = run_tiers(tr, te, tiers_orig_dedup)
    # Is the original start tier identical to sites + num_sponsors alone?
    p_sn = A._fit_predict(tr, te, ["sites", "num_sponsors"])
    p_o = preds_orig["h0_start"]
    same_as_sites_sponsors = {
        "spearman_rho": r4(pd.Series(p_sn).corr(pd.Series(p_o), method="spearman")),
        "max_abs_prob_diff": r4(np.abs(p_sn - p_o).max()),
        "n_distinct_scores": [int(len(np.unique(p_o))), int(len(np.unique(p_sn)))],
        "auprc_auroc_identical": bool(A._auprc(yte, p_sn) == A._auprc(yte, p_o)
                                      and A._auroc(yte, p_sn) == A._auroc(yte, p_o)),
        "note": "HINT columns abstain for all test trials, so the original 5-LF start tier ranks test "
                "trials exactly as sites + num_sponsors does"}

    # ---- C1-C4 ----
    c1 = claim_c1(j, cut)
    c2 = claim_c2(j, cut)
    c3 = {"verdict": "TRUE (with refinements, see lf_timing)", "lf_timing": LF_TIMING,
          "results_posting_evidence": claim_c3_evidence(j),
          "original_audit_misassignments": {
              "hint_train/2/3": "START in audit_cto.py -> external post-completion (outcome label of the TOP benchmark)",
              "linkage/2": "DURING -> external post-completion (later-phase trials / approvals)",
              "amendments": "DURING -> as-of-scrape record-history total (available only at/after completion)",
              "patient_drop, num_patients, serious_ae, death_ae, all_ae": "DURING -> results section, posted after primary completion",
              "sites": "START -> final-record facility count; start-time only under a lenient reading"},
          "regulatory_quotes": {
              "42_CFR_11.10(b)(18)": "\"Enrollment means the estimated total number of human subjects to be enrolled "
                                     "(target number) or the actual total number of human subjects that are enrolled "
                                     "in the clinical trial. Once the trial has reached the primary completion date, "
                                     "the responsible party must update the Enrollment data element to reflect the "
                                     "actual number of human subjects enrolled in the clinical trial.\"",
              "42_CFR_11.44(a)": "results information \"must be submitted no later than 1 year after the primary "
                                 "completion date of the applicable clinical trial.\"",
              "42_CFR_11.64": "\"Individual Site Status must be updated not later than 30 calendar days after a "
                              "change in status for any individual site.\""}}
    c4 = claim_c4(j, cut, preds_orig["h0_start"])
    c4["verdict"] = "TRUE"
    print("[C] claim checks done", flush=True)

    # ---- A. Corrected tiers ----
    main_res, preds_main, _ = run_tiers(tr, te, tiers_v2())
    main_res["start_set_status"] = {
        lf: {"test_abstain": r4((te[lf] == -1).mean()), "test_nan": r4(te[lf].isna().mean())} for lf in START_V2}
    print(f"[A] corrected: start AUPRC {main_res['tiers']['t0_start']['auprc']} "
          f"AUROC {main_res['tiers']['t0_start']['auroc']}; all {main_res['tiers']['t2_all']['auprc']}; "
          f"LAP {main_res['LAP_all_minus_start']['auprc']}", flush=True)

    # ---- B. Sensitivities ----
    sens = {}
    for tag, excl in (("excl_withdrawn_refit", ["WITHDRAWN"]),
                      ("excl_withdrawn_suspended_refit", ["WITHDRAWN", "SUSPENDED"])):
        trk, tek = tr[~tr["overall_status"].isin(excl)], te[~te["overall_status"].isin(excl)]
        sens[tag], _, _ = run_tiers(trk, tek, tiers_v2())
    for tag, excl in (("excl_withdrawn_eval_only", ["WITHDRAWN"]),
                      ("excl_withdrawn_suspended_eval_only", ["WITHDRAWN", "SUSPENDED"])):
        keep = (~te["overall_status"].isin(excl)).values
        sens[tag], _, _ = run_tiers(tr, te[keep], tiers_v2(),
                                    preds={k: v[keep] for k, v in preds_main.items()})
    sites_post = tiers_v2(start=["num_sponsors"], compl=["sites"] + COMPL_V2)
    sens["sites_as_post_start"], _, _ = run_tiers(tr, te, sites_post)
    trk, tek = tr[tr["overall_status"] != "WITHDRAWN"], te[te["overall_status"] != "WITHDRAWN"]
    sens["sites_as_post_start_excl_withdrawn_refit"], _, _ = run_tiers(trk, tek, sites_post)
    print("[B] sensitivities done", flush=True)

    # ---- C. Per-LF marginal gain over the corrected start-time set (exploratory) ----
    others = COMPL_V2 + EXT_V2
    lf_preds = {"__start__": preds_main["t0_start"]}
    for lf in others:
        lf_preds[lf] = A._fit_predict(tr, te, START_V2 + [lf])
    dl = clustered_draws(yte, lf_preds, te["source"].values)
    per_lf = {}
    for lf in others:
        pr = paired(yte, lf_preds["__start__"], lf_preds[lf], dl["__start__"], dl[lf])
        per_lf[lf] = {"tier": LF_TIMING[lf]["tier"],
                      "auprc_with_lf": r4(A._auprc(yte, lf_preds[lf])),
                      "marginal_auprc": pr["auprc"], "marginal_auroc": pr["auroc"],
                      "test_abstain": r4((te[lf] == -1).mean())}
    per_lf = dict(sorted(per_lf.items(), key=lambda kv: -kv[1]["marginal_auprc"]["point"]))
    print("[C] per-LF done", flush=True)

    out = {
        "benchmark": "CTO (Gao et al.; arXiv:2406.10292v3; Nature Health 2026, doi:10.1038/s44360-026-00081-6)",
        "script": "scripts/audit_cto_v2.py (reuses scripts/audit_cto.py unchanged)",
        "settings": {"learner": "audit_cto._fit_predict (HistGradientBoosting, max_depth 3, lr 0.06, 300 iter, "
                                "l2 1.0, seed 0)",
                     "split": f"temporal: train start_year <= {cut} (60th percentile), test later",
                     "bootstrap": f"sponsor-clustered percentile, {N_BOOT} resamples, seed {SEED}; paired for contrasts",
                     "lift": "AUPRC minus test base rate (per resample for CIs)",
                     "cto_source": "github.com/sunlabuiuc/CTO @ f8fb234 (2026-02-17)"},
        "D_reproduction_of_published": repro,
        "claims": {"C1_labels_not_independent_of_LFs": c1,
                   "C2_duplicate_columns_and_abstention": c2,
                   "C3_vintage_by_public_availability": c3,
                   "C4_sites_abstention_and_withdrawn": c4},
        "A_corrected_tiers": {"tier_definitions": {"start": START_V2, "completion_registry": COMPL_V2,
                                                   "external_post_completion": EXT_V2},
                              **main_res},
        "original_tiering_with_auroc_and_lift": orig_enriched,
        "original_tiering_deduplicated": orig_dedup,
        "original_start_tier_vs_sites_plus_num_sponsors": same_as_sites_sponsors,
        "B_sensitivity": sens,
        "C_per_lf_marginal_over_corrected_start_exploratory": per_lf,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out} ({out['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
