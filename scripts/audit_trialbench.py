"""Split-level temporal audit of TrialBench (approval forecasting, Phases I-III).

TrialBench's approval-forecasting FEATURES are design-time (arm counts, masking,
eligibility, sponsor class, ...) -- clean at the feature level, unlike CTO's
post-completion labeling functions. The question here is the SPLIT: the provided
train/test split is not temporal, so a model may be trained on trials that started
after much of its test set.

Per phase we fit the same GBDT under (a) the provided split and (b) a temporal split
ordered by NCT number (a registration-time proxy; earliest -> train, latest -> test,
same test size as provided). Because the two test sets differ, AUPRC alone is not
comparable across them: AUPRC scales with the test positive rate. We therefore report,
for each test set, the base rate, AUPRC, AUROC (prevalence-invariant) and AUPRC lift
over the base rate, each with a trial-level percentile bootstrap CI (1,000 resamples),
and the provided-minus-temporal difference of each with a CI from independent
bootstraps of the two test sets.

``--check-chronology`` additionally fetches trial start dates from ClinicalTrials.gov
(API v2, cached to data/trialbench/ctgov_start_dates.csv) and reports: provided-split
train vs test start years (one-sided Mann-Whitney U, H1: train starts earlier),
Spearman correlation between NCT number and start year (validating the proxy), and
the approval rate by start-year quartile (outcome maturation).

Data (downloaded on first use):
  features  Zenodo doi:10.5281/zenodo.14975339, trial-approval-forecasting.zip
  labels    github.com/ML2Health/ML2ClinicalTrials (train_y/test_y), pinned commit

Usage:  python scripts/audit_trialbench.py [--check-chronology] [--out outputs/trialbench_audit.json]
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_DIR = "data/trialbench"
X_BASE = f"{DATA_DIR}/trial-approval-forecasting"
Y_BASE = f"{DATA_DIR}/tb"
PHASES = ("Phase1", "Phase2", "Phase3")
ZENODO_ZIP = ("https://zenodo.org/api/records/14975339/files/"
              "trial-approval-forecasting.zip/content")
LABELS_COMMIT = "0694eba8f3f202be8dfa3fb2c2f05eddf7e538ce"
LABELS_BASE = (f"https://raw.githubusercontent.com/ML2Health/ML2ClinicalTrials/{LABELS_COMMIT}/"
               "Trialbench/data/trial-approval-forecasting")
CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
START_DATES_CSV = f"{DATA_DIR}/ctgov_start_dates.csv"
N_BOOT = 1000
CATS = ["sponsors/lead_sponsor/agency_class", "study_design_info/allocation",
        "study_design_info/intervention_model", "study_design_info/primary_purpose",
        "study_type", "phase", "eligibility/gender", "eligibility/healthy_volunteers"]


def ensure_data():
    """Download the TrialBench features (Zenodo) and labels (GitHub) if absent."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isdir(X_BASE):
        zpath = f"{DATA_DIR}/trial-approval-forecasting.zip"
        if not os.path.exists(zpath):
            print("  downloading TrialBench features from Zenodo ...")
            urllib.request.urlretrieve(ZENODO_ZIP, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(DATA_DIR)
    for ph in PHASES:
        for split in ("train_y.csv", "test_y.csv"):
            path = f"{Y_BASE}/{ph}/{split}"
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(f"{LABELS_BASE}/{ph}/{split}", path)


def _nct_num(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else -1


def _featurize(df):
    """Design-time features -> numeric matrix (numeric cols kept; low-card categoricals one-hot)."""
    ids = df["Unnamed: 0"].values
    body = df.drop(columns=["Unnamed: 0"])
    num = body.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in num.columns if num[c].notna().mean() > 0.5]
    X = num[keep].fillna(0.0)
    for c in CATS:
        if c in body.columns and c not in keep:
            d = pd.get_dummies(body[c].astype(str).str[:30], prefix=c.split("/")[-1][:10]).astype(float)
            X = pd.concat([X, d], axis=1)
    return X, ids


def _auprc(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")


def _auroc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def _fit(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xtr.values, ytr)
    return clf.predict_proba(Xte.values)[:, 1]


def _boot_draws(y, p, n=N_BOOT, seed=0):
    """Trial-level bootstrap draws of (AUPRC, AUROC, AUPRC - base rate)."""
    rng = np.random.default_rng(seed)
    n0 = len(y)
    out = {"auprc": [], "auroc": [], "auprc_lift": []}
    for _ in range(n):
        idx = rng.integers(0, n0, n0)
        yb, pb = y[idx], p[idx]
        if len(set(yb)) < 2:
            continue
        a = _auprc(yb, pb)
        out["auprc"].append(a)
        out["auroc"].append(_auroc(yb, pb))
        out["auprc_lift"].append(a - yb.mean())
    return {k: np.array(v) for k, v in out.items()}


def _q(a):
    return [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)]


def _evaluate(y, p, seed):
    base = float(y.mean())
    a = _auprc(y, p)
    draws = _boot_draws(y, p, seed=seed)
    res = {"n_test": int(len(y)), "test_base_rate": round(base, 4),
           "auprc": round(a, 4), "auroc": round(_auroc(y, p), 4),
           "auprc_lift": round(a - base, 4)}
    for k in ("auprc", "auroc", "auprc_lift"):
        res[f"{k}_ci"] = _q(draws[k])
    return res


def audit_phase(ph):
    xtr = pd.read_csv(f"{X_BASE}/{ph}/train_x.csv", low_memory=False)
    xte = pd.read_csv(f"{X_BASE}/{ph}/test_x.csv", low_memory=False)
    ytr = pd.read_csv(f"{Y_BASE}/{ph}/train_y.csv")
    yte = pd.read_csv(f"{Y_BASE}/{ph}/test_y.csv")
    xtr = xtr.merge(ytr, on="Unnamed: 0", how="inner")
    xte = xte.merge(yte, on="Unnamed: 0", how="inner")
    allrows = pd.concat([xtr, xte], ignore_index=True)
    y = allrows["outcome"].astype(int).values
    X, ids = _featurize(allrows.drop(columns=["outcome"]))
    n_test, n = len(xte), len(allrows)

    # (a) provided split: the last n_test rows are the provided test set.
    prov_test = np.zeros(n, bool)
    prov_test[len(xtr):] = True
    # (b) temporal split by NCT number (registration-time proxy): latest -> test.
    order = np.argsort([_nct_num(i) for i in ids])
    temp_test = np.zeros(n, bool)
    temp_test[order[-n_test:]] = True

    res, preds = {}, {}
    for name, mask, seed in (("provided_split", prov_test, 0), ("temporal_split", temp_test, 0)):
        p = _fit(X[~mask], y[~mask], X[mask])
        preds[name] = (y[mask], p)
        res[name] = _evaluate(y[mask], p, seed)
        res[name]["train_base_rate"] = round(float(y[~mask].mean()), 4)

    # Provided-minus-temporal differences. The test sets differ, so the two arms are
    # bootstrapped independently (different seeds) and differenced draw by draw.
    d_prov = _boot_draws(*preds["provided_split"], seed=0)
    d_temp = _boot_draws(*preds["temporal_split"], seed=1)
    diff = {}
    for k in ("auprc", "auroc", "auprc_lift"):
        m = min(len(d_prov[k]), len(d_temp[k]))
        diff[k] = {"point": round(res["provided_split"][k] - res["temporal_split"][k], 4),
                   "ci": _q(d_prov[k][:m] - d_temp[k][:m])}
    return {"phase": ph, "n": n, "n_test": n_test, "base_rate": round(float(y.mean()), 4),
            "overlap_provided_temporal_test": int((prov_test & temp_test).sum()),
            **res, "provided_minus_temporal": diff,
            "_ids": ids, "_y": y, "_prov_test": prov_test}


# --------------------------------------------------------------------------- #
# Chronology check (ClinicalTrials.gov start dates)
# --------------------------------------------------------------------------- #

def fetch_start_years(ncts, batch=200):
    """NCT id -> start year via ClinicalTrials.gov API v2 (batched; cached to CSV)."""
    cache = {}
    if os.path.exists(START_DATES_CSV):
        c = pd.read_csv(START_DATES_CSV)
        cache = dict(zip(c["nct_id"], c["start_year"]))
    todo = [x for x in dict.fromkeys(ncts) if x not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        params = {"filter.ids": ",".join(chunk), "fields": "NCTId,StartDate",
                  "pageSize": 1000, "format": "json"}
        url = CTGOV_URL + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    studies = json.load(r).get("studies", [])
                break
            except Exception:  # noqa: BLE001 -- transient network error: back off and retry
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"ClinicalTrials.gov request failed: {url[:120]}")
        for s in studies:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            date = (ps.get("statusModule", {}).get("startDateStruct") or {}).get("date")
            m = re.match(r"(\d{4})", date or "")
            cache[nct] = int(m.group(1)) if m else np.nan
        for x in chunk:  # ids the registry did not return (withdrawn / renamed)
            cache.setdefault(x, np.nan)
        time.sleep(0.2)
    pd.DataFrame({"nct_id": list(cache), "start_year": list(cache.values())}).to_csv(
        START_DATES_CSV, index=False)
    return cache


def chronology(phase_results):
    ncts = [str(i) for r in phase_results for i in r["_ids"]]
    years = fetch_start_years(ncts)
    out, pooled_tr, pooled_te, nct_all, yr_all = {}, [], [], [], []
    for r in phase_results:
        yr = np.array([years.get(str(i), np.nan) for i in r["_ids"]], dtype=float)
        ok = np.isfinite(yr)
        tr, te = yr[ok & ~r["_prov_test"]], yr[ok & r["_prov_test"]]
        mw = stats.mannwhitneyu(tr, te, alternative="less")
        # Approval rate by start-year quartile (all trials in the phase).
        q = pd.qcut(pd.Series(yr[ok]), 4, duplicates="drop")
        by_q = pd.Series(r["_y"][ok]).groupby(q.values, observed=True).mean()
        out[r["phase"]] = {
            "n_dated": int(ok.sum()), "n_total": int(len(yr)),
            "provided_train_median_start_year": float(np.median(tr)),
            "provided_test_median_start_year": float(np.median(te)),
            "mann_whitney_train_earlier_p": round(float(mw.pvalue), 4),
            "approval_rate_by_start_year_quartile": {str(k): round(float(v), 4)
                                                     for k, v in by_q.items()},
        }
        pooled_tr += list(tr)
        pooled_te += list(te)
        nct_all += [_nct_num(i) for i, k in zip(r["_ids"], ok) if k]
        yr_all += list(yr[ok])
    mw = stats.mannwhitneyu(pooled_tr, pooled_te, alternative="less")
    rho = stats.spearmanr(nct_all, yr_all)
    out["pooled"] = {
        "n_dated": len(yr_all),
        "provided_train_median_start_year": float(np.median(pooled_tr)),
        "provided_test_median_start_year": float(np.median(pooled_te)),
        "mann_whitney_train_earlier_p": round(float(mw.pvalue), 4),
        "spearman_nct_number_vs_start_year": round(float(rho.statistic), 4),
        "spearman_p": float(rho.pvalue),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Split-level temporal audit of TrialBench.")
    ap.add_argument("--check-chronology", action="store_true",
                    help="fetch ClinicalTrials.gov start dates (network; cached)")
    ap.add_argument("--out", default="outputs/trialbench_audit.json")
    args = ap.parse_args()
    ensure_data()

    phase_results = []
    for ph in PHASES:
        r = audit_phase(ph)
        phase_results.append(r)
        pv, tp, d = r["provided_split"], r["temporal_split"], r["provided_minus_temporal"]
        print(f"{ph}: n_test={r['n_test']}")
        for name, v in (("provided", pv), ("temporal", tp)):
            print(f"  {name:9s} base {v['test_base_rate']:.3f}  AUPRC {v['auprc']:.3f} {v['auprc_ci']}  "
                  f"AUROC {v['auroc']:.3f} {v['auroc_ci']}  lift {v['auprc_lift']:+.3f} {v['auprc_lift_ci']}")
        print(f"  provided - temporal: AUPRC {d['auprc']['point']:+.3f} {d['auprc']['ci']}  "
              f"AUROC {d['auroc']['point']:+.3f} {d['auroc']['ci']}  "
              f"lift {d['auprc_lift']['point']:+.3f} {d['auprc_lift']['ci']}")

    out = {
        "benchmark": "TrialBench approval forecasting (arXiv:2407.00631; Zenodo 14975339)",
        "labels_commit": LABELS_COMMIT,
        "design": ("provided (non-temporal) split vs NCT-ordered temporal split of equal test "
                   "size; trial-level percentile bootstrap CIs (1,000 resamples)"),
        "phases": [{k: v for k, v in r.items() if not k.startswith("_")} for r in phase_results],
    }
    for k in ("auprc", "auroc", "auprc_lift"):
        out[f"mean_provided_minus_temporal_{k}"] = round(float(np.mean(
            [r["provided_minus_temporal"][k]["point"] for r in phase_results])), 4)
    if args.check_chronology:
        out["chronology"] = chronology(phase_results)
        c = out["chronology"]["pooled"]
        print(f"\nchronology (pooled, n={c['n_dated']}): provided train median start "
              f"{c['provided_train_median_start_year']:.0f} vs test "
              f"{c['provided_test_median_start_year']:.0f}; Mann-Whitney (train earlier) "
              f"p={c['mann_whitney_train_earlier_p']}; Spearman(NCT, start) "
              f"rho={c['spearman_nct_number_vs_start_year']}")
        for ph in PHASES:
            print(f"  {ph} approval rate by start-year quartile: "
                  f"{out['chronology'][ph]['approval_rate_by_start_year_quartile']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nmean provided-minus-temporal: AUPRC {out['mean_provided_minus_temporal_auprc']:+.4f}  "
          f"AUROC {out['mean_provided_minus_temporal_auroc']:+.4f}  "
          f"AUPRC lift {out['mean_provided_minus_temporal_auprc_lift']:+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
