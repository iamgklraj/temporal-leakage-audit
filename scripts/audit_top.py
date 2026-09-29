"""Chronology and field audit of the TOP benchmark (HINT; Fu et al., Patterns 2022).

TOP's released phase-specific train/valid/test files (github.com/futianfan/clinical-trial-outcome-
prediction, data snapshot of ClinicalTrials.gov on 20 February 2021) carry no dates. This script
(1) retrieves each trial's start and primary-completion dates from the ClinicalTrials.gov API v2
(cached), (2) checks whether the provided split is temporal (median dates per split; one-sided
Mann-Whitney tests that training trials start and complete earlier than test trials; share of
test trials that start before the latest training start), and (3) tabulates the label against
the post-completion fields distributed in the same files (final status and reason for stopping).
No model is fitted: TOP's reference models use only drugs, diseases and eligibility criteria.

Usage:  python scripts/audit_top.py [--out outputs/top_audit.json]
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RAW = "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/"
DATA_DIR = "data/top"
PHASES = ["I", "II", "III"]
SPLITS = ["train", "valid", "test"]
CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
DATES_CSV = os.path.join(DATA_DIR, "ctgov_dates.csv")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for ph in PHASES:
        for sp in SPLITS:
            name = f"phase_{ph}_{sp}.csv"
            path = os.path.join(DATA_DIR, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(RAW + name, path)


def _date(struct):
    d = (struct or {}).get("date")
    return pd.to_datetime(d, errors="coerce") if d else pd.NaT


def fetch_dates(ncts, batch=200):
    cache = pd.read_csv(DATES_CSV, parse_dates=["start", "primary_completion"]) if os.path.exists(DATES_CSV) \
        else pd.DataFrame(columns=["nct_id", "start", "primary_completion"])
    have = set(cache["nct_id"])
    todo = [x for x in dict.fromkeys(ncts) if x not in have]
    rows = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        params = {"filter.ids": ",".join(chunk), "fields": "NCTId,StartDate,PrimaryCompletionDate",
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
        got = set()
        for s in studies:
            sm = s.get("protocolSection", {}).get("statusModule", {})
            nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            got.add(nct)
            rows.append({"nct_id": nct, "start": _date(sm.get("startDateStruct")),
                         "primary_completion": _date(sm.get("primaryCompletionDateStruct"))})
        rows += [{"nct_id": x, "start": pd.NaT, "primary_completion": pd.NaT} for x in chunk if x not in got]
        time.sleep(0.2)
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(DATES_CSV, index=False)
    return cache.set_index("nct_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/top_audit.json")
    args = ap.parse_args()
    download()
    frames = []
    for ph in PHASES:
        for sp in SPLITS:
            d = pd.read_csv(os.path.join(DATA_DIR, f"phase_{ph}_{sp}.csv"), usecols=["nctid", "status", "why_stop", "label"])
            d["phase"], d["split"] = ph, sp
            frames.append(d)
    top = pd.concat(frames, ignore_index=True)
    dates = fetch_dates(top["nctid"].tolist())
    top = top.join(dates, on="nctid")
    for col in ("start", "primary_completion"):
        top[col] = pd.to_datetime(top[col], errors="coerce")
    top["start_year"] = top["start"].dt.year + (top["start"].dt.dayofyear - 1) / 365.25
    top["completion_year"] = top["primary_completion"].dt.year + (top["primary_completion"].dt.dayofyear - 1) / 365.25
    out = {"benchmark": "TOP (HINT; github.com/futianfan/clinical-trial-outcome-prediction; snapshot 2021-02-20)",
           "script": "scripts/audit_top.py", "n_trials": int(len(top)),
           "dates_found_fraction": round(float(top["start"].notna().mean()), 4), "phases": {}}
    for ph in PHASES:
        t = top[top["phase"] == ph]
        res = {}
        for sp in SPLITS:
            s = t[t["split"] == sp]
            res[sp] = {"n": int(len(s)), "label_rate": round(float(s["label"].mean()), 4),
                       "median_start_year": round(float(s["start_year"].median()), 2),
                       "median_completion_year": round(float(s["completion_year"].median()), 2),
                       "start_year_range": [round(float(s["start_year"].min()), 2), round(float(s["start_year"].max()), 2)],
                       "completion_year_range": [round(float(s["completion_year"].min()), 2),
                                                 round(float(s["completion_year"].max()), 2)]}
        tr, te = t[t["split"] == "train"], t[t["split"] == "test"]
        for col in ("start_year", "completion_year"):
            a, b = tr[col].dropna(), te[col].dropna()
            res[f"mannwhitney_train_earlier_{col}_p"] = float(mannwhitneyu(a, b, alternative="less").pvalue)
        res["test_start_before_latest_train_start_fraction"] = round(
            float((te["start_year"] < tr["start_year"].max()).mean()), 4)
        res["test_completion_after_latest_train_completion_fraction"] = round(
            float((te["completion_year"] > tr["completion_year"].max()).mean()), 4)
        first_test_start = te["start_year"].min()
        res["train_completion_after_first_test_start_fraction"] = round(
            float((tr["completion_year"] > first_test_start).mean()), 4)
        res["label_by_status"] = {st: {"n": int(len(g)), "label_1": int(g["label"].sum())}
                                  for st, g in t.groupby("status")}
        res["why_stop_nonempty_label_1"] = int(t.loc[t["why_stop"].notna(), "label"].sum())
        res["why_stop_nonempty_n"] = int(t["why_stop"].notna().sum())
        out["phases"][ph] = res
        print(ph, {sp: (res[sp]["n"], res[sp]["median_start_year"], res[sp]["median_completion_year"]) for sp in SPLITS},
              "P(start)=%.3g P(completion)=%.3g" % (res["mannwhitney_train_earlier_start_year_p"],
                                                    res["mannwhitney_train_earlier_completion_year_p"]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
