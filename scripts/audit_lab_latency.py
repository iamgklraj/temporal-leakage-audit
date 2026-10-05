"""Availability-time censoring of ICU measurements, with reporting latencies from MIMIC-IV.

A measurement's observation time is the valid time of the measured value, but the value becomes
available to clinicians and decision support only once it is stored in the record. The PhysioNet/
Computing in Cardiology Challenge 2012 data (audit_physionet2012.py) carry observation times only. The
MIMIC-IV Clinical Database Demo (100 patients; open access, Open Data Commons Open Database License
v1.0; doi:10.13026/dp1f-ex47; Beth Israel Deaconess Medical Center, the institution whose MIMIC-II
records underlie the 2012 challenge) records both: charttime, when a specimen was drawn or a value
observed, and storetime, when the value was stored. For every PhysioNet 2012 variable we take the
empirical distribution of storetime - charttime in the demo (laboratory values: hosp/labevents, blood
specimens; bedside values: icu/chartevents; urine output: icu/outputevents; negative latencies set to
zero and counted), draw one latency for every PhysioNet 2012 measurement of that variable, and censor by
availability time (observation time + latency) instead of observation time.

Design (fixed before fitting). Decision 24 hours after ICU admission (12 hours as a sensitivity
analysis). Deployable matrices: valid-time censoring (measurements observed by t; identical to
audit_physionet2012.py) and availability-time censoring (measurements available by t). Naive matrix: the
full 48-hour record. Features, learner (models.fit_gbdt), split (train A + B, test C) and the
patient-level paired percentile bootstrap (1,000 resamples, seed 0) are those of audit_physionet2012.py.
Reported: AUPRC and AUROC of the three models; LAP relative to each deployable matrix; and the
difference between the two deployable models (valid-time minus availability-time AUPRC), the leakage
that valid-time censoring admits. Latency draws use seed 0 for the reported estimate and are repeated for
seeds 1-19 to show their spread. Troponin I uses the Troponin T latency (the demo has no Troponin I) and
mechanical ventilation the pooled latency of the bedside variables.

Usage:  python scripts/audit_lab_latency.py [--out outputs/lab_latency_sensitivity.json]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_physionet2012 as E  # noqa: E402
from temporal_leakage_audit import metrics as M  # noqa: E402

DEMO = "data/mimic_iv_demo"
DEMO_URL = "https://physionet.org/files/mimic-iv-demo/2.2/"
DEMO_FILES = ["hosp/labevents.csv.gz", "hosp/d_labitems.csv.gz", "icu/chartevents.csv.gz",
              "icu/d_items.csv.gz", "icu/outputevents.csv.gz"]
N_SEEDS = 20
DESIGNS = {"t24": 24, "t12": 12}

LAB = {"ALP": ["Alkaline Phosphatase"], "ALT": ["Alanine Aminotransferase (ALT)"],
       "AST": ["Asparate Aminotransferase (AST)"], "Albumin": ["Albumin"], "BUN": ["Urea Nitrogen"],
       "Bilirubin": ["Bilirubin, Total"], "Cholesterol": ["Cholesterol, Total"], "Creatinine": ["Creatinine"],
       "Glucose": ["Glucose"], "HCO3": ["Bicarbonate"], "HCT": ["Hematocrit"], "K": ["Potassium"],
       "Lactate": ["Lactate"], "Mg": ["Magnesium"], "Na": ["Sodium"], "PaCO2": ["pCO2"], "PaO2": ["pO2"],
       "Platelets": ["Platelet Count"], "SaO2": ["Oxygen Saturation"], "TroponinI": ["Troponin T"],
       "TroponinT": ["Troponin T"], "WBC": ["White Blood Cells"], "pH": ["pH"]}
BEDSIDE = {"HR": ["Heart Rate"], "RespRate": ["Respiratory Rate"], "MAP": ["Arterial Blood Pressure mean"],
           "SysABP": ["Arterial Blood Pressure systolic"], "DiasABP": ["Arterial Blood Pressure diastolic"],
           "NIMAP": ["Non Invasive Blood Pressure mean"], "NISysABP": ["Non Invasive Blood Pressure systolic"],
           "NIDiasABP": ["Non Invasive Blood Pressure diastolic"],
           "Temp": ["Temperature Fahrenheit", "Temperature Celsius"],
           "GCS": ["GCS - Eye Opening", "GCS - Verbal Response", "GCS - Motor Response"],
           "FiO2": ["Inspired O2 Fraction"], "Weight": ["Daily Weight"]}
URINE_LABELS = ["Foley", "Void", "Condom Cath", "Straight Cath", "Suprapubic", "Urine"]


def download():
    for f in DEMO_FILES:
        path = os.path.join(DEMO, f)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(DEMO_URL + f, path)


def latency_table():
    """Empirical latencies (minutes) per PhysioNet 2012 variable from the MIMIC-IV demo."""
    lab = pd.read_csv(os.path.join(DEMO, "hosp/labevents.csv.gz"),
                      usecols=["itemid", "charttime", "storetime"], parse_dates=["charttime", "storetime"])
    dl = pd.read_csv(os.path.join(DEMO, "hosp/d_labitems.csv.gz"), usecols=["itemid", "label", "fluid"])
    lab = lab.merge(dl, on="itemid", how="left")
    lab = lab[lab["fluid"] == "Blood"]
    di = pd.read_csv(os.path.join(DEMO, "icu/d_items.csv.gz"), usecols=["itemid", "label"])
    ch = pd.read_csv(os.path.join(DEMO, "icu/chartevents.csv.gz"),
                     usecols=["itemid", "charttime", "storetime"], parse_dates=["charttime", "storetime"])
    ch = ch.merge(di, on="itemid", how="left")
    oe = pd.read_csv(os.path.join(DEMO, "icu/outputevents.csv.gz"),
                     usecols=["itemid", "charttime", "storetime"], parse_dates=["charttime", "storetime"])
    oe = oe.merge(di, on="itemid", how="left")
    out, summary = {}, {}

    def add(var, frame, labels, source):
        d = frame[frame["label"].isin(labels)].dropna(subset=["charttime", "storetime"])
        lat = (d["storetime"] - d["charttime"]).dt.total_seconds().values / 60.0
        neg = float(np.mean(lat < 0)) if len(lat) else 0.0
        lat = np.clip(lat, 0, None)
        out[var] = lat
        summary[var] = {"source": source, "labels": labels, "n": int(len(lat)),
                        "share_negative_set_to_zero": round(neg, 4),
                        "median_hours": round(float(np.median(lat)) / 60, 3),
                        "q1_hours": round(float(np.quantile(lat, 0.25)) / 60, 3),
                        "q3_hours": round(float(np.quantile(lat, 0.75)) / 60, 3),
                        "share_over_1h": round(float(np.mean(lat > 60)), 4),
                        "share_over_6h": round(float(np.mean(lat > 360)), 4)}

    for var, labels in LAB.items():
        add(var, lab, labels, "hosp/labevents (blood)")
    for var, labels in BEDSIDE.items():
        add(var, ch, labels, "icu/chartevents")
    urine = oe[oe["label"].fillna("").str.contains("|".join(URINE_LABELS), case=False)]
    add("Urine", urine.assign(label="urine"), ["urine"], "icu/outputevents (urine)")
    pooled = np.concatenate([out[v] for v in BEDSIDE])
    out["MechVent"] = pooled
    summary["MechVent"] = {"source": "pooled bedside latencies (no direct item)", "n": int(len(pooled)),
                           "median_hours": round(float(np.median(pooled)) / 60, 3),
                           "share_over_1h": round(float(np.mean(pooled > 60)), 4)}
    summary["TroponinI"]["note"] = "Troponin T latency used (no Troponin I in the demo)"
    return out, summary


def availability_minutes(long, lat, seed):
    """Observation minute plus a latency drawn from the variable's empirical distribution."""
    rng = np.random.default_rng(seed)
    avail = long["minute"].values.astype(float).copy()
    for var, values in lat.items():
        m = (long["param"] == var).values
        if m.any() and len(values):
            avail[m] = avail[m] + rng.choice(values, size=int(m.sum()), replace=True)
    return avail


def features_available(long, static, avail, cutoff_min, params):
    """audit_physionet2012.features with measurements admitted by availability time."""
    return E.features(long.loc[avail <= cutoff_min], static, 10 ** 9, params)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/lab_latency_sensitivity.json")
    args = ap.parse_args()
    t0 = time.time()
    download()
    E.download()
    lat, lat_summary = latency_table()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    y = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    yte = y[~is_tr]
    X48 = E.features(long, static, 48 * 60, params)
    p48 = E.fit_predict(X48[is_tr], y[is_tr], X48[~is_tr])
    out = {"script": "scripts/audit_lab_latency.py",
           "latency_source": "MIMIC-IV Clinical Database Demo v2.2 (doi:10.13026/dp1f-ex47; ODbL 1.0): "
                             "storetime - charttime; blood laboratory values from hosp/labevents, bedside values "
                             "from icu/chartevents, urine from icu/outputevents; negative latencies set to zero",
           "latency_by_variable": lat_summary,
           "design": "deployable matrices censored by observation time (valid time) or by availability time "
                     "(observation + sampled latency); naive = full 48-hour record; learner, features, split and "
                     "patient-level paired bootstrap as audit_physionet2012.py",
           "n_latency_seeds": N_SEEDS}
    for key, t_h in DESIGNS.items():
        cut = t_h * 60
        Xv = E.features(long, static, cut, params)
        pv = E.fit_predict(Xv[is_tr], y[is_tr], Xv[~is_tr])
        seeds = []
        pa0 = None
        share_late = None
        for s in range(N_SEEDS):
            avail = availability_minutes(long, lat, s)
            Xa = features_available(long, static, avail, cut, params)
            pa = E.fit_predict(Xa[is_tr], y[is_tr], Xa[~is_tr])
            seeds.append(M.auprc(yte, pv) - M.auprc(yte, pa))
            if s == 0:
                pa0 = pa
                obs = long["minute"].values <= cut
                late = obs & (avail > cut)
                share_late = {"overall": round(float(late.sum() / obs.sum()), 4),
                              "by_variable": {v: round(float(late[(long["param"] == v).values].sum()
                                                             / max(1, obs[(long["param"] == v).values].sum())), 4)
                                              for v in params}}
        boot = E.paired_boot(yte, {"valid": pv, "avail": pa0, "naive": p48})
        res = {"decision_time_hours": t_h, "test_base_rate": round(float(yte.mean()), 4),
               "share_of_measurements_observed_by_t_but_available_after_t": share_late}
        for k, p in (("valid_time_deployable", pv), ("availability_time_deployable", pa0), ("naive_48h", p48)):
            kk = {"valid_time_deployable": "valid", "availability_time_deployable": "avail", "naive_48h": "naive"}[k]
            res[k] = {"auprc": round(M.auprc(yte, p), 4), "auprc_ci": E.ci(boot[kk]["auprc"]),
                      "auroc": round(M.auroc(yte, p), 4), "auroc_ci": E.ci(boot[kk]["auroc"])}

        def diff(a, b):
            pa_, pb_ = {"valid": pv, "avail": pa0, "naive": p48}[a], {"valid": pv, "avail": pa0, "naive": p48}[b]
            d = boot[a]["auprc"] - boot[b]["auprc"]
            return {"auprc": round(M.auprc(yte, pa_) - M.auprc(yte, pb_), 4), "auprc_ci": E.ci(d),
                    "auprc_p_gt_0": round(float(np.mean(d > 0)), 3)}

        res["LAP_valid_time"] = diff("naive", "valid")
        res["LAP_availability_time"] = diff("naive", "avail")
        res["valid_minus_availability_deployable"] = diff("valid", "avail")
        res["valid_minus_availability_over_latency_seeds"] = {
            "mean": round(float(np.mean(seeds)), 4), "min": round(float(np.min(seeds)), 4),
            "max": round(float(np.max(seeds)), 4), "values": [round(float(x), 4) for x in seeds]}
        out[key] = res
        print(key, "valid", res["valid_time_deployable"]["auprc"], "avail", res["availability_time_deployable"]["auprc"],
              "diff", res["valid_minus_availability_deployable"], "seeds", res["valid_minus_availability_over_latency_seeds"]["mean"],
              "late share", share_late["overall"], flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
