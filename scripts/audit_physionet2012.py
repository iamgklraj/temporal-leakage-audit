"""Patient-level demonstration of the leakage-response instrument on ICU records.

Data: PhysioNet/Computing in Cardiology Challenge 2012 (open access, Open Data Commons
Attribution License v1.0): 12,000 ICU stays with time-stamped measurements over the first
48 hours and in-hospital mortality (sets A, B and C with their outcome files).

A model intended for use at decision time t after ICU admission (24 h; 12 h as a sensitivity
analysis) may only use measurements recorded by t. The instrument refits the same learner as
measurements recorded up to t + h are admitted and evaluates it on the same test stays; LAP is
the AUPRC with the full 48-hour record (naive) minus the AUPRC with data up to t (deployable),
with a paired patient-level bootstrap. Features per variable: last, minimum, maximum and mean
value and the number of measurements (a care-process signal), plus admission descriptors.
Training: sets A and B (8,000 stays); test: set C (4,000 stays).

Usage:  python scripts/audit_physionet2012.py [--out outputs/physionet2012_audit.json]
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from temporal_leakage_audit import metrics as M  # noqa: E402
from temporal_leakage_audit import models  # noqa: E402

BASE = "https://physionet.org/files/challenge-2012/1.0.0/"
DATA_DIR = "data/physionet2012"
SETS = ["a", "b", "c"]
STATIC = ["Age", "Gender", "Height", "ICUType"]          # general descriptors (Weight is also a series)
N_BOOT, SEED = 1000, 0
DESIGNS = {"t24": (24, [0, 6, 12, 18, 24]), "t12": (12, [0, 6, 12, 24, 36])}  # (t hours, horizons h)
AGGS = ["last", "min", "max", "mean", "count"]


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for s in SETS:
        for name in (f"set-{s}.tar.gz", f"Outcomes-{s}.txt"):
            path = os.path.join(DATA_DIR, name)
            if not os.path.exists(path):
                print("downloading", name, flush=True)
                urllib.request.urlretrieve(BASE + name, path)


def load_records():
    """Long table (record_id, minute, param, value), static descriptors and outcomes."""
    rows, static = [], []
    for s in SETS:
        with tarfile.open(os.path.join(DATA_DIR, f"set-{s}.tar.gz")) as tar:
            for m in tar.getmembers():
                if not m.name.endswith(".txt"):
                    continue
                df = pd.read_csv(io.BytesIO(tar.extractfile(m).read()))
                df = df.dropna(subset=["Time", "Parameter", "Value"])   # skip blank lines
                rid = int(df.loc[df["Parameter"] == "RecordID", "Value"].iloc[0])
                hh, mm = df["Time"].str.split(":", expand=True).astype(int).T.values
                df["minute"] = hh * 60 + mm
                st = {"record_id": rid, "set": s}
                at0 = df[df["minute"] == 0]
                for p in STATIC + ["Weight"]:
                    v = at0.loc[at0["Parameter"] == p, "Value"]
                    st[p if p != "Weight" else "Weight_admission"] = (
                        float(v.iloc[0]) if len(v) and float(v.iloc[0]) >= 0 else np.nan)
                static.append(st)
                ts = df[~df["Parameter"].isin(["RecordID"] + STATIC)].copy()
                # admission weight at 00:00 is a descriptor; later Weight entries are measurements
                ts = ts[~((ts["Parameter"] == "Weight") & (ts["minute"] == 0))]
                ts = ts[ts["Value"] >= 0]                 # -1 marks missing
                ts["record_id"] = rid
                rows.append(ts[["record_id", "minute", "Parameter", "Value"]])
    long = pd.concat(rows, ignore_index=True).rename(columns={"Parameter": "param", "Value": "value"})
    static = pd.DataFrame(static).set_index("record_id")
    outc = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f"Outcomes-{s}.txt")) for s in SETS])
    outc = outc.set_index("RecordID")["In-hospital_death"]
    return long, static, outc


def features(long, static, cutoff_min, params):
    """Per-record aggregates of measurements recorded no later than cutoff_min."""
    d = long[long["minute"] <= cutoff_min].sort_values(["record_id", "minute"])
    g = d.groupby(["record_id", "param"])["value"]
    agg = pd.concat({"last": g.last(), "min": g.min(), "max": g.max(), "mean": g.mean(),
                     "count": g.size()}, axis=1)
    wide = agg.unstack("param")
    wide.columns = [f"{p}_{a}" for a, p in wide.columns]
    cols = [f"{p}_{a}" for p in params for a in AGGS]
    wide = wide.reindex(index=static.index, columns=cols)
    wide[[c for c in cols if c.endswith("_count")]] = wide[[c for c in cols if c.endswith("_count")]].fillna(0)
    X = static[STATIC + ["Weight_admission"]].join(wide)
    X["ICUType"] = X["ICUType"].astype(float)
    return X


def paired_boot(y, preds, n_boot=N_BOOT, seed=SEED):
    """Patient-level bootstrap of AUPRC and AUROC on shared resamples (aligned draws)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {k: {"auprc": [], "auroc": []} for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, p in preds.items():
            out[k]["auprc"].append(M.auprc(yb, p[idx]))
            out[k]["auroc"].append(M.auroc(yb, p[idx]))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def ci(a):
    return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)]


def fit_predict(Xtr, ytr, Xte):
    clf = models.fit_gbdt(Xtr, ytr, seed=SEED)
    return models.predict_gbdt(clf, Xte)


def run_design(long, static, y, is_tr, params, t_hours, horizons):
    Xs = {h: features(long, static, (t_hours + h) * 60, params) for h in horizons}
    ytr, yte = y[is_tr], y[~is_tr]
    preds = {h: fit_predict(Xs[h][is_tr], ytr, Xs[h][~is_tr]) for h in horizons}
    h0, hN = horizons[0], horizons[-1]
    # per-source placebo: admit one feature group at the full record, keep the rest deployable
    groups = {"measurement_counts": [c for c in Xs[h0].columns if c.endswith("_count")],
              "measured_values": [c for c in Xs[h0].columns
                                  if any(c.endswith("_" + a) for a in ("last", "min", "max", "mean"))]}
    for gname, cols in groups.items():
        Xg = Xs[h0].copy()
        Xg[cols] = Xs[hN][cols]
        preds[f"placebo:{gname}"] = fit_predict(Xg[is_tr], ytr, Xg[~is_tr])
    # per-variable placebo (exploratory): one variable's features at the full record
    for p in params:
        cols = [f"{p}_{a}" for a in AGGS]
        Xg = Xs[h0].copy()
        Xg[cols] = Xs[hN][cols]
        preds[f"var:{p}"] = fit_predict(Xg[is_tr], ytr, Xg[~is_tr])
    boot = paired_boot(yte, preds)
    base = float(yte.mean())
    curve = []
    for h in horizons:
        curve.append({"h_hours": h, "data_up_to_hours": t_hours + h,
                      "auprc": round(M.auprc(yte, preds[h]), 4), "auprc_ci": ci(boot[h]["auprc"]),
                      "auroc": round(M.auroc(yte, preds[h]), 4), "auroc_ci": ci(boot[h]["auroc"]),
                      "lift": round(M.auprc(yte, preds[h]) - base, 4)})

    def contrast(k):
        return {"auprc": round(M.auprc(yte, preds[k]) - M.auprc(yte, preds[h0]), 4),
                "auprc_ci": ci(boot[k]["auprc"] - boot[h0]["auprc"]),
                "auprc_p_gt_0": round(float(np.mean(boot[k]["auprc"] - boot[h0]["auprc"] > 0)), 3),
                "auroc": round(M.auroc(yte, preds[k]) - M.auroc(yte, preds[h0]), 4),
                "auroc_ci": ci(boot[k]["auroc"] - boot[h0]["auroc"])}

    res = {"decision_time_hours": t_hours, "horizons_hours": horizons, "test_base_rate": round(base, 4),
           "curve": curve, "LAP": contrast(hN),
           "per_source_placebo": {g: contrast(f"placebo:{g}") for g in groups},
           "per_variable_placebo_exploratory": dict(sorted(
               ((p, contrast(f"var:{p}")) for p in params), key=lambda kv: -kv[1]["auprc"]))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/physionet2012_audit.json")
    args = ap.parse_args()
    t0 = time.time()
    download()
    long, static, outc = load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    y = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    print(f"stays={len(ids)} train={is_tr.sum()} test={(~is_tr).sum()} variables={len(params)} "
          f"measurements={len(long)} death_rate={y.mean():.3f}", flush=True)
    out = {"dataset": "PhysioNet/CinC Challenge 2012 (physionet.org/content/challenge-2012/1.0.0; ODC-By 1.0)",
           "script": "scripts/audit_physionet2012.py",
           "outcome": "in-hospital death",
           "split": "train = sets A and B; test = set C",
           "n_train": int(is_tr.sum()), "n_test": int((~is_tr).sum()),
           "n_train_deaths": int(y[is_tr].sum()), "n_test_deaths": int(y[~is_tr].sum()),
           "n_variables": len(params), "variables": params,
           "features": "per variable: last, min, max, mean, count of measurements recorded by the cutoff; "
                       "admission age, gender, height, ICU type and weight",
           "learner": "models.fit_gbdt (HistGradientBoosting, max_depth 3, lr 0.06, 300 iter, l2 1.0, seed 0)",
           "bootstrap": f"patient-level paired percentile bootstrap, {N_BOOT} resamples, seed {SEED}"}
    for key, (t_hours, horizons) in DESIGNS.items():
        out[key] = run_design(long, static, y, is_tr, params, t_hours, horizons)
        r = out[key]
        print(f"{key}: deployable AUPRC {r['curve'][0]['auprc']} naive {r['curve'][-1]['auprc']} "
              f"LAP {r['LAP']['auprc']:+.4f} {r['LAP']['auprc_ci']}", flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
