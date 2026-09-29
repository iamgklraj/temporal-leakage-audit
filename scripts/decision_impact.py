"""Decision-level impact of leakage: precision among the top-ranked 10%.

A portfolio screen or an early-warning list acts on the highest-ranked units, so we report the
precision (positive predictive value) among the top 10% of the test set, with and without
post-decision information, and its paired bootstrap CI:

  TrialBench (temporal test set): all features versus start-time features (enrollment and the
      three final-record city covariates removed), per phase;
  ICU records (PhysioNet 2012, test set C): measurements up to 48 h versus up to 24 h (decision
      at 24 h after admission).

Reuses the fitting code of scripts/audit_trialbench_v2.py and scripts/audit_physionet2012.py.
Usage:  python scripts/decision_impact.py [--out outputs/decision_impact.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_physionet2012 as E  # noqa: E402
import audit_trialbench as T  # noqa: E402
import audit_trialbench_v2 as V  # noqa: E402

TOP, N_BOOT, SEED = 0.10, 1000, 0


def ppv_top(y, p, frac=TOP):
    k = max(1, int(np.ceil(frac * len(y))))
    order = np.argsort(-p, kind="mergesort")      # stable: ties keep input order
    return float(y[order[:k]].mean())


def paired(y, p_leaky, p_deploy, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n, a, b = len(y), [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a.append(ppv_top(y[idx], p_leaky[idx]))
        b.append(ppv_top(y[idx], p_deploy[idx]))
    a, b = np.array(a), np.array(b)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]  # noqa: E731
    return {"ppv_top10_with_leakage": round(ppv_top(y, p_leaky), 4), "ci_with_leakage": q(a),
            "ppv_top10_deployable": round(ppv_top(y, p_deploy), 4), "ci_deployable": q(b),
            "difference": round(ppv_top(y, p_leaky) - ppv_top(y, p_deploy), 4), "difference_ci": q(a - b),
            "base_rate": round(float(y.mean()), 4), "n_test": int(n), "n_top": int(np.ceil(TOP * n))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/decision_impact.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"metric": "precision (positive predictive value) among the top-ranked 10% of the test set",
           "bootstrap": f"paired percentile bootstrap, {N_BOOT} resamples, seed {SEED}", "trialbench": {}}
    T.ensure_data()
    for ph in T.PHASES:
        P = V.load_phase(ph)
        fs = V.feature_sets(P["X"])
        mask = P["temp_test"]
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        y = P["y"][te]
        p_all = V.fit_cols(P, fs["all_features"], tr, te)
        p_start = V.fit_cols(P, fs["ablated_non_start_removed"], tr, te)
        out["trialbench"][ph] = paired(y, p_all, p_start)
        print(ph, out["trialbench"][ph], flush=True)
    E.download()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    yy = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    X24 = E.features(long, static, 24 * 60, params)
    X48 = E.features(long, static, 48 * 60, params)
    p24 = E.fit_predict(X24[is_tr], yy[is_tr], X24[~is_tr])
    p48 = E.fit_predict(X48[is_tr], yy[is_tr], X48[~is_tr])
    out["icu_physionet2012_decision_24h"] = paired(yy[~is_tr], p48, p24)
    print("ICU", out["icu_physionet2012_decision_24h"], flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
