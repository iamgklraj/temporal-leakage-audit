"""Feature-level LAP of TrialBench's post-start features, with the test set held fixed.

Applies the instrument's fixed-test-set logic to the four non-start-time features identified in
scripts/audit_trialbench_v2.py (enrollment = actual enrollment; city Aging/GDP/Population =
covariates of final-record facility cities). For each phase, on the TEMPORAL split and separately
on the PROVIDED split, the default learner (audit_trialbench._fit) is fitted on the same training
set with
  (a) all features,
  (b) the four non-start-time features removed ("ablated"),
  (c) only 'enrollment' removed (city covariates kept),
and all arms are evaluated on the SAME test set. Reported per arm: AUPRC, AUROC, lift (AUPRC minus
test base rate). Paired differences (trial-level PAIRED bootstrap, 1,000 resamples, seed 0; the
same resampled rows for every arm):
  LAP_total      = all - ablated           (feature-level LAP of the four post-start features)
  LAP_enrollment = all - minus_enrollment  (enrollment's share, city covariates present)
  LAP_city       = minus_enrollment - ablated (city covariates' share, enrollment absent)
LAP_total = LAP_enrollment + LAP_city exactly (point estimates and per-resample draws). On a fixed
test set the lift difference equals the AUPRC difference.

Reuses scripts/audit_trialbench.py and scripts/audit_trialbench_v2.py unchanged.

Usage:  python scripts/audit_trialbench_v2_lap.py [--out outputs/trialbench_lap_poststart.json]
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

import audit_trialbench as T  # noqa: E402
import audit_trialbench_v2 as V  # noqa: E402

N_BOOT = 1000
SEED = 0
ARMS = ("all_features", "ablated_non_start_removed", "minus_enrollment_only")
CONTRASTS = {"LAP_total_all_minus_ablated": ("ablated_non_start_removed", "all_features"),
             "LAP_enrollment_all_minus_minus_enrollment": ("minus_enrollment_only", "all_features"),
             "LAP_city_minus_enrollment_minus_ablated": ("ablated_non_start_removed", "minus_enrollment_only")}


def paired_draws(y, preds, n_boot=N_BOOT, seed=SEED):
    """Trial-level paired bootstrap: same resampled rows for every arm; AUPRC, AUROC, base rate."""
    rng = np.random.default_rng(seed)
    d = {k: {"auprc": [], "auroc": [], "base": []} for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        b = yb.mean()
        for k, p in preds.items():
            d[k]["auprc"].append(T._auprc(yb, p[idx]))
            d[k]["auroc"].append(T._auroc(yb, p[idx]))
            d[k]["base"].append(b)
    return {k: {m: np.array(v) for m, v in dd.items()} for k, dd in d.items()}


def main():
    ap = argparse.ArgumentParser(description="Feature-level LAP of TrialBench post-start features.")
    ap.add_argument("--out", default="outputs/trialbench_lap_poststart.json")
    args = ap.parse_args()
    t0 = time.time()
    T.ensure_data()  # no-op when present
    v2 = json.load(open("outputs/trialbench_audit_v2.json"))["T2_provided_vs_temporal"]

    out = {"benchmark": "TrialBench approval forecasting (arXiv:2407.00631; Zenodo 14975339)",
           "script": "scripts/audit_trialbench_v2_lap.py",
           "settings": {"learner": "audit_trialbench._fit (HGB default, seed 0)",
                        "bootstrap": f"trial-level PAIRED percentile bootstrap on the fixed test set, "
                                     f"{N_BOOT} resamples, seed {SEED} (same rows for every arm)",
                        "non_start_time_features": V.NON_START,
                        "arms": {"all_features": "all model features",
                                 "ablated_non_start_removed": "enrollment + city Aging/GDP/Population removed",
                                 "minus_enrollment_only": "enrollment removed, city covariates kept"},
                        "contrasts": {k: f"{b} minus {a}" for k, (a, b) in CONTRASTS.items()},
                        "note": "fixed test set, so lift difference == AUPRC difference"},
           "phases": {}}
    for ph in T.PHASES:
        P = V.load_phase(ph)
        fs = V.feature_sets(P["X"])
        y = P["y"]
        res_ph = {}
        for split, mask in (("temporal_split", P["temp_test"]), ("provided_split", P["prov_test"])):
            tr, te = np.where(~mask)[0], np.where(mask)[0]
            yte = y[te]
            preds = {a: V.fit_cols(P, fs[a], tr, te) for a in ARMS}
            d = paired_draws(yte, preds)
            base = float(yte.mean())
            arms = {}
            for a in ARMS:
                ap_ = T._auprc(yte, preds[a])
                arms[a] = {"n_features": len(fs[a]),
                           "auprc": V.r4(ap_), "auprc_ci": V.q(d[a]["auprc"]),
                           "auroc": V.r4(T._auroc(yte, preds[a])), "auroc_ci": V.q(d[a]["auroc"]),
                           "lift": V.r4(ap_ - base), "lift_ci": V.q(d[a]["auprc"] - d[a]["base"])}
            contr = {}
            for name, (a, b) in CONTRASTS.items():
                c = {}
                for m, f in (("auprc", T._auprc), ("auroc", T._auroc)):
                    dd = d[b][m] - d[a][m]
                    c[m] = {"point": V.r4(f(yte, preds[b]) - f(yte, preds[a])), "ci": V.q(dd),
                            "p_gt_0": V.r4(np.mean(dd > 0))}
                contr[name] = c
            lap_auprc = T._auprc(yte, preds["all_features"]) - T._auprc(yte, preds["ablated_non_start_removed"])
            lift_all = T._auprc(yte, preds["all_features"]) - base
            # consistency with the T2 arm estimates in outputs/trialbench_audit_v2.json
            consistent = all(arms[a]["auprc"] == v2[ph][a][split]["auprc"]
                             and arms[a]["auroc"] == v2[ph][a][split]["auroc"] for a in ARMS)
            res_ph[split] = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_base_rate": V.r4(base),
                             "arms": arms, "paired_differences": contr,
                             "share_of_all_feature_lift_that_is_LAP_total": V.r4(lap_auprc / lift_all),
                             "matches_trialbench_audit_v2_T2_point_estimates": bool(consistent)}
            print(f"{ph} {split}: all {arms['all_features']['auroc']}/{arms['all_features']['auprc']}  "
                  f"ablated {arms['ablated_non_start_removed']['auroc']}/{arms['ablated_non_start_removed']['auprc']}  "
                  f"LAP AUROC {contr['LAP_total_all_minus_ablated']['auroc']['point']} "
                  f"AUPRC {contr['LAP_total_all_minus_ablated']['auprc']['point']}  consistent={consistent}",
                  flush=True)
        out["phases"][ph] = res_ph
    for split in ("temporal_split", "provided_split"):
        for name in CONTRASTS:
            for m in ("auprc", "auroc"):
                out[f"mean_{name}_{m}_{split}"] = V.r4(np.mean(
                    [out["phases"][ph][split]["paired_differences"][name][m]["point"] for ph in T.PHASES]))
    out["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out} ({out['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
