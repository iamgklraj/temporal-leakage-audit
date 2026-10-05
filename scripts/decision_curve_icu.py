"""Clinical usefulness of the ICU mortality model with and without post-decision data.

Decision-curve analysis (net benefit; Vickers and Elkin 2006) of the 24-hour mortality model on
the PhysioNet/CinC Challenge 2012 test set C. Net benefit at a risk threshold p_t counts true
positives per patient minus false positives weighted by the odds p_t / (1 - p_t) at which a
clinician would act. The deployable model uses measurements recorded by 24 h after ICU admission;
the naive evaluation admits the full 48-hour record. The difference is the net benefit that a
leaky evaluation overstates. Paired patient-level percentile bootstrap (1,000 resamples, seed 0).
Also reported: Brier score, calibration-in-the-large (mean predicted minus observed risk), the
calibration intercept and slope (logistic regression of the outcome on the logit of the predicted
risk), and LAP by ICU type (coronary care, cardiac surgery recovery, medical, surgical; exploratory,
patient-level paired bootstrap within each type).

Reuses the features and learner of scripts/audit_physionet2012.py unchanged.
Usage:  python scripts/decision_curve_icu.py [--out outputs/decision_curve_icu.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_physionet2012 as E  # noqa: E402

THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.501, 0.05)]
N_BOOT, SEED = 1000, 0


def net_benefit(y, p, pt):
    n = len(y)
    act = p >= pt
    tp = np.sum(act & (y == 1))
    fp = np.sum(act & (y == 0))
    return tp / n - fp / n * pt / (1 - pt)


def treat_all(y, pt):
    pi = y.mean()
    return pi - (1 - pi) * pt / (1 - pt)


ICU_TYPES = {1: "coronary care", 2: "cardiac surgery recovery", 3: "medical", 4: "surgical"}


def calibration_line(y, p):
    """Calibration intercept and slope: logistic regression of y on logit(p) (unpenalized)."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return round(float(m.intercept_[0]), 4), round(float(m.coef_[0, 0]), 4)


def q(a):
    return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/decision_curve_icu.json")
    args = ap.parse_args()
    t0 = time.time()
    E.download()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    y = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    X24 = E.features(long, static, 24 * 60, params)
    X48 = E.features(long, static, 48 * 60, params)
    p24 = E.fit_predict(X24[is_tr], y[is_tr], X24[~is_tr])
    p48 = E.fit_predict(X48[is_tr], y[is_tr], X48[~is_tr])
    yte = y[~is_tr]
    types = static["ICUType"].values[~is_tr]
    lap_by_type = {}
    for code, name in ICU_TYPES.items():
        m = types == code
        yt, a, b = yte[m], p48[m], p24[m]
        rng_t = np.random.default_rng(SEED)
        draws = []
        for _ in range(N_BOOT):
            idx = rng_t.integers(0, m.sum(), m.sum())
            if yt[idx].min() == yt[idx].max():
                continue
            draws.append(average_precision_score(yt[idx], a[idx]) - average_precision_score(yt[idx], b[idx]))
        lap_by_type[name] = {"n": int(m.sum()), "deaths": int(yt.sum()), "base_rate": round(float(yt.mean()), 4),
                             "deployable_auprc": round(float(average_precision_score(yt, b)), 4),
                             "naive_auprc": round(float(average_precision_score(yt, a)), 4),
                             "LAP": round(float(average_precision_score(yt, a) - average_precision_score(yt, b)), 4),
                             "LAP_ci": q(draws)}
    rng = np.random.default_rng(SEED)
    n = len(yte)
    draws = {pt: {"dep": [], "naive": [], "all": []} for pt in THRESHOLDS}
    brier = {"dep": [], "naive": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = yte[idx]
        for pt in THRESHOLDS:
            draws[pt]["dep"].append(net_benefit(yb, p24[idx], pt))
            draws[pt]["naive"].append(net_benefit(yb, p48[idx], pt))
            draws[pt]["all"].append(treat_all(yb, pt))
        brier["dep"].append(np.mean((p24[idx] - yb) ** 2))
        brier["naive"].append(np.mean((p48[idx] - yb) ** 2))
    curve = []
    for pt in THRESHOLDS:
        d = {k: np.array(v) for k, v in draws[pt].items()}
        nb_dep, nb_naive = net_benefit(yte, p24, pt), net_benefit(yte, p48, pt)
        curve.append({"threshold": pt,
                      "net_benefit_deployable": round(nb_dep, 4), "ci_deployable": q(d["dep"]),
                      "net_benefit_naive": round(nb_naive, 4), "ci_naive": q(d["naive"]),
                      "net_benefit_treat_all": round(treat_all(yte, pt), 4),
                      "overstatement": round(nb_naive - nb_dep, 4),
                      "overstatement_ci": q(d["naive"] - d["dep"]),
                      "overstatement_per_100_patients": round(100 * (nb_naive - nb_dep), 2),
                      "deployable_minus_treat_all": round(nb_dep - treat_all(yte, pt), 4),
                      "deployable_minus_treat_all_ci": q(d["dep"] - d["all"]),
                      "share_flagged_deployable": round(float(np.mean(p24 >= pt)), 4),
                      "share_flagged_naive": round(float(np.mean(p48 >= pt)), 4)})
    b_dep, b_naive = float(np.mean((p24 - yte) ** 2)), float(np.mean((p48 - yte) ** 2))
    out = {"dataset": "PhysioNet/CinC Challenge 2012, test set C", "script": "scripts/decision_curve_icu.py",
           "decision_time_hours": 24, "n_test": int(n), "test_base_rate": round(float(yte.mean()), 4),
           "method": "net benefit = TP/n - FP/n * pt/(1-pt); paired patient-level percentile bootstrap, "
                     f"{N_BOOT} resamples, seed {SEED}",
           "curve": curve,
           "brier_deployable": round(b_dep, 4), "brier_deployable_ci": q(brier["dep"]),
           "brier_naive": round(b_naive, 4), "brier_naive_ci": q(brier["naive"]),
           "brier_difference_naive_minus_deployable": round(b_naive - b_dep, 4),
           "brier_difference_ci": q(np.array(brier["naive"]) - np.array(brier["dep"])),
           "calibration_in_the_large_deployable": round(float(p24.mean() - yte.mean()), 4),
           "calibration_in_the_large_naive": round(float(p48.mean() - yte.mean()), 4),
           "calibration_intercept_slope_deployable": calibration_line(yte, p24),
           "calibration_intercept_slope_naive": calibration_line(yte, p48),
           "lap_by_icu_type_exploratory": lap_by_type,
           "runtime_seconds": round(time.time() - t0, 1)}
    for c in curve:
        print(c["threshold"], c["net_benefit_deployable"], c["net_benefit_naive"], c["overstatement"],
              c["overstatement_ci"], flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
