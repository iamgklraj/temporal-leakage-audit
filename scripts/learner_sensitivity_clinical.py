"""Learner sensitivity of LAP in the two patient-level demonstrations.

LAP is defined for a stated learner. The trial-level audits report it for three learners
(scripts/robustness.py); this script does the same for the ICU mortality demonstration (PhysioNet 2012,
decision at 24 hours; scripts/audit_physionet2012.py) and the sepsis demonstration (PhysioNet 2019,
decision at 24 hours; scripts/audit_physionet2019_sepsis.py), with the feature matrices, cohorts and
splits of those scripts unchanged. Learners: the paper's gradient-boosted trees (models.fit_gbdt),
L2 logistic regression (median imputation, standardization, C = 1) and a random forest (500 trees,
minimum leaf size 5, seed 0; missing values handled natively), as in scripts/robustness.py. Columns
never observed in the training rows are dropped for every learner. LAP = AUPRC(naive) - AUPRC(deployable)
with the patient-level paired percentile bootstrap (1,000 resamples, seed 0).

Usage:  python scripts/learner_sensitivity_clinical.py [--out outputs/learner_sensitivity_clinical.json]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_physionet2012 as E  # noqa: E402
import audit_physionet2019_sepsis as S  # noqa: E402
from temporal_leakage_audit import metrics as M  # noqa: E402
from temporal_leakage_audit import models  # noqa: E402


def fit_predict(learner, Xtr, ytr, Xte):
    keep = Xtr.columns[Xtr.notna().any().values]
    Xtr, Xte = Xtr[keep], Xte[keep]
    if learner == "hgb_default":
        return models.predict_gbdt(models.fit_gbdt(Xtr, ytr, seed=0), Xte)
    if learner == "logreg_l2":
        clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                        ("clf", LogisticRegression(C=1.0, max_iter=10000))])
    else:
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=1)
    clf.fit(Xtr.values, ytr)
    return clf.predict_proba(Xte.values)[:, 1]


def lap_block(yte, p_dep, p_naive):
    b = E.paired_boot(yte, {"dep": p_dep, "naive": p_naive})
    d = b["naive"]["auprc"] - b["dep"]["auprc"]
    return {"deployable_auprc": round(M.auprc(yte, p_dep), 4), "naive_auprc": round(M.auprc(yte, p_naive), 4),
            "deployable_auroc": round(M.auroc(yte, p_dep), 4), "naive_auroc": round(M.auroc(yte, p_naive), 4),
            "LAP": round(M.auprc(yte, p_naive) - M.auprc(yte, p_dep), 4), "LAP_ci": E.ci(d),
            "LAP_p_gt_0": round(float(np.mean(d > 0)), 3)}


def icu():
    E.download()
    long, static, outc = E.load_records()
    ids = static.index.intersection(outc.index)
    static = static.loc[ids]
    y = outc.loc[ids].values.astype(int)
    is_tr = static["set"].isin(["a", "b"]).values
    params = sorted(long["param"].unique())
    X24 = E.features(long, static, 24 * 60, params)
    X48 = E.features(long, static, 48 * 60, params)
    out = {}
    for lr in ("hgb_default", "logreg_l2", "random_forest"):
        p24 = fit_predict(lr, X24[is_tr], y[is_tr], X24[~is_tr])
        p48 = fit_predict(lr, X48[is_tr], y[is_tr], X48[~is_tr])
        out[lr] = lap_block(y[~is_tr], p24, p48)
        print("ICU", lr, out[lr], flush=True)
    return out


def sepsis():
    S.download()
    hourly = S.load_hourly()
    stays = S.stay_table(hourly)
    t_d = 24
    long = S.long_table(hourly[hourly["ICULOS"] <= t_d + 24])
    del hourly
    eligible, y, _, _ = S.cohort(stays, t_d)
    st = stays[eligible]
    yy = y[eligible].values.astype(int)
    is_tr = (st["set"] == "A").values
    Xd = S.features(long, st, t_d, S.SERIES)
    Xn = S.features(long, st, t_d + 24, S.SERIES)
    out = {}
    for lr in ("hgb_default", "logreg_l2", "random_forest"):
        pd_ = fit_predict(lr, Xd[is_tr], yy[is_tr], Xd[~is_tr])
        pn = fit_predict(lr, Xn[is_tr], yy[is_tr], Xn[~is_tr])
        out[lr] = lap_block(yy[~is_tr], pd_, pn)
        print("sepsis", lr, out[lr], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/learner_sensitivity_clinical.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"script": "scripts/learner_sensitivity_clinical.py",
           "learners": {"hgb_default": "models.fit_gbdt (the paper's learner)",
                        "logreg_l2": "SimpleImputer(median) + StandardScaler + LogisticRegression(C=1)",
                        "random_forest": "RandomForestClassifier(500 trees, min_samples_leaf=5, seed 0)"},
           "bootstrap": "patient-level paired percentile bootstrap, 1,000 resamples, seed 0"}
    out["icu_mortality_decision_24h"] = icu()
    out["sepsis_onset_decision_24h"] = sepsis()
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
