"""Positivity-aware causal layer.

Estimates the effect of a binary treatment (default: having genetic support) on
advancement, adjusting for measured confounders, rather than the naive
association. Provides a cross-fitted AIPW (doubly robust) ATE, the
overlap-weighted ATO (Li, Morgan & Zaslavsky 2018), a trimmed ATE, covariate
balance diagnostics, an outcome-permutation test, DoWhy-style refutations,
target-clustered bootstrap CIs, and VanderWeele-Ding E-values.

Dependencies are limited to numpy and scikit-learn; DoWhy/EconML estimators are
drop-in alternatives.

CAVEAT: all estimates rest on the untestable no-unmeasured-confounding
assumption. Report the refutations and E-values alongside the point estimates.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Nuisance models for the causal layer are deliberately simple and
# well-regularised: on a modest set of confounder columns, scaled logistic
# regression gives far more stable AIPW/T-learner estimates than boosting.


def _outcome_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))


def _prop_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))


def _fit_prob(model, X, y):
    """Fit a probability model; fall back to a base rate if a class is missing.

    Guards the empty-group case (a cross-fitting fold with zero treated or zero
    control units): ``np.mean([])`` is NaN and would silently poison the AIPW
    estimate, turning the whole ATE/CI into NaN with no error.
    """
    if len(y) == 0:
        return lambda Z: np.full(len(Z), 0.5)
    if len(np.unique(y)) < 2:
        rate = float(np.mean(y))
        return lambda Z: np.full(len(Z), rate)
    model.fit(X, y)
    return lambda Z: model.predict_proba(Z)[:, 1]


def naive_association(t: np.ndarray, y: np.ndarray) -> float:
    """Unadjusted difference in success rate (treated - control)."""
    if (t == 1).sum() == 0 or (t == 0).sum() == 0:
        return float("nan")
    return float(y[t == 1].mean() - y[t == 0].mean())


def t_learner_ate(X: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    f1 = _fit_prob(_outcome_model(seed), X[t == 1], y[t == 1])
    f0 = _fit_prob(_outcome_model(seed), X[t == 0], y[t == 0])
    cate = f1(X) - f0(X)
    return float(np.mean(cate)), cate


def _aipw_core(X, t, y, seed: int = 0, n_folds: int = 5, trim: float = 0.1) -> Dict:
    """Cross-fitted nuisances -> AIPW ATE, overlap-weighted ATO, and a trimmed ATE.

    Returns point estimands plus the raw (unclipped) propensities so callers can
    judge positivity. The three estimands answer three questions:
      * ate        -- doubly-robust ATE on the whole sample (needs positivity).
      * ato        -- overlap-weighted (Li-Morgan-Zaslavsky) ATE on the population
                       with propensity mass in the middle; weights e(1-e) vanish
                       where e->0 or 1, so it is robust to the positivity failure
                       here (some units have raw e ~ 0).
      * trimmed_ate-- AIPW restricted to the overlap region e in [trim, 1-trim].
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=int)
    n = len(y)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    e = np.zeros(n)
    n_folds = max(2, min(n_folds, n // 50 if n >= 100 else 2))
    # Stratify folds on treatment so each training fold keeps treated AND control
    # units (else a fold's nuisance model sees an empty arm -> NaN). Needs at least
    # n_folds of the minority arm; the _fit_prob empty guard covers any residual.
    n_minority = int(min((t == 1).sum(), (t == 0).sum()))
    if n_minority >= n_folds:
        split_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, t)
    else:
        split_iter = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)
    for tr, te in split_iter:
        Xtr, ttr, ytr = X[tr], t[tr], y[tr]
        f1 = _fit_prob(_outcome_model(seed), Xtr[ttr == 1], ytr[ttr == 1])
        f0 = _fit_prob(_outcome_model(seed), Xtr[ttr == 0], ytr[ttr == 0])
        mu1[te] = f1(X[te])
        mu0[te] = f0(X[te])
        fe = _fit_prob(_prop_model(seed), Xtr, ttr)
        e[te] = fe(X[te])

    e_raw = e.copy()
    e_overlap = [round(float(np.min(e_raw)), 3), round(float(np.max(e_raw)), 3)]  # RAW
    ec = np.clip(e, 0.02, 0.98)
    psi = (mu1 - mu0) + t * (y - mu1) / ec - (1 - t) * (y - mu0) / (1 - ec)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    po1 = float(np.mean(mu1 + t * (y - mu1) / ec))
    po0 = float(np.mean(mu0 + (1 - t) * (y - mu0) / (1 - ec)))

    # Overlap-weighted (ATO): Hajek estimator, positivity-robust by construction.
    w_t = t * (1.0 - e_raw)
    w_c = (1 - t) * e_raw
    if np.sum(w_t) > 0 and np.sum(w_c) > 0:
        ato_po1 = float(np.sum(w_t * y) / np.sum(w_t))
        ato_po0 = float(np.sum(w_c * y) / np.sum(w_c))
        ato = ato_po1 - ato_po0
        # Kish effective sample size of the overlap-weighted comparison: how many
        # units the ATO effectively uses once units with poor overlap are down-
        # weighted. w_ato assigns each unit its own tilting weight (treated 1-e,
        # control e); ESS = (sum w)^2 / sum w^2.
        w_ato = np.where(t == 1, 1.0 - e_raw, e_raw)
        ato_ess = float(np.sum(w_ato) ** 2 / np.sum(w_ato ** 2)) if np.sum(w_ato ** 2) > 0 else float("nan")
    else:
        ato_po1 = ato_po0 = ato = ato_ess = float("nan")

    # Trimmed AIPW on the overlap region.
    keep = (e_raw >= trim) & (e_raw <= 1.0 - trim)
    ate_trim = (float(np.mean(psi[keep]))
                if keep.sum() > 0 and len(np.unique(t[keep])) == 2 else float("nan"))

    return {
        "ate": ate, "se": se, "propensity_overlap": e_overlap,
        "po_treated": po1, "po_control": po0,
        "ato": ato, "ato_po1": ato_po1, "ato_po0": ato_po0, "ato_ess": ato_ess,
        "trimmed_ate": ate_trim, "n_trimmed": int(keep.sum()),
        "trim_range": [trim, round(1.0 - trim, 3)], "psi": psi, "e": e_raw,
    }


def _balance(X, t, e, conf_names=None) -> Dict:
    """Covariate balance before vs after overlap weighting (Love-plot data).

    Standardised mean difference (SMD) per confounder: treated-minus-control mean over the
    pooled SD. Overlap weights (treated: 1-e, control: e) should drive SMDs toward 0 -- that
    exact-balance property is why we lead with the ATO. |SMD| < 0.1 is the usual "balanced"
    threshold. Reporting max/mean |SMD| unweighted vs weighted makes the ATO's fairness
    visible instead of asserted."""
    X = np.asarray(X, float); t = np.asarray(t, int); e = np.asarray(e, float)
    wt, wc = t * (1.0 - e), (1 - t) * e
    names = list(conf_names) if conf_names is not None else [f"X{i}" for i in range(X.shape[1])]
    per, raws, wtds = {}, [], []
    for i, nm in enumerate(names):
        x = X[:, i]
        sd = np.sqrt(0.5 * (x[t == 1].var() + x[t == 0].var()))
        if sd == 0:
            continue
        raw = (x[t == 1].mean() - x[t == 0].mean()) / sd
        wm1 = np.sum(wt * x) / np.sum(wt) if np.sum(wt) > 0 else 0.0
        wm0 = np.sum(wc * x) / np.sum(wc) if np.sum(wc) > 0 else 0.0
        wtd = (wm1 - wm0) / sd
        per[nm] = [round(float(raw), 3), round(float(wtd), 3)]
        raws.append(abs(raw)); wtds.append(abs(wtd))
    return {
        "max_abs_smd_unweighted": round(float(max(raws)), 3) if raws else None,
        "max_abs_smd_overlap_weighted": round(float(max(wtds)), 3) if wtds else None,
        "mean_abs_smd_unweighted": round(float(np.mean(raws)), 3) if raws else None,
        "mean_abs_smd_overlap_weighted": round(float(np.mean(wtds)), 3) if wtds else None,
        "n_balanced_after_weighting": int(sum(w < 0.1 for w in wtds)),
        "n_covariates": len(wtds),
        "per_covariate_smd_raw_then_weighted": per,
    }


def _ato_from_e(t, y, e):
    """Overlap-weighted (ATO) mean difference given fixed propensities e (Hajek form)."""
    wt, wc = t * (1.0 - e), (1 - t) * e
    if np.sum(wt) > 0 and np.sum(wc) > 0:
        return float(np.sum(wt * y) / np.sum(wt) - np.sum(wc * y) / np.sum(wc))
    return float("nan")


def aipw_ate(X: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int = 0, n_folds: int = 5) -> Dict:
    """Cross-fitted AIPW ATE (+ overlap-weighted ATO and trimmed ATE) with an
    influence-function SE on the ATE."""
    c = _aipw_core(X, t, y, seed=seed, n_folds=n_folds)
    ate, se = c["ate"], c["se"]
    return {
        "ate": round(ate, 4),
        "se": round(se, 4),
        "ci95": [round(ate - 1.96 * se, 4), round(ate + 1.96 * se, 4)],
        "propensity_overlap": c["propensity_overlap"],
        "po_treated": round(c["po_treated"], 4),
        "po_control": round(c["po_control"], 4),
        "ato": round(c["ato"], 4) if np.isfinite(c["ato"]) else None,
        "ato_po1": round(c["ato_po1"], 4) if np.isfinite(c["ato_po1"]) else None,
        "ato_po0": round(c["ato_po0"], 4) if np.isfinite(c["ato_po0"]) else None,
        "ato_ess": round(c["ato_ess"], 1) if np.isfinite(c["ato_ess"]) else None,
        "trimmed_ate": round(c["trimmed_ate"], 4) if np.isfinite(c["trimmed_ate"]) else None,
        "n_trimmed": c["n_trimmed"],
        "trim_range": c["trim_range"],
    }


def _clustered_causal_ci(X, t, y, clusters, seed: int = 0, n_boot: int = 300,
                         min_arm: int = 5) -> Dict:
    """Target-clustered bootstrap CIs for the ATE, ATO, and trimmed ATE.

    Refits the full cross-fitted pipeline on each resample of whole target
    clusters, so the CI reflects both nuisance estimation and the gene-correlated
    dependence structure (the influence-function SE ignores clustering)."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=int)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    cl_to_rows = {c: np.where(clusters == c)[0] for c in uniq}
    ates, atos, trims = [], [], []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([cl_to_rows[c] for c in chosen])
        tb = t[rows]
        if len(np.unique(tb)) < 2 or min(int((tb == 1).sum()), int((tb == 0).sum())) < min_arm:
            continue
        try:
            c = _aipw_core(X[rows], tb, y[rows], seed=seed)
        except Exception:
            continue
        if np.isfinite(c["ate"]):
            ates.append(c["ate"])
        if np.isfinite(c["ato"]):
            atos.append(c["ato"])
        if np.isfinite(c["trimmed_ate"]):
            trims.append(c["trimmed_ate"])

    def q(a):
        a = np.array(a, dtype=float)
        if len(a) == 0:
            return [float("nan"), float("nan")]
        return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)]

    return {
        "ate_ci_clustered": q(ates),
        "ato_ci_clustered": q(atos),
        "trimmed_ate_ci_clustered": q(trims),
        "n_boot_effective": len(ates),
    }


def evalue_rr(rr: float) -> float:
    """VanderWeele & Ding (2017) E-value for a risk ratio.

    The minimum strength -- on the risk-ratio scale -- of an unmeasured
    confounder's association with BOTH the treatment and the outcome that could
    fully explain away an observed effect, assuming no other bias. Larger E-value
    => the finding is more robust to unmeasured confounding. Symmetric for
    protective effects (rr < 1) via 1/rr.
    """
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def sensitivity(main: Dict) -> Dict:
    """Convert the AIPW effect to a risk ratio and report its E-value.

    It quantifies how much unmeasured confounding would be needed to explain
    away the adjusted estimate.
    """
    po1, po0 = main.get("po_treated"), main.get("po_control")
    ate = main.get("ate", float("nan"))
    ci = list(main.get("ci95") or [])
    lo = ci[0] if len(ci) > 0 else float("nan")
    hi = ci[1] if len(ci) > 1 else float("nan")
    if po0 is None or po1 is None or not np.isfinite(po0) or not np.isfinite(po1):
        return {"risk_ratio_adjusted": float("nan"), "e_value_point": float("nan"),
                "e_value_lower_ci": float("nan"), "note": "undefined (non-finite potential outcomes)"}
    # AIPW doubly-robust means are unbounded under extrapolation; clip into (0,1]
    # before forming a risk ratio so a spurious rr<1 (or rr<0) cannot flip the E-value.
    po1c = min(max(po1, 1e-6), 1.0)
    po0c = min(max(po0, 1e-6), 1.0)
    rr = po1c / po0c
    # E-value for the confidence limit nearest the null (RR=1). If the CI already
    # spans the null, no unmeasured confounding is needed to explain the estimate
    # away, so the E-value is 1.0 (VanderWeele & Ding convention) -- NOT the naive
    # 1/rr flip, which would spuriously exceed the point E-value.
    if np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi:
        e_limit = 1.0
    else:
        limit = lo if ate >= 0 else hi       # confidence bound closest to the null
        e_limit = evalue_rr((po0c + limit) / po0c)
    return {
        "risk_ratio_adjusted": round(float(rr), 3),
        "e_value_point": round(evalue_rr(rr), 3),
        "e_value_lower_ci": round(float(e_limit), 3),
        "note": (
            "E-value: minimum risk-ratio-scale strength an unmeasured "
            "target-disease confounder would need with BOTH genetic support and "
            "advancement to explain away the adjusted effect. e_value_lower_ci is "
            "for the CI limit nearest the null; 1.0 means the CI includes the null "
            "(not robust)."
        ),
    }


def refute(X: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)

    # Placebo treatment: permute t; a valid estimate should collapse toward 0.
    t_perm = rng.permutation(t)
    placebo = aipw_ate(X, t_perm, y, seed=seed)["ate"]

    # Random common cause: add a noise covariate; estimate should be stable.
    X_aug = np.column_stack([X, rng.normal(size=len(y))])
    rcc = aipw_ate(X_aug, t, y, seed=seed)["ate"]

    # Subset: re-estimate on a random 70% subsample; should be stable.
    m = rng.random(len(y)) < 0.7
    subset = aipw_ate(X[m], t[m], y[m], seed=seed)["ate"]

    return {
        "placebo_treatment_ate": placebo,      # want ~0
        "random_common_cause_ate": rcc,        # want ~ main ate
        "subset_70pct_ate": subset,            # want ~ main ate
    }


def causal_report(
    X_conf: pd.DataFrame, t: pd.Series, y: pd.Series, seed: int = 0,
    clusters=None, n_boot: int = 300,
) -> Dict:
    Xc = X_conf.values.astype(float)
    tt = t.values.astype(int)
    yy = y.values.astype(int)

    main = aipw_ate(Xc, tt, yy, seed=seed)
    tl_ate, cate = t_learner_ate(Xc, tt, yy, seed=seed)
    naive = naive_association(tt, yy)
    ref = refute(Xc, tt, yy, seed=seed)
    sens = sensitivity(main)

    # Diagnostics for the ATO: (i) covariate balance under overlap weighting (Love-plot
    # data; near-exact mean balance is expected by construction for a well-fitted logistic
    # propensity model), (ii) an outcome-permutation test of the ATO.
    core = _aipw_core(Xc, tt, yy, seed=seed)
    balance = _balance(Xc, tt, core["e"], conf_names=list(X_conf.columns))
    # Permutation test on the OUTCOME, holding the fitted propensities fixed: permute y
    # 2,000 times and recompute the ATO; p = fraction of |permuted ATO| >= |real ATO|.
    # Reusing e avoids nuisance refits (so the null ignores propensity-estimation noise).
    e_fixed = core["e"]; real_ato = _ato_from_e(tt, yy, e_fixed)
    prng = np.random.default_rng(seed)
    null = np.array([_ato_from_e(tt, prng.permutation(yy), e_fixed) for _ in range(2000)])
    null = null[np.isfinite(null)]
    neg_control = {
        "real_ato": round(float(real_ato), 4),
        "null_mean": round(float(np.mean(null)), 4),
        "null_ci95": [round(float(np.quantile(null, 0.025)), 4),
                      round(float(np.quantile(null, 0.975)), 4)],
        "permutation_p_value": round(float(np.mean(np.abs(null) >= abs(real_ato))), 4),
        "note": ("Outcome-permutation null holding propensities fixed (null_mean ~ 0 by "
                 "construction); permutation_p_value is P(|permuted ATO| >= |real ATO|)."),
    }

    # Because positivity is imperfect (raw propensity min ~ 0), the ATO
    # (overlap-weighted) is the primary robust estimand; the ATE and trimmed
    # ATE are reported alongside with target-clustered bootstrap CIs.
    clustered = None
    if clusters is not None:
        clustered = _clustered_causal_ci(Xc, tt, yy, clusters, seed=seed, n_boot=n_boot)
        main["ate_ci_clustered"] = clustered["ate_ci_clustered"]
        main["ato_ci_clustered"] = clustered["ato_ci_clustered"]
        main["trimmed_ate_ci_clustered"] = clustered["trimmed_ate_ci_clustered"]
        main["n_boot_effective"] = clustered["n_boot_effective"]

    # E-value for the ATO (the identified, positivity-robust estimand): how strong
    # an unmeasured confounder would have to be, on the risk-ratio scale, to explain
    # away the overlap-weighted effect and its CI limit nearest the null.
    ato_sens = None
    po1a, po0a = main.get("ato_po1"), main.get("ato_po0")
    if po1a is not None and po0a is not None and po0a > 0:
        rr_ato = min(max(po1a, 1e-6), 1.0) / min(max(po0a, 1e-6), 1.0)
        lo_ci = (main.get("ato_ci_clustered") or [None, None])[0]
        if lo_ci is not None and np.isfinite(lo_ci) and lo_ci > 0:
            e_lo = round(float(evalue_rr((po0a + lo_ci) / po0a)), 3)
        else:
            e_lo = 1.0  # CI touches/crosses the null
        ato_sens = {
            "risk_ratio_adjusted": round(float(rr_ato), 3),
            "e_value_point": round(float(evalue_rr(rr_ato)), 3),
            "e_value_lower_ci": e_lo,
            "note": ("E-value for the overlap-weighted (ATO) effect: min risk-ratio-scale "
                     "confounding needed to explain it away; e_value_lower_ci uses the "
                     "clustered-CI limit nearest the null."),
        }

    return {
        "treatment": t.name,
        "n_treated": int((tt == 1).sum()),
        "n_control": int((tt == 0).sum()),
        "naive_association_diff": round(naive, 4),
        "aipw_ate": main,
        "primary_estimand": "ATO (overlap-weighted); positivity imperfect so ATE is fragile",
        "t_learner_ate": round(tl_ate, 4),
        "sensitivity": sens,
        "ato_sensitivity": ato_sens,
        "covariate_balance": balance,
        "negative_control": neg_control,
        "refutations": ref,
        "cate_mean": round(float(np.mean(cate)), 4),
        "cate_std": round(float(np.std(cate)), 4),
        "interpretation": (
            "Compare naive_association_diff (confounded) with the adjusted "
            "estimands. ATO is the primary robust estimand (overlap-weighted, "
            "positivity-robust); ATE and trimmed_ate are shown with target-"
            "clustered bootstrap CIs. A much smaller adjusted effect means the "
            "raw association is largely confounding. placebo_treatment_ate should "
            "be near 0; the other refutations should stay near the ATE."
        ),
    }
