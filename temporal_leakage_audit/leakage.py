"""Temporal leakage diagnostics: LAP and the leakage-response curve.

LAP (Leakage-Attributable Performance) = performance of a time-blind (naive)
model minus performance of a time-censored (as-of-t) model, evaluated on the
same labels. A large LAP means much of the reported skill depends on information
that did not exist at decision time. LAP is decomposed by evidence source with a
temporal placebo (admit one source at full hindsight, keep the rest censored).

Every headline number is reported with a *target-clustered* bootstrap CI: the
GBDT is deterministic given the data, so the real uncertainty is sampling
variation on the (small, gene-correlated) test block, not seed spread. LAP CIs
use paired resamples (naive and deployable AUPRC differenced within each
bootstrap draw) so the contrast keeps its correlation structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import features as features
from . import metrics as M
from . import models as models


def _fit_eval(X_tr, y_tr, X_te, y_te, seed, cols=None) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit the GBDT on the training block; return test metrics and test predictions."""
    clf = models.fit_gbdt(X_tr, y_tr, seed=seed, cols=cols)
    p = models.predict_gbdt(clf, X_te, cols=cols)
    return {"auprc": M.auprc(y_te, p), "auroc": M.auroc(y_te, p), "brier": M.brier(y_te, p)}, p


def _clustered_auprc_samples(
    yte: np.ndarray,
    pred_by_key: Dict[str, np.ndarray],
    clusters: Sequence,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Target-clustered bootstrap of AUPRC for several prediction vectors on shared
    resamples. Returns, per key, an aligned list of AUPRC values (one per bootstrap
    draw in which the resampled test set had both classes). Alignment lets callers
    form *paired* contrasts (e.g. LAP = naive - deployable) within each draw.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    cl_to_rows = {c: np.where(clusters == c)[0] for c in uniq}
    samples: Dict[str, List[float]] = {k: [] for k in pred_by_key}
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([cl_to_rows[c] for c in chosen])
        yb = yte[rows]
        if len(np.unique(yb)) < 2:
            continue  # AUPRC undefined; drop the draw for ALL keys (keeps alignment)
        for k, p in pred_by_key.items():
            samples[k].append(float(M.auprc(yb, p[rows])))
    return samples


def _ci(vals: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {
        "mean": round(float(np.mean(arr)), 4),
        "lo": round(float(np.quantile(arr, alpha / 2)), 4),
        "hi": round(float(np.quantile(arr, 1 - alpha / 2)), 4),
        "n": int(len(arr)),
    }


def lap_report(
    X_cens: pd.DataFrame,
    X_naive: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
    groups: Dict[str, List[str]],
    seed: int = 0,
    budget_frac: float = 0.2,
    clusters: Optional[Sequence] = None,
    n_boot: int = 1000,
) -> Dict:
    ytr = y.loc[train_idx].values
    yte = y.loc[test_idx].values
    base_rate = float(np.mean(yte))

    # Full naive vs full censored (headline LAP).
    naive_m, p_naive = _fit_eval(X_naive.loc[train_idx], ytr, X_naive.loc[test_idx], yte, seed)
    cens_m, p_cens = _fit_eval(X_cens.loc[train_idx], ytr, X_cens.loc[test_idx], yte, seed)

    lap = {
        "auprc": naive_m["auprc"] - cens_m["auprc"],
        "auroc": naive_m["auroc"] - cens_m["auroc"],
        "policy_value": (
            M.policy_value_at_budget(yte, p_naive, budget_frac)
            - M.policy_value_at_budget(yte, p_cens, budget_frac)
        ),
    }

    # Target-clustered bootstrap CIs on the headline AUPRCs and on LAP (paired).
    ci_block = None
    if clusters is not None:
        s = _clustered_auprc_samples(yte, {"naive": p_naive, "cens": p_cens},
                                     clusters, n_boot=n_boot, seed=seed)
        lap_draws = [n - c for n, c in zip(s["naive"], s["cens"])]
        ci_block = {
            "naive_auprc_ci": _ci(s["naive"]),
            "censored_auprc_ci": _ci(s["cens"]),
            "LAP_auprc_ci": _ci(lap_draws),
            "LAP_auprc_p_gt_0": round(float(np.mean(np.array(lap_draws) > 0)), 3) if lap_draws else float("nan"),
        }

    # Leave-one-group-out drop-column importance on the NAIVE model: AUPRC lost when a
    # feature group is removed. NOTE: this is *feature importance*, NOT a clean leakage
    # attribution -- a GBDT can route around a dropped group via correlated columns, so it
    # UNDER-states a group's true leakage contribution. The per-source leakage
    # isolation is the temporal placebo below (placebo_literature_censored_auprc); use
    # that, not this, to claim "which source carries the leakage".
    all_cols = list(X_naive.columns)
    dropcol_importance = {}
    for gname, gcols in groups.items():
        if gname == "confounders":
            continue
        keep = [c for c in all_cols if c not in set(gcols)]
        m_drop, _ = _fit_eval(
            X_naive.loc[train_idx], ytr, X_naive.loc[test_idx], yte, seed, cols=keep
        )
        dropcol_importance[gname] = round(naive_m["auprc"] - m_drop["auprc"], 4)

    # Temporal placebo: replace the naive literature features with their
    # censored counterparts; the drop is the literature leakage, isolated.
    lit_cols = groups.get("literature", [])
    X_placebo = X_naive.copy()
    X_placebo[lit_cols] = X_cens[lit_cols]
    plac_m, _ = _fit_eval(
        X_placebo.loc[train_idx], ytr, X_placebo.loc[test_idx], yte, seed
    )

    out = {
        "test_base_rate": round(base_rate, 4),
        "naive": {k: round(v, 4) for k, v in naive_m.items()},
        "censored": {k: round(v, 4) for k, v in cens_m.items()},
        "deployable_auprc_lift_over_base": round(cens_m["auprc"] - base_rate, 4),
        "LAP": {k: round(v, 4) for k, v in lap.items()},
        "auprc_group_dropcol_importance": dropcol_importance,
        "placebo_literature_censored_auprc": round(plac_m["auprc"], 4),
        "interpretation": (
            "LAP.auprc is the AUPRC that vanishes once features are restricted to what "
            "was actually knowable at decision time. Read AUPRC against test_base_rate "
            "(a random ranker scores ~= base rate). placebo_literature_censored_auprc "
            "isolates the literature channel (naive AUPRC minus this value = "
            "literature-attributable LAP). "
            "auprc_group_dropcol_importance is leave-one-group-out feature importance and "
            "UNDER-states leakage; do not read it as a clean per-source attribution."
        ),
    }
    if ci_block is not None:
        out.update(ci_block)
    return out


def per_source_placebo_lap(
    X_cens: pd.DataFrame,
    X_naive: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
    groups: Dict[str, List[str]],
    clusters: Optional[Sequence] = None,
    seed: int = 0,
    n_boot: int = 1000,
) -> Dict:
    """Per-evidence-source leakage decomposition via the temporal placebo.

    For each evidence source ``s``, admit ONLY ``s`` at full hindsight while every other
    feature stays as-of-time censored, and measure the AUPRC gain over the fully deployable
    model: ``LAP_s = AUPRC(deployable + s naive) - AUPRC(deployable)``. Unlike leave-one-group-out
    drop-column importance (which a GBDT routes around via correlated columns), this isolates
    each source's leakage on the deployable baseline, with paired target-clustered CIs.
    """
    ytr, yte = y.loc[train_idx].values, y.loc[test_idx].values
    dep_m, p_dep = _fit_eval(X_cens.loc[train_idx], ytr, X_cens.loc[test_idx], yte, seed)
    sources = [g for g in groups if g not in ("structural", "confounders")]
    preds = {"__deployable__": p_dep}
    for g in sources:
        gcols = groups[g]
        Xg = X_cens.copy()
        Xg[gcols] = X_naive[gcols]
        _, p_g = _fit_eval(Xg.loc[train_idx], ytr, Xg.loc[test_idx], yte, seed)
        preds[g] = p_g

    per_source = {}
    if clusters is not None:
        s = _clustered_auprc_samples(yte, preds, clusters, n_boot=n_boot, seed=seed)
        for g in sources:
            draws = [a - b for a, b in zip(s[g], s["__deployable__"])]
            per_source[g] = {
                "LAP": round(float(M.auprc(yte, preds[g]) - dep_m["auprc"]), 4),
                "ci": [_ci(draws)["lo"], _ci(draws)["hi"]],
                "p_gt_0": round(float(np.mean(np.array(draws) > 0)), 3) if draws else None,
            }
    else:
        for g in sources:
            per_source[g] = {"LAP": round(float(M.auprc(yte, preds[g]) - dep_m["auprc"]), 4)}
    per_source = dict(sorted(per_source.items(), key=lambda kv: -kv[1]["LAP"]))
    return {"deployable_auprc": round(dep_m["auprc"], 4), "per_source": per_source}


def leakage_response_curve(
    programs: pd.DataFrame,
    evidence: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    clusters: Optional[Sequence] = None,
    horizons=(0, 1, 2, 3, 5, 10, float("inf")),
    seed: int = 0,
    n_boot: int = 1000,
    budget_frac: float = 0.2,
) -> Dict:
    """Temporal leakage-response curve: performance vs. hindsight horizon.

    For each horizon ``h`` (years), the censored feature matrix admits evidence
    dated ``<= info_time + h``: ``h=0`` is the deployable matrix and
    ``h=inf`` is the naive/leaky one. The GBDT is deterministic, so each horizon
    is fit once; uncertainty comes from a target-clustered bootstrap of the test
    block, reported as a CI per horizon and on the total LAP (naive - deployable),
    the latter using paired resamples.

    This generalises the single LAP number to an instrument: it shows how quickly
    apparent skill accrues as post-decision (mostly literature) evidence leaks in,
    and yields (a) a deployable estimate, (b) the total leakage bias with a CI, and
    (c) a leakage "half-life" -- how few years of hindsight already produce half
    the (point) bias.
    """
    y = programs.set_index("program_id")["label"].astype(int)
    ytr, yte = y.loc[train_idx].values, y.loc[test_idx].values
    test_base_rate = float(np.mean(yte))

    pred_by_key: Dict[str, np.ndarray] = {}
    key_of_h: Dict[float, str] = {}
    curve = []
    for h in horizons:
        X_h, _, _, _, _ = features.build(programs, evidence, horizon=float(h))
        clf = models.fit_gbdt(X_h.loc[train_idx], ytr, seed=seed)
        p = models.predict_gbdt(clf, X_h.loc[test_idx])
        key = "naive" if h == float("inf") else f"h{int(h)}"
        pred_by_key[key] = p
        key_of_h[h] = key
        curve.append({
            "horizon": (None if h == float("inf") else int(h)),
            "auprc_mean": round(float(M.auprc(yte, p)), 4),
            "policy_mean": round(float(M.policy_value_at_budget(yte, p, budget_frac)), 4),
        })

    base = curve[0]["auprc_mean"]        # h=0, deployable
    naive = curve[-1]["auprc_mean"]      # h=inf, naive
    total_lap = round(naive - base, 4)

    # The response curve can be non-monotone (post-freeze evidence eventually adds noise),
    # so naive-minus-deployable understates the true leakage ceiling. peak_LAP is the largest
    # excess over the deployable estimate across any admitted horizon -- the maximum a
    # time-blind model could gain from hindsight.
    peak_auprc = max(c["auprc_mean"] for c in curve)
    peak_h = next((c["horizon"] for c in curve if c["auprc_mean"] == peak_auprc), None)
    peak_lap = round(peak_auprc - base, 4)

    half_life = None
    if total_lap > 0:
        for c in curve:
            if c["horizon"] is not None and (c["auprc_mean"] - base) >= 0.5 * total_lap:
                half_life = c["horizon"]
                break

    # Target-clustered bootstrap CIs (per horizon + paired LAP).
    lap_ci = None
    lap_p_gt_0 = None
    if clusters is not None:
        s = _clustered_auprc_samples(yte, pred_by_key, clusters, n_boot=n_boot, seed=seed)
        for c in curve:
            k = "naive" if c["horizon"] is None else f"h{c['horizon']}"
            c["auprc_ci"] = [_ci(s[k])["lo"], _ci(s[k])["hi"]]
        h0k = key_of_h[0]
        lap_draws = [n - b for n, b in zip(s["naive"], s[h0k])]
        lap_ci = [_ci(lap_draws)["lo"], _ci(lap_draws)["hi"]]
        lap_p_gt_0 = round(float(np.mean(np.array(lap_draws) > 0)), 3) if lap_draws else float("nan")

    return {
        "curve": curve,
        "test_base_rate": round(test_base_rate, 4),
        "deployable_auprc": base,
        "deployable_lift_over_base": round(base - test_base_rate, 4),
        "naive_auprc": naive,
        "total_LAP_auprc": total_lap,
        "total_LAP_auprc_ci": lap_ci,
        "total_LAP_auprc_p_gt_0": lap_p_gt_0,
        "peak_LAP_auprc": peak_lap,
        "peak_horizon": peak_h,
        "leakage_half_life_years": half_life,
        "n_boot": n_boot,
        "interpretation": (
            "AUPRC vs. hindsight horizon h (years of post-freeze evidence admitted). "
            "h=0 is the deployable estimate; the rise toward h=inf is temporal leakage. "
            "total_LAP_auprc is the point leakage bias and total_LAP_auprc_ci its "
            "target-clustered 95% bootstrap CI; total_LAP_auprc_p_gt_0 is the bootstrap "
            "probability that LAP>0. AUPRC is read against test_base_rate."
        ),
    }
