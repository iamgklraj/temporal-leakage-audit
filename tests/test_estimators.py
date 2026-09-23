"""Tests for the leakage and causal estimators.

Run with either:  pytest tests/  OR  python tests/test_estimators.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from temporal_leakage_audit import causal, features, leakage, metrics, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402


def _synth_split():
    cfg = get_config()
    cfg["synth"]["n_programs"] = 1500
    cfg["synth"]["n_targets"] = 900
    cfg["seed"] = 3
    programs, evidence = synth.generate(cfg)
    Xc, Xn, y, meta, groups = features.build(programs, evidence)
    tr, te, _ = splits.temporal_split(meta, 2016, (2017, 2020), enforce_target_disjoint=True)
    return programs, evidence, Xc, Xn, y, meta, groups, tr, te


def test_curve_endpoints_match_lap_report():
    """h=0 of the leakage-response curve is the censored model; h=inf is the naive one."""
    programs, evidence, Xc, Xn, y, meta, groups, tr, te = _synth_split()
    cl = meta.loc[te, "target_id"].values
    curve = leakage.leakage_response_curve(programs, evidence, tr, te, clusters=cl, n_boot=50,
                                           horizons=(0, 2, float("inf")))
    lap = leakage.lap_report(Xc, Xn, y, tr, te, groups, clusters=cl, n_boot=50)
    assert abs(curve["deployable_auprc"] - lap["censored"]["auprc"]) < 1e-4
    assert abs(curve["naive_auprc"] - lap["naive"]["auprc"]) < 1e-4
    assert abs(curve["total_LAP_auprc"] - lap["LAP"]["auprc"]) < 1e-4
    assert curve["total_LAP_auprc"] > 0, "synthetic data has leakage baked in"
    lo, hi = curve["total_LAP_auprc_ci"]
    assert lo <= hi
    print("OK: leakage-response curve endpoints match the LAP report")


def test_clustered_bootstrap_draws_are_aligned():
    """Every key gets the same number of draws, so paired contrasts are well defined."""
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.3).astype(int)
    preds = {"a": rng.random(60), "b": rng.random(60)}
    clusters = np.repeat(np.arange(12), 5)
    s = leakage._clustered_auprc_samples(y, preds, clusters, n_boot=200, seed=1)
    assert len(s["a"]) == len(s["b"]) > 0
    print("OK: clustered bootstrap draws are aligned across keys")


def test_overlap_weighting_removes_confounding():
    """Treatment has NO effect but is confounded: naive assoc is large, ATO ~ 0."""
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.normal(size=(n, 2))
    t = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * x[:, 0])))).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 1.0 * x[:, 0] - 0.5 * x[:, 1])))).astype(int)
    naive = causal.naive_association(t, y)
    res = causal.aipw_ate(x, t, y, seed=0)
    assert naive > 0.15, naive
    assert abs(res["ato"]) < 0.05, res["ato"]
    assert abs(res["ate"]) < 0.05, res["ate"]
    print(f"OK: confounding removed (naive {naive:+.3f} -> ATO {res['ato']:+.3f}, ATE {res['ate']:+.3f})")


def test_overlap_weighting_balances_covariates():
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=(n, 3))
    t = (rng.random(n) < 1 / (1 + np.exp(-(x[:, 0] - 0.5 * x[:, 2])))).astype(int)
    e = causal._aipw_core(x, t, (rng.random(n) < 0.4).astype(int), seed=0)["e"]
    b = causal._balance(x, t, e)
    assert b["max_abs_smd_unweighted"] > 0.3
    assert b["max_abs_smd_overlap_weighted"] < 0.1
    print("OK: overlap weights balance the confounders")


def test_evalue():
    assert abs(causal.evalue_rr(2.0) - (2.0 + np.sqrt(2.0))) < 1e-12
    assert abs(causal.evalue_rr(0.5) - causal.evalue_rr(2.0)) < 1e-12   # symmetric
    assert causal.evalue_rr(1.0) == 1.0
    print("OK: E-value formula")


def test_decision_metrics():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert metrics.policy_value_at_budget(y, s, 0.4) == 0.5          # top-2: [1, 0]
    # net benefit at pt=0.5 (weight 1): TP=2, FP=1 among p>=0.5 -> (2-1)/5
    assert abs(metrics.net_benefit(y, s, 0.5) - 0.2) < 1e-12
    print("OK: decision metrics")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all estimator tests passed")
