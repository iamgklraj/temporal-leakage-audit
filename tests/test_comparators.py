"""Unit tests for the comparator checks and the decision-curve analysis.

Run with either:  pytest tests/  OR  python tests/test_comparators.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import compare_methods as CM  # noqa: E402
import decision_curve_icu as DC  # noqa: E402


def test_net_benefit_by_hand():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2])
    # at p_t = 0.5 the first two are treated: 1 TP, 1 FP, odds 1 -> 1/4 - 1/4 = 0
    assert abs(DC.net_benefit(y, p, 0.5)) < 1e-12
    # at p_t = 0.05 all four are treated: 2 TP, 2 FP -> 2/4 - 2/4 * (0.05 / 0.95)
    assert abs(DC.net_benefit(y, p, 0.05) - (0.5 - 0.5 * 0.05 / 0.95)) < 1e-12
    # treating everyone equals the treat-all reference
    assert abs(DC.net_benefit(y, np.ones(4), 0.2) - DC.treat_all(y, 0.2)) < 1e-12


def test_univariate_screen_is_direction_free_and_handles_missing():
    y = np.array([1, 1, 0, 0])
    assert CM.univariate_auroc(np.array([1.0, 2.0, 3.0, 4.0]), y) == 1.0   # inversely related
    assert CM.univariate_auroc(np.array([4.0, 3.0, 2.0, 1.0]), y) == 1.0
    assert CM.univariate_auroc(np.array([np.nan, np.nan, np.nan, np.nan]), y) == 0.5
    assert CM.univariate_auroc(np.array([5.0, 5.0, 5.0, 5.0]), y) == 0.5
    # missing values rank lowest: here they mark the negatives
    assert CM.univariate_auroc(np.array([2.0, 3.0, np.nan, np.nan]), y) == 1.0


def test_joint_permutation_of_uninformative_columns_changes_nothing():
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.normal(size=n)
    y = (x0 + 0.3 * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([x0, np.zeros(n), np.zeros(n)])   # constant columns carry nothing
    clf = CM.gbdt().fit(X, y)
    res = CM.permutation(clf, X, y, ["x0", "c1", "c2"], ["c1", "c2"], n_perm=5)
    assert res["joint_drop"] == 0.0 and res["sum_of_single_drops"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
