"""Unit test for the sepsis demonstration (scripts/audit_physionet2019_sepsis.py).

Run with either:  pytest tests/  OR  python tests/test_sepsis_features.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_physionet2019_sepsis as S  # noqa: E402


def _hourly(stays):
    """Hourly rows in the layout of load_hourly(); stays = {id: (last hour, first labelled hour, values)}."""
    rows = []
    for pid, (last_hour, label_from, values) in stays.items():
        for t in range(1, last_hour + 1):
            r = dict.fromkeys(S.SERIES, np.nan)
            r.update(patient=pid, Age=60.0, Gender=1.0, Unit1=np.nan, Unit2=np.nan, HospAdmTime=-5.0,
                     ICULOS=t, SepsisLabel=int(label_from is not None and t >= label_from), set="A")
            r.update(values.get(t, {}))
            rows.append(r)
    return pd.DataFrame(rows)


def test_features_respect_the_cutoff_and_cohort_follows_the_definitions():
    """Features at a cut-off ignore later rows; eligibility and outcome follow the pre-specified design."""
    v1 = {t: {"HR": 80.0} for t in range(1, 24)}
    v1.update({24: {"HR": 90.0}, 5: {"HR": 80.0, "Lactate": 2.0}, 26: {"HR": 200.0, "Lactate": 9.0}})
    v1.update({t: {"HR": 200.0} for t in (25, 27, 28, 29, 30)})
    hourly = _hourly({1: (30, None, v1),    # never septic, in the ICU past 24 h
                      2: (40, 20, {}),      # t_sepsis = 20 + 6 = 26
                      3: (40, 18, {}),      # t_sepsis = 24: septic at the decision
                      4: (20, None, {}),    # left the ICU before hour 24
                      5: (60, 43, {}),      # t_sepsis = 49: just after the 24-hour window
                      6: (60, 42, {}),      # t_sepsis = 48: last hour of the window
                      7: (24, None, {}),    # record ends at the decision hour
                      8: (30, 1, {})})      # labelled from the first row: t_sepsis = 7
    stays = S.stay_table(hourly)
    assert stays.loc[2, "t_sepsis"] == 26 and np.isnan(stays.loc[1, "t_sepsis"])

    # features at the cut-off use only rows with ICULOS <= cut-off (the row at the cut-off included)
    long = S.long_table(hourly)
    X = S.features(long, stays, 24, S.SERIES)
    assert X.shape[1] == len(S.STATIC) + 5 * len(S.SERIES)
    assert not ({"ICULOS", "SepsisLabel", "first_hour", "last_hour", "t_sepsis", "labelled_from_first_row", "set"}
                & set(X.columns))
    assert X.loc[1, "HR_max"] == 90.0 and X.loc[1, "HR_last"] == 90.0 and X.loc[1, "HR_count"] == 24
    assert abs(X.loc[1, "HR_mean"] - (23 * 80.0 + 90.0) / 24) < 1e-12
    assert X.loc[1, "Lactate_count"] == 1 and X.loc[1, "Lactate_last"] == 2.0
    assert X.loc[1, "Platelets_count"] == 0 and np.isnan(X.loc[1, "Platelets_last"])
    X30 = S.features(long, stays, 30, S.SERIES)
    assert X30.loc[1, "HR_max"] == 200.0 and X30.loc[1, "HR_count"] == 30 and X30.loc[1, "Lactate_last"] == 9.0
    # rewriting or adding anything after the cut-off leaves the features unchanged
    late = long["hour"] > 24
    noisy = long.copy()
    noisy.loc[late, "value"] = np.random.default_rng(0).normal(1e3, 1e2, late.sum())
    extra = pd.DataFrame({"patient": [2, 5], "hour": [25, 47], "param": ["WBC", "HR"], "value": [99.0, 1.0]})
    pd.testing.assert_frame_equal(S.features(pd.concat([noisy, extra], ignore_index=True), stays, 24, S.SERIES), X)

    # eligibility (a row at ICULOS >= t_d and t_sepsis > t_d) and outcome (t_d < t_sepsis <= t_d + 24)
    eligible, y, _, _ = S.cohort(stays, 24)
    assert eligible.to_dict() == {1: True, 2: True, 3: False, 4: False, 5: True, 6: True, 7: True, 8: False}
    assert y[eligible].to_dict() == {1: 0, 2: 1, 5: 0, 6: 1, 7: 0}
    eligible, y, _, _ = S.cohort(stays, 12)
    assert eligible.to_dict() == {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: False}
    assert y[eligible].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}


if __name__ == "__main__":
    test_features_respect_the_cutoff_and_cohort_follows_the_definitions()
    print("all tests passed")
