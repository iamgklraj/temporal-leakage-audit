"""Unit test for availability-time censoring (scripts/audit_lab_latency.py).

Run with either:  pytest tests/  OR  python tests/test_lab_latency.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_lab_latency as LL  # noqa: E402


def test_availability_censoring_excludes_late_results():
    # one stay; K observed at 23:00 (available 24:30) and at 20:00 (available 21:00)
    long = pd.DataFrame({"record_id": [1, 1], "minute": [23 * 60, 20 * 60],
                         "param": ["K", "K"], "value": [6.0, 4.0]})
    static = pd.DataFrame({"Age": [70.0], "Gender": [1.0], "Height": [170.0], "ICUType": [3.0],
                           "Weight_admission": [80.0]}, index=pd.Index([1], name="record_id"))
    avail = np.array([24 * 60 + 30, 21 * 60], dtype=float)
    X = LL.features_available(long, static, avail, 24 * 60, ["K"])
    assert X.loc[1, "K_count"] == 1 and X.loc[1, "K_last"] == 4.0     # the late result is not yet available
    Xv = LL.E.features(long, static, 24 * 60, ["K"])
    assert Xv.loc[1, "K_count"] == 2 and Xv.loc[1, "K_last"] == 6.0   # valid-time censoring admits it


def test_latency_draw_is_reproducible_and_non_negative():
    long = pd.DataFrame({"record_id": [1] * 3, "minute": [0, 60, 120], "param": ["HR"] * 3, "value": [80, 90, 100]})
    lat = {"HR": np.array([0.0, 10.0, 30.0])}
    a = LL.availability_minutes(long, lat, 0)
    b = LL.availability_minutes(long, lat, 0)
    assert np.array_equal(a, b) and np.all(a >= long["minute"].values)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
