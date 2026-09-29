"""Guardrail tests for the censoring invariant.

Run with either:  pytest tests/  OR  python tests/test_no_leakage.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from temporal_leakage_audit import features, models  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402


def _small_cfg():
    cfg = get_config()
    cfg["synth"]["n_programs"] = 800
    cfg["synth"]["n_targets"] = 300
    cfg["seed"] = 1
    return cfg


def test_censoring_matches_manual_counts():
    """The censored literature count must equal a hand-computed as-of-t count."""
    cfg = _small_cfg()
    programs, evidence = synth.generate(cfg)
    X_cens, X_naive, y, meta, groups = features.build(programs, evidence)

    ev = evidence.merge(programs[["program_id", "info_time"]], on="program_id")
    manual = (
        ev[(ev["evidence_type"] == "literature") & (ev["evidence_date"] <= ev["info_time"])]
        .groupby("program_id").size()
    )
    got = X_cens["ev_literature_count"].copy()
    # Compare on programs that have any censored literature.
    for pid, cnt in manual.items():
        assert abs(got.loc[pid] - cnt) < 1e-6, f"{pid}: censored count {got.loc[pid]} != {cnt}"

    # No program may have a censored count exceeding its naive count.
    assert (X_cens["ev_literature_count"] <= X_naive["ev_literature_count"] + 1e-9).all()
    print("OK: censored literature counts match manual as-of-t computation")


def test_leakage_is_detectable():
    """Naive model should beat censored model on the synthetic data (leakage present)."""
    cfg = _small_cfg()
    programs, evidence = synth.generate(cfg)
    X_cens, X_naive, y, meta, groups = features.build(programs, evidence)

    # simple time split
    cut = int(meta["info_time"].quantile(0.6))
    tr = meta.index[meta["info_time"] <= cut]
    te = meta.index[meta["info_time"] > cut]
    ytr, yte = y.loc[tr].values, y.loc[te].values

    from sklearn.metrics import average_precision_score

    cn = models.fit_gbdt(X_naive.loc[tr], ytr, seed=1)
    pn = models.predict_gbdt(cn, X_naive.loc[te])
    cc = models.fit_gbdt(X_cens.loc[tr], ytr, seed=1)
    pc = models.predict_gbdt(cc, X_cens.loc[te])

    ap_n = average_precision_score(yte, pn)
    ap_c = average_precision_score(yte, pc)
    assert ap_n >= ap_c - 1e-6, f"expected naive>=censored, got naive={ap_n:.3f} censored={ap_c:.3f}"
    print(f"OK: leakage detectable (naive AUPRC={ap_n:.3f} >= censored AUPRC={ap_c:.3f})")




def test_icu_features_respect_the_cutoff():
    """ICU features at a cut-off must ignore measurements recorded after it."""
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import audit_physionet2012 as E

    long = pd.DataFrame({"record_id": [1, 1, 1, 2, 2],
                         "minute": [60, 1440, 1500, 30, 2000],
                         "param": ["HR", "HR", "HR", "HR", "HR"],
                         "value": [80.0, 100.0, 150.0, 70.0, 200.0]})
    static = pd.DataFrame({"Age": [60, 70], "Gender": [1, 0], "Height": [170, 160], "ICUType": [1, 2],
                           "Weight_admission": [80, 60]}, index=pd.Index([1, 2], name="record_id"))
    X = E.features(long, static, 24 * 60, ["HR"])
    assert X.loc[1, "HR_max"] == 100.0 and X.loc[1, "HR_count"] == 2   # the 150 at minute 1500 is excluded
    assert X.loc[2, "HR_max"] == 70.0 and X.loc[2, "HR_count"] == 1    # the 200 at minute 2000 is excluded
    X48 = E.features(long, static, 48 * 60, ["HR"])
    assert X48.loc[1, "HR_last"] == 150.0 and X48.loc[2, "HR_count"] == 2


if __name__ == "__main__":
    test_censoring_matches_manual_counts()
    test_leakage_is_detectable()
    test_icu_features_respect_the_cutoff()
    print("all tests passed")
