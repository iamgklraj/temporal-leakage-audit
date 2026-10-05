"""Unit test for the separated naive representation of scripts/validate_comparators.py.

Run with either:  pytest tests/  OR  python tests/test_comparator_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import validate_comparators as VC  # noqa: E402
from temporal_leakage_audit import features  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402


def test_x_post_contains_only_post_decision_evidence():
    programs = pd.DataFrame({
        "program_id": ["A", "B"], "target_id": ["T1", "T2"], "indication": ["x_1", "y_1"],
        "therapeutic_area": ["oncology", "neurology"], "sponsor_tier": [0, 1],
        "modality": ["biologic", "small_molecule"], "info_time": [2015, 2018],
        "phase_from": [1, 2], "label": [1, 0]})
    evidence = pd.DataFrame([
        ("A", "literature", 0.9, 2014),   # before the decision: pre-decision
        ("A", "literature", 0.7, 2015),   # on the decision date: available, so pre-decision
        ("A", "literature", 0.4, 2016),   # after: post-decision
        ("A", "genetic", 0.5, 2017),      # after: post-decision
        ("B", "pathway", 0.2, 2018),      # on the decision date: pre-decision
    ], columns=["program_id", "evidence_type", "score", "evidence_date"])
    post = VC.post_block(programs, evidence)
    assert post.loc["A", "post_ev_literature_count"] == 1
    assert np.isclose(post.loc["A", "post_ev_literature_score"], 0.4)
    assert post.loc["A", "post_ev_genetic_count"] == 1
    assert post.loc["A"].filter(like="_count").sum() == 2 and post.loc["B"].sum() == 0

    cfg = get_config()
    cfg["seed"] = 3
    cfg["synth"].update(n_programs=200, n_targets=150)
    for P, E in ((programs, evidence), synth.generate(cfg)):
        post = VC.post_block(P, E)
        dated = E.merge(P[["program_id", "info_time"]], on="program_id")
        # X_post counts exactly the evidence dated after the decision, nothing else
        assert post.filter(like="_count").values.sum() == (dated["evidence_date"] > dated["info_time"]).sum()
        # and X(0) + X_post = X(inf) on every evidence aggregate
        Xc, Xn, _, _, _ = features.build(P, E)
        ev = [c for c in Xc.columns if c.startswith("ev_")]
        np.testing.assert_allclose(Xc[ev].values + post[[VC.POST + c for c in ev]].values, Xn[ev].values)


if __name__ == "__main__":
    test_x_post_contains_only_post_decision_evidence()
    print("ok")
