"""Temporal splitting with target-level clustering.

Random splits are forbidden here: the whole question is prospective, so we
train on the past and test on the future (forward-chaining / rolling origin).
Because programs reuse a smaller pool of targets, we also enforce that a target
appearing in training never appears in the test block -- otherwise the model
can memorise a gene rather than generalise.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def temporal_split(
    meta: pd.DataFrame,
    train_max_year: int,
    test_years: Tuple[int, int] | List[int],
    enforce_target_disjoint: bool = True,
) -> Tuple[pd.Index, pd.Index, Dict]:
    """One forward-chaining split.

    train : info_time <= train_max_year
    test  : min(test_years) <= info_time <= max(test_years)
    """
    lo, hi = min(test_years), max(test_years)
    train_idx = meta.index[meta["info_time"] <= train_max_year]
    test_mask = (meta["info_time"] >= lo) & (meta["info_time"] <= hi)
    test_idx = meta.index[test_mask]

    dropped = 0
    if enforce_target_disjoint:
        train_targets = set(meta.loc[train_idx, "target_id"])
        keep = ~meta.loc[test_idx, "target_id"].isin(train_targets)
        dropped = int((~keep).sum())
        test_idx = test_idx[keep.values]

    info = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_dropped_shared_target": dropped,
        "train_max_year": int(train_max_year),
        "test_years": [int(lo), int(hi)],
    }
    return train_idx, test_idx, info


def sealed_prospective(
    meta: pd.DataFrame, seal_year: int, enforce_target_disjoint: bool = True
) -> Tuple[pd.Index, pd.Index, Dict]:
    """Train on <= seal_year, test on the sealed future block (> seal_year).

    Use this exactly once, at the very end. Do not tune against it.
    """
    train_idx = meta.index[meta["info_time"] <= seal_year]
    test_idx = meta.index[meta["info_time"] > seal_year]

    dropped = 0
    if enforce_target_disjoint:
        train_targets = set(meta.loc[train_idx, "target_id"])
        keep = ~meta.loc[test_idx, "target_id"].isin(train_targets)
        dropped = int((~keep).sum())
        test_idx = test_idx[keep.values]

    info = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_dropped_shared_target": dropped,
        "seal_year": int(seal_year),
    }
    return train_idx, test_idx, info


def rolling_origin(
    meta: pd.DataFrame, cutoffs: List[int], window: int = 1, enforce_target_disjoint: bool = True
):
    """Yield (train_idx, test_idx, info) for each cutoff (temporal cross-validation)."""
    for c in cutoffs:
        yield temporal_split(
            meta, train_max_year=c, test_years=(c + 1, c + window),
            enforce_target_disjoint=enforce_target_disjoint,
        )
