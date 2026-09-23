"""Feature construction with strict as-of-t censoring.

Two feature matrices are produced from the same data:

  * CENSORED ("as-of-t"): for each program, only evidence with
    ``evidence_date <= info_time`` is aggregated. This is the deployable matrix.
  * NAIVE: all evidence is aggregated regardless of date. This is what a
    time-blind pipeline would use, and comparing the two is how leakage is
    measured (see leakage.py).

Feature construction is per-program and deterministic, so it is safe to build
on the full table and split afterwards -- no information flows between rows.
Only *model fitting* is ever restricted to the training split.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

EVIDENCE_TYPES = [
    "genetic", "somatic", "known_drug", "pathway",
    "animal_model", "expression", "literature",
]
GENETIC_TYPES = {"genetic", "somatic"}
LITERATURE_TYPES = {"literature"}


def _aggregate_evidence(evidence: pd.DataFrame, program_ids: pd.Index) -> pd.DataFrame:
    """Aggregate an (already date-filtered) evidence log to per-program features."""
    if len(evidence) == 0:
        agg = pd.DataFrame(index=program_ids)
    else:
        g = evidence.groupby(["program_id", "evidence_type"])["score"]
        wide_sum = g.sum().unstack("evidence_type")
        wide_cnt = g.count().unstack("evidence_type")
        wide_sum.columns = [f"ev_{c}_score" for c in wide_sum.columns]
        wide_cnt.columns = [f"ev_{c}_count" for c in wide_cnt.columns]
        agg = wide_sum.join(wide_cnt, how="outer")
        agg = agg.reindex(program_ids)

    # Ensure every evidence type has both columns, even if absent in the data.
    for t in EVIDENCE_TYPES:
        for suffix in ("score", "count"):
            col = f"ev_{t}_{suffix}"
            if col not in agg.columns:
                agg[col] = 0.0
    agg = agg.fillna(0.0)
    agg = agg[[f"ev_{t}_{s}" for t in EVIDENCE_TYPES for s in ("score", "count")]]
    return agg


def _structural(programs: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Always-available, non-leaky program features + a stable dummy layout."""
    p = programs.set_index("program_id")
    base = p[["sponsor_tier", "phase_from", "info_time"]].astype(float)
    area = pd.get_dummies(p["therapeutic_area"], prefix="area").astype(float)
    mod = pd.get_dummies(p["modality"], prefix="mod").astype(float)
    struct = base.join(area).join(mod)
    return struct, list(struct.columns)


def build(
    programs: pd.DataFrame, evidence: pd.DataFrame, horizon: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, List[str]]]:
    """Build censored + naive feature matrices.

    ``horizon`` is the hindsight window (in years) allowed into the "censored"
    matrix: evidence with ``evidence_date <= info_time + horizon`` is included.
    ``horizon=0`` (default) is the deployable as-of-t matrix; ``horizon=inf`` reproduces
    the naive matrix. Sweeping the horizon traces the temporal leakage-response
    curve (see leakage.leakage_response_curve). The X_naive output is always the
    full-hindsight matrix regardless of ``horizon``.

    Returns
    -------
    X_censored, X_naive : DataFrames indexed by program_id (same columns/order)
    y                   : label Series
    meta                : DataFrame with target_id, info_time, therapeutic_area
    groups              : dict mapping feature-group name -> column list
    """
    programs = programs.copy()
    ev = evidence.merge(programs[["program_id", "info_time"]], on="program_id", how="left")

    ev_censored = ev[ev["evidence_date"] <= ev["info_time"] + horizon].drop(columns="info_time")
    ev_naive = ev.drop(columns="info_time")

    pid_index = programs.set_index("program_id").index

    agg_c = _aggregate_evidence(ev_censored, pid_index)
    agg_n = _aggregate_evidence(ev_naive, pid_index)
    struct, struct_cols = _structural(programs)

    # has_genetic_support (the causal "treatment") -- computed as-of-t.
    gen_cols = [f"ev_{t}_count" for t in GENETIC_TYPES]
    has_gen_c = (agg_c[gen_cols].sum(axis=1) > 0).astype(float)
    has_gen_n = (agg_n[gen_cols].sum(axis=1) > 0).astype(float)

    X_censored = struct.join(agg_c)
    X_censored["has_genetic_support"] = has_gen_c
    X_naive = struct.join(agg_n)
    X_naive["has_genetic_support"] = has_gen_n

    # Keep identical column order.
    X_naive = X_naive[X_censored.columns]

    y = programs.set_index("program_id")["label"].astype(int)
    meta = programs.set_index("program_id")[["target_id", "info_time", "therapeutic_area"]]

    genetic_cols = [f"ev_{t}_{s}" for t in GENETIC_TYPES for s in ("score", "count")] + ["has_genetic_support"]
    literature_cols = [f"ev_{t}_{s}" for t in LITERATURE_TYPES for s in ("score", "count")]
    other_ev_cols = [
        f"ev_{t}_{s}"
        for t in EVIDENCE_TYPES
        if t not in GENETIC_TYPES and t not in LITERATURE_TYPES
        for s in ("score", "count")
    ]
    groups = {
        "structural": struct_cols,
        "genetic": genetic_cols,
        "literature": literature_cols,
        "other_evidence": other_ev_cols,
        "confounders": struct_cols,         # used by the causal layer
    }
    return X_censored, X_naive, y, meta, groups
