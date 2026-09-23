"""Synthetic drug-development data with realistic pathologies baked in.

The point of the synthetic data is NOT to be pretty -- it is to make the whole
scientific argument testable end-to-end before touching real data. It encodes:

  1. LEAKAGE. "literature" evidence mostly carries dates AFTER a program's
     information-time (i.e. papers that appeared *because* the program later
     succeeded) and its score/count tracks the label. A model that ignores
     dates therefore looks strong; a time-censored model does not.

  2. CONFOUNDING. Whether a program has genetic support depends on sponsor tier
     and therapeutic area, which also drive outcomes. So the naive association
     between genetic support and success overstates the *causal* effect.

  3. A WEAK TRUE SIGNAL. The genuine causal effect of genetic support is small
     (``true_genetic_logodds``), and the only legitimately available predictive
     signal (structural features + some pre-dated evidence) is modest.

  4. NON-INDEPENDENCE. Programs reuse a smaller pool of targets (genes), so
     target-level clustering matters for both splitting and significance.

Output schema (identical to the real-data contract in data/connectors.py):

  programs.csv : program_id, target_id, indication, therapeutic_area,
                 sponsor_tier, modality, info_time, phase_from, label
  evidence.csv : program_id, evidence_type, score, evidence_date

``label`` = 1 if the program advanced from ``phase_from`` to the next phase
(resolved *after* ``info_time``); ``info_time`` is the year features are frozen.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

AREAS = [
    "oncology", "immunology", "neurology", "cardiovascular",
    "metabolic", "infectious", "respiratory", "rare_disease",
]
MODALITIES = ["small_molecule", "biologic", "other"]
NON_LEAKY_TYPES = ["pathway", "animal_model", "expression", "known_drug"]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = cfg["synth"]
    rng = np.random.default_rng(cfg["seed"])

    n = int(s["n_programs"])
    n_targets = int(s["n_targets"])
    n_areas = min(int(s["n_areas"]), len(AREAS))
    y0, y1 = int(s["year_min"]), int(s["year_max"])
    true_gen = float(s["true_genetic_logodds"])
    leak = float(s["leak_strength"])
    intercept = float(s["base_intercept"])

    # Fixed per-area log-odds effects (stable across a run).
    area_rng = np.random.default_rng(cfg["seed"] + 1)
    area_effect = area_rng.normal(0.0, 0.5, size=n_areas)

    tier_effect = np.array([-0.20, 0.10, 0.40])      # small / mid / large sponsor
    phase_effect = np.array([-0.10, 0.20, 0.50])     # phase 1 / 2 / 3

    prog_rows = []
    evid_rows = []

    for i in range(n):
        pid = f"P{i:05d}"
        area = int(rng.integers(0, n_areas))
        tier = int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
        year = int(rng.integers(y0, y1 + 1))
        modality = str(rng.choice(MODALITIES, p=[0.6, 0.3, 0.1]))
        phase_from = int(rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25]))
        target_id = f"T{int(rng.integers(0, n_targets)):04d}"

        # --- Confounded treatment: has_genetic_support depends on tier & area ---
        logit_gen = -0.5 + 0.6 * (tier - 1) + 0.5 * area_effect[area]
        has_gen = bool(rng.random() < _sigmoid(logit_gen))

        year_trend = (year - y0) * 0.02

        # --- True outcome model (label sampled from this) ---
        logit_y = (
            intercept
            + area_effect[area]
            + tier_effect[tier]
            + phase_effect[phase_from - 1]
            + year_trend
            + true_gen * (1.0 if has_gen else 0.0)
            + rng.normal(0.0, 0.30)
        )
        p_y = float(_sigmoid(logit_y))
        label = int(rng.random() < p_y)

        prog_rows.append(
            dict(
                program_id=pid,
                target_id=target_id,
                indication=f"{AREAS[area]}_{int(rng.integers(0, 40))}",
                therapeutic_area=AREAS[area],
                sponsor_tier=tier,
                modality=modality,
                info_time=year,
                phase_from=phase_from,
                label=label,
            )
        )

        # --- Genetic evidence: dated on/before info_time (legitimately available) ---
        if has_gen:
            for _ in range(int(rng.integers(1, 4))):
                d = year - int(rng.integers(0, 5))
                d = max(d, y0)
                evid_rows.append((pid, "genetic", float(rng.uniform(0.3, 0.9)), d))

        # --- Non-leaky evidence: pre-dated, mildly informative via the true logit ---
        pre_decision_signal = logit_y - rng.normal(0.0, 0.30)  # strip the label noise
        for etype in NON_LEAKY_TYPES:
            if rng.random() < 0.45:
                d = year - int(rng.integers(0, 6))
                d = max(d, y0)
                score = float(np.clip(0.5 + 0.15 * pre_decision_signal + rng.normal(0, 0.25), 0, 1))
                evid_rows.append((pid, etype, score, d))

        # --- Literature evidence: LEAKY. Mostly post-dated; score tracks the label ---
        n_lit = int(rng.integers(0, 6)) + (2 if label == 1 else 0)  # success attracts papers
        for _ in range(n_lit):
            post_hoc = rng.random() < 0.80
            if post_hoc:
                d = year + int(rng.integers(0, 6))   # appears AFTER the decision point
            else:
                d = year - int(rng.integers(0, 3))   # a minority genuinely predates it
                d = max(d, y0)
            score = float(np.clip(0.3 + 0.1 * leak * label + rng.normal(0, 0.2), 0, 1))
            evid_rows.append((pid, "literature", score, d))

    programs = pd.DataFrame(prog_rows)
    evidence = pd.DataFrame(evid_rows, columns=["program_id", "evidence_type", "score", "evidence_date"])
    return programs, evidence


def write(cfg: Dict) -> Tuple[str, str]:
    programs, evidence = generate(cfg)
    out_dir = cfg["data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    p_path = os.path.join(out_dir, "programs.csv")
    e_path = os.path.join(out_dir, "evidence.csv")
    programs.to_csv(p_path, index=False)
    evidence.to_csv(e_path, index=False)
    return p_path, e_path
