"""Central configuration.

Defaults live here so the project runs with zero setup. Any field can be
overridden by passing a YAML file to ``get_config(path)``; only the keys you
specify are overridden (recursive merge of nested dictionaries). The YAML files
that reproduce the paper's datasets are in ``config/``.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "seed": 7,
    "data_dir": "data/synth",      # where programs.csv / evidence.csv live
    "output_dir": "outputs",       # where reports / plots are written

    # Synthetic-data generator. Ignored once you plug in real data.
    "synth": {
        "n_programs": 4000,
        "n_targets": 3500,          # < n_programs => genes are reused (non-independence);
                                    # kept moderate so target-disjoint splits stay evaluable
        "n_areas": 8,
        "year_min": 2008,
        "year_max": 2020,
        "true_genetic_logodds": 0.35,  # the *causal* effect of genetic support (small, on purpose)
        "leak_strength": 2.4,          # how strongly post-hoc literature tracks the label
        "base_intercept": -0.35,       # tunes overall base success rate
    },

    # Real-data connectors (temporal_leakage_audit/data/connectors.py). Ignored on synthetic data.
    # These drive live pulls from Open Targets GraphQL + ClinicalTrials.gov v2 +
    # PubMed. See connectors.py for the full meaning of each key and the
    # modelling decisions they control.
    "real": {
        "ot_graphql_url": "https://api.platform.opentargets.org/api/v4/graphql",
        "ctgov_v2_url": "https://clinicaltrials.gov/api/v2/studies",
        "pubmed_esummary_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        "ncbi_api_key": None,               # optional; raises PubMed rate limit to 10/s
        "ncbi_tool": "temporal-leakage-audit",  # E-utilities `tool` parameter (NCBI usage policy)
        "ncbi_email": None,                 # optional E-utilities `email` parameter

        # Disease universe to pull -> all clinical drug candidates for each. Entries
        # may be current Open Targets disease ids (EFO_*, MONDO_*, HP_*) OR plain disease
        # names (the connector resolves either via OT search, so stale ids self-correct).
        # Open Targets migrated many diseases from EFO_* to MONDO_* in the 26.x line.
        # A small example set; the paper's 20- and 41-disease sets are in config/.
        # (The legacy key name ``efo_ids`` is still accepted.)
        "disease_ids": [
            "MONDO_0008383",   # rheumatoid arthritis
            "MONDO_0004989",   # breast carcinoma
            "MONDO_0005148",   # type 2 diabetes mellitus
            "MONDO_0004979",   # asthma
            "Alzheimer disease",       # resolved by name via OT search
            "Crohn disease",           # resolved by name via OT search
        ],
        "max_drug_rows": None,   # cap drug candidates per disease (None = all)

        # Schema drift guard. OT ships bi-monthly; warn (don't fail) if it moves.
        "expected_data_version": {"year": "26", "month": "06"},
        "check_version": True,

        # Literature-evidence dating -- the load-bearing leakage choice.
        #   "pubmed_year"        : year = min PubMed pub-year of the datum's PMIDs;
        #                          drop the datum if no PMID resolves (recommended).
        #   "drop"               : drop all literature evidence entirely.
        #   "ot_publication_year": use Open Targets' publicationYear only; drop nulls.
        "literature_policy": "pubmed_year",
        "use_pmid_year_fallback": True,     # also backfill non-lit years from PMIDs
        # Literature co-occurrence evidence can run to tens of thousands of rows per
        # (disease, target) pair; cap it with a seeded random sample (None = keep all).
        "max_literature_per_pair": None,

        # Program-universe modelling knobs.
        "drop_unmapped_targets": True,      # drop programs whose drug has no Ensembl target

        # Transport.
        "evidence_size": 3000,              # per-page cap on disease.evidences (hard max 3000)
        "pubmed_batch": 200,
        "request_timeout": 60,
        "max_retries": 5,
        "backoff_base": 0.5,                # seconds; exponential backoff base
        "sleep_between": 0.1,               # polite pacing between requests
        "cache_dir": "data/api_cache",    # on-disk JSON cache; None disables
    },

    # Which column is the "treatment" for the causal layer.
    "task": {"treatment_col": "has_genetic_support"},

    # Temporal split. Nothing dated after a program's info_time is ever used as a feature.
    "split": {
        "train_max_year": 2016,     # train: info_time <= this
        "test_years": [2017, 2018], # rolling-origin evaluation block
        "seal_year": 2018,          # sealed prospective test: info_time > this (touched once)
        "enforce_target_disjoint": True,  # a gene may not appear in both train and test
    },

    # Decision / policy evaluation.
    "decision": {
        "budget_frac": 0.20,        # you can advance this fraction of programs
        "pt_for_netbenefit": 0.20,  # threshold probability for the headline net-benefit number
        "pt_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    },

    "bootstrap": {"n_boot": 500},   # target-clustered bootstrap resamples for CIs
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the default configuration, optionally overridden by a YAML file."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml  # optional; only needed if you pass a YAML override
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    return cfg
