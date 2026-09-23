# temporal-leakage-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22907786.svg)](https://doi.org/10.5281/zenodo.22907786)

Code for the paper *Auditing temporal leakage in clinical trial outcome prediction*. It
provides a **temporal leakage-response instrument** for machine-learning models
that predict clinical-trial outcomes and drug phase advancement, audits of two public benchmarks
(CTO and TrialBench), a study of memorization and knowledge leakage in language-model predictors, an
as-of-time-censored drug-program benchmark, and a positivity-aware causal layer.

The question it answers: *how much of a model's reported skill depends on information that did
not exist at the decision time?* The instrument refits the same model as post-decision
information is admitted (a hindsight horizon *h*), traces performance against *h*, and reports
the leakage-attributable performance, LAP = naive − deployable, with cluster-bootstrap
confidence intervals.

## Main results

All intervals are 95% percentile bootstrap CIs; the clustering unit is given per row.

| Analysis | Result | Output |
|---|---|---|
| **Instrument validation** (synthetic data, known leakage) | Two mechanisms (post-decision literature scores; extra post-decision papers), 100 replicates per level: without leakage the CI lies above 0 in 2–3% of replicates (nominal 2.5%); LAP rises monotonically with true leakage; detection 55% at LAP ≈ 0.06 and 96% at ≈ 0.11 under both mechanisms; CI coverage 83–95%. | `outputs/instrument_validation.json` |
| **CTO labeling signals** (Gao et al., *Nature Health* 2026) | CTO distributes its labeling-function outputs with its labels. Used as features against curated human labels on a temporal split (*n* = 9,705; test base rate 0.17), all signals reach AUPRC 0.98 but start-time signals only 0.23 (AUROC 0.64). LAP = +0.75 [0.71, 0.79] (sponsor-clustered): ~77% depends on information unavailable at trial start (during-trial +0.46, post-completion +0.30). CTO's own baselines use design-time text features only (AUROC 0.55–0.62) and avoid this hazard; the start-time estimate corroborates them. | `outputs/cto_audit.json` |
| **TrialBench** (Chen et al., *Sci. Data* 2025) | The provided split is not temporal (median start year 2011 in both train and test; one-sided Mann–Whitney *P* = 0.60, *n* = 25,972). A temporal split lowers AUPRC (mean −0.08) but *raises* AUROC (mean +0.03): the drop tracks a lower approval rate among recent trials (outcome immaturity), not inflated discrimination (trial-level CIs). | `outputs/trialbench_audit.json` |
| **Language models** (4 Claude models + Llama 3.1 8B; *n* = 600 CTO trials, 108 successes; tools disabled) | Five nested prompts. Design only: AUPRC 0.20–0.29 (base rate 0.18). Registry identifier alone adds ≤ +0.03. Masked title (design semantics) adds +0.02 → +0.16 with model tier; the named intervention adds a further +0.07 to +0.13 per Claude model (pre-specified primary contrast; significant after Holm correction for 3 of 4 Claude models). Dated against ChEMBL approval histories, that gain concerns mostly drugs approved before trial start; only Opus 5 shows a residual on never-approved compounds (+0.07 [0.005, 0.14]). Llama 3.1 8B gains nothing. | `outputs/llm_memorization_study.json`, `outputs/llm_predictions.csv`, `outputs/llm_raw/` |
| **Censored drug-program benchmark** (negative control) | Built with strict as-of-time censoring (3,990 programs, 20 diseases): LAP = +0.074 [−0.07, 0.16] (target-clustered), at the instrument's detection limit for 200 test programs. At 41 diseases (8,357 programs), the two largest rolling-origin test blocks give +0.09 and +0.08 (CIs exclude 0). | `outputs/censored_benchmark_{20,41}disease.json` |
| **Dating-rule sensitivity** | Same 20-disease programs re-dated by Open Targets publication year instead of earliest PubMed year: LAP +0.074 → +0.050 (both CIs span 0). | `outputs/dating_rule_sensitivity.json` |
| **Causal layer** (genetic support → advancement) | Naive difference +0.14; the ATE is not identified (positivity violation; AIPW ATE −0.005 [−0.21, 0.15]). Overlap-weighted ATO = +0.12 [0.01, 0.23] (target-clustered), max \|SMD\| 0.53 → 0.01, outcome-permutation *P* = 0.014, E-value 1.9 (CI limit 1.2). At 41 diseases: ATO = +0.10 [0.02, 0.18]. | `outputs/censored_benchmark_20disease.json` |

## Repository layout

```
temporal_leakage_audit/        the library (importable package)
  features.py                  as-of-time-censored vs naive feature matrices
  leakage.py                   leakage-response curve, LAP, per-source placebo, clustered bootstrap
  causal.py                    cross-fitted AIPW, overlap-weighted ATO, balance, permutation test, E-values
  decision.py                  policy value at a budget, net benefit, go/no-go gate
  splits.py                    temporal, target-disjoint splits
  metrics.py  models.py  config.py  report.py
  data/connectors.py           Open Targets GraphQL + ClinicalTrials.gov v2 + PubMed (cached)
  data/synth.py                synthetic testbed with known leakage and confounding
scripts/                       one script per analysis (see "Reproducing the paper")
tests/                         unit tests (censoring invariant, connectors, estimators)
config/                        YAML configs; benchmark_*.yaml rebuild the paper's datasets
outputs/                       reference results (JSON) reported in the paper
figures/                       paper figures, regenerated by scripts/make_figures.py
```

## System requirements

- Python ≥ 3.10. Tested with Python 3.13.5 on macOS 26.6 (Apple M4 Pro, 24 GB RAM) with the exact
  package versions in `requirements.txt` (numpy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0,
  scipy 1.18.0, matplotlib 3.11.1, PyYAML 6.0.3).
- No GPU or other non-standard hardware.
- Internet access only for (re)building datasets from the public APIs and for the first
  download of the CTO and TrialBench tables.
- The LLM study additionally needs the Claude Code CLI (`claude`) with access to the models, and
  [Ollama](https://ollama.com) with `llama3.1:8b` for the open-weight comparison.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact tested versions
pip install -e .                  # makes `temporal_leakage_audit` importable
```

Typical install time: under 2 minutes on a standard desktop with a broadband connection.

## Demo (synthetic data, ~10 s)

```bash
python scripts/make_synth.py            # writes data/synth/ (4,000 programs, 22,642 evidence rows)
python scripts/run_synth_benchmark.py   # full pipeline on the sealed synthetic test block
```

Expected output (abridged; exact with the pinned versions): the naive model reaches AUPRC 0.947
and the censored model 0.810, so `LAP.auprc=+0.136`; the causal layer reports a naive association
of +0.131 and an AIPW ATE of +0.066 [0.027, 0.104]; the decision gate prints `VERDICT: GO`.
Results are written to `outputs/synth_demo/`. Run time: about 10 s.

## Reproducing the paper

Each script writes one JSON file to `outputs/`; `scripts/make_figures.py` redraws every figure
from those files. Run times were measured on the test machine above.

| Paper item | Command | Run time |
|---|---|---|
| Unit tests | `python tests/test_no_leakage.py && python tests/test_connectors.py && python tests/test_estimators.py` (or `pytest tests/`) | ~20 s |
| Fig. 1 (instrument validation) | `python scripts/validate_instrument.py` | ~1 min (8 workers) |
| Table 1, Fig. 2 (CTO) | `python scripts/audit_cto.py` (downloads four CTO tables from Hugging Face on first run) | ~3 min |
| Fig. 3 (TrialBench) | `python scripts/audit_trialbench.py --check-chronology` (downloads features from Zenodo and labels from GitHub; fetches start dates from ClinicalTrials.gov) | ~1.5 min |
| Fig. 4 (language models) | `python scripts/llm_memorization_study.py --models <model> --workers 3` per model (Claude via the Claude Code CLI with tools disabled; `ollama:llama3.1:8b` via local Ollama); `--analyze-only` recomputes every summary from the saved responses in `outputs/llm_raw/` | ~1–2 h per Claude model; ~40 min for Llama |
| Fig. 4c (dating the language models' drug knowledge) | `python scripts/llm_knowledge_dating.py` (ChEMBL lookups cached in `outputs/llm_sample_chembl.json`) | ~10 min first run; seconds when cached |
| Build the drug-program benchmarks | `python -m temporal_leakage_audit.data.connectors config/benchmark_20disease.yaml` (likewise `benchmark_41disease.yaml`) | hours (API-bound; cached in `data/api_cache/`) |
| Censored benchmark, Extended Data Fig. 1 | `python scripts/audit_censored_benchmark.py --config config/benchmark_20disease.yaml --out outputs/censored_benchmark_20disease.json` (likewise 41) | ~10–30 s each |
| Dating-rule sensitivity | `python scripts/dating_rule_sensitivity.py --build data/benchmark_20disease data/benchmark_20disease_otyear` then `--compare data/benchmark_20disease data/benchmark_20disease_otyear` | ~30 s (compare) |
| All figures | `python scripts/make_figures.py --out figures` | ~5 s |

**Data provenance.** Open Targets Platform release 26.06 (the connector warns if the live
release differs), ClinicalTrials.gov API v2, NCBI PubMed E-utilities; CTO from Hugging Face
(`chufangao/CTO`); TrialBench features from Zenodo (doi:10.5281/zenodo.14975339) and labels from
`ML2Health/ML2ClinicalTrials` at commit `0694eba`. Because the live APIs change, the processed
tables used for the paper are deposited on Zenodo
([10.5281/zenodo.22907882](https://doi.org/10.5281/zenodo.22907882)); unzip them and place the
folders under `data/` to reproduce the reference outputs exactly.

**Determinism.** The gradient-boosted learner and all bootstraps use fixed seeds, so the
reference outputs are reproduced exactly from the same data and package versions. The LLM study
is stochastic (models are queried with default sampling settings).

## Using the instrument on your own data

Provide two tables with the schema documented in `temporal_leakage_audit/data/connectors.py`:
`programs.csv` (one row per prediction unit with a decision time `info_time` and a binary
`label`) and `evidence.csv` (dated feature events: `program_id, evidence_type, score,
evidence_date`). Then:

```python
from temporal_leakage_audit import features, leakage, splits
Xc, Xn, y, meta, groups = features.build(programs, evidence)       # censored and naive matrices
tr, te, _ = splits.temporal_split(meta, train_max_year=2014, test_years=(2015, 2026))
curve = leakage.leakage_response_curve(programs, evidence, tr, te,
                                       clusters=meta.loc[te, "target_id"].values)
print(curve["total_LAP_auprc"], curve["total_LAP_auprc_ci"])
```

## Known limitations

- Language-model outputs are stochastic (default sampling for Claude models), so re-running
  the queries gives slightly different numbers; `--analyze-only` reproduces the reported ones
  exactly from the saved responses. An earlier exploratory LLM run (September 7, 2026), whose tool
  access was not controlled, is archived in `outputs/archive/` and superseded.
- Open Targets' clinical-stage-to-phase mapping is inferred (see `connectors.py`), and coverage
  is biased toward trials registered on ClinicalTrials.gov.

## Citation

Archived on Zenodo: [10.5281/zenodo.22907786](https://doi.org/10.5281/zenodo.22907786) (all
versions); v1.0.0, which produced the paper's results, is
[10.5281/zenodo.22907787](https://doi.org/10.5281/zenodo.22907787). See `CITATION.cff` (GitHub
renders it as a "Cite this repository" button).

## License

MIT — see [LICENSE](LICENSE).
