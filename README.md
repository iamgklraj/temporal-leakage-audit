# temporal-leakage-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22907786.svg)](https://doi.org/10.5281/zenodo.22907786)

Code for the paper *Foresight or hindsight? Auditing temporal leakage in clinical trial outcome prediction*. It
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
| **Instrument validation** (synthetic data, known leakage) | Default setting (≈245 test programs, prevalence ≈0.5; 500 replicates): without leakage the CI lies above 0 in 3.4% and below 0 in 2.6% of replicates; detection 96% at LAP ≈ 0.11; coverage 89–94% (a variance-inflated interval restores 94–99%). Setting matched to the 20-disease benchmark (≈205 programs, prevalence 0.15; 3,000 null replicates): 2.8% / 3.2%; coverage 92–95%; 50% detection needs LAP ≈ 0.13. | `outputs/instrument_validation*.json` |
| **CTO labeling signals** — positive control (Gao et al., *Nature Health* 2026) | CTO's curated labels are largely defined by the same status and *P*-value rule as its labeling functions (76% of trials). The 22 columns hold 17 distinct signals, tiered by when they become public. All signals reach AUPRC 0.98; LAP = +0.75 [0.71, 0.79] (sponsor-clustered), almost all from registry fields available at completion (+0.74). Start-time signals are uninformative once withdrawn trials are excluded (AUROC 0.51). | `outputs/cto_audit_v2.json` |
| **TrialBench** (Chen et al., *Sci. Data* 2025) | Features include **actual enrollment** (updated at trial completion) and final-record city covariates: on the temporal test set they add +0.21, +0.16, +0.12 AUPRC (Phases I–III; paired CIs exclude 0). The provided split is not temporal, but holding the test set fixed it adds only 0.005–0.017 AUROC; gaps between splits reflect prevalence and immature labels. | `outputs/trialbench_audit_v2.json`, `outputs/trialbench_lap_poststart.json` |
| **Language models** (4 Claude models + Llama 3.1 8B; *n* = 600 CTO trials, 108 successes; tools disabled) | Five nested prompts (the design condition includes an actual-enrollment bin). Identifier alone adds ≤ +0.03. Masked title adds +0.02 → +0.16 with model tier; the named intervention adds +0.07 to +0.13 per Claude model (Holm-significant for 3 of 4 with trial-level and 2 of 4 with batch-level bootstrap; the two primary contrasts were designated after the responses were collected, so Holm adjustment is a robustness check), robust to excluding COVID-19, imperfectly masked and withdrawn trials. The gain appears for drugs approved before trial start and, for both Opus models, for never-approved development-code compounds (+0.08). All trials completed before each Claude model's documented training cutoff; for Llama 3.1 8B (cutoff Dec 2023), 148 trials completed later but only 7 were successes, too few for an informative post-cutoff comparison. Llama 3.1 8B gains nothing. | `outputs/llm_memorization_study.json`, `outputs/llm_validity_checks.json`, `outputs/llm_raw/` |
| **Open-weight language models** (exploratory; Gemma 3 12B, Phi-4 14B; same 600 trials and prompts) | Neither gains from identifiers or named interventions (named-intervention contrasts +0.009 and −0.021; CIs span 0), like Llama 3.1 8B: drug-name knowledge appears only in the larger Claude models. No sampled trial completing after either model's documented cutoff (Aug/Jun 2024) was a success, so no post-cutoff comparison was possible. Analyzed separately so the primary tests are unchanged. | `outputs/llm_open_models.json`, `outputs/llm_raw/ollama_*` |
| **Censored drug-program benchmark** — negative control | Strict as-of-time censoring (3,990 programs, 20 diseases): LAP = +0.074 [−0.07, 0.16] (target-clustered); post-decision evidence adds at most ≈0.16 AUPRC. Learner-dependent (logistic regression +0.11, random forest −0.04) and stable to label maturity; at 41 diseases only the two largest rolling-origin blocks exclude 0. | `outputs/censored_benchmark_{20,41}disease.json`, `outputs/sensitivity_reviewer.json`, `outputs/robustness.json` |
| **ICU records** — patient-level demonstration (PhysioNet/CinC Challenge 2012, 12,000 stays) | For a mortality model meant for use 24 h after ICU admission, admitting hours 24–48 raises test AUPRC from 0.487 to 0.594: LAP = +0.107 [0.079, 0.135] (patient-level), mostly from later measured values (+0.093; Glasgow Coma Scale +0.049) rather than measurement counts (+0.015). Decision at 12 h: LAP = +0.148 [0.114, 0.181]. | `outputs/physionet2012_audit.json` |
| **Sepsis onset** — second patient-level demonstration (PhysioNet/CinC Challenge 2019, 40,336 stays; train hospital system A, test B) | At a decision 24 h after ICU admission (14,823 eligible test stays, 212 onsets within 24 h), the deployable model reaches AUPRC 0.035; admitting the next 24 h raises it to 0.124: LAP = +0.089 [0.063, 0.128] (patient-level), from later measured values (+0.050, led by vital signs) and measurement counts (+0.033; the challenge truncates septic records within 3 h of onset). Decision at 12 h: LAP = +0.139 [0.100, 0.182]. Design fixed before any model was fitted. | `outputs/physionet2019_sepsis_audit.json` |
| **TOP** (HINT; Fu et al., *Patterns* 2022) | The provided split is temporal by start date (every test trial starts after the last training trial; dates for 99.8% of 12,477 trials), but the files distribute final status and reasons for stopping with the labels. | `outputs/top_audit.json` |
| **Decision-level impact** | Precision among the top-ranked 10%: TrialBench +7 to +21 percentage points with actual enrollment (Phase I 0.67 vs 0.46); ICU mortality 0.61 vs 0.55 with 48 vs 24 hours of data. | `outputs/decision_impact.json` |
| **Comparison with common leakage checks** | Same data and learner. A random split gives CTO's signals the same AUPRC as the temporal split (0.985 vs 0.981), so a temporal split neither reveals nor removes feature leakage. A univariate "too good to be true" screen (training AUROC ≥ 0.90) flags only CTO's final status and misses TrialBench's actual enrollment (0.65–0.79) and everything in the ICU records. Permutation importance in the naive model overstates TrialBench's LAP by 46–89% and, for CTO, gives ≤0.054 to every signal except final status; removing final status and refitting still leaves LAP = +0.649 [0.609, 0.685] (86%). In the censored benchmark and the ICU records, post-decision information shares columns with earlier information, so no column holds only post-decision information and nothing can be permuted without also removing earlier information. | `outputs/method_comparison.json` |
| **Clinical usefulness (ICU)** | Decision-curve analysis of the 24-hour mortality model: the deployable model beats treating all or none at thresholds 5–50%; evaluating it with the 48-hour record overstates net benefit by 0.8–1.1 net true positives per 100 patients (9–26%) at thresholds 10–30%. Both models are calibrated (slopes 0.96 and 1.02); LAP is positive in every ICU type (0.10–0.21, exploratory). | `outputs/decision_curve_icu.json` |
| **Variance-inflated intervals** (training variability added) | Refits of both learners on resampled training clusters widen every interval but change no conclusion: CTO [0.708, 0.797]; TrialBench [0.137, 0.279], [0.121, 0.190], [0.094, 0.152]; ICU mortality [0.071, 0.143]; censored benchmark [−0.109, 0.258] (upper limit 0.26 instead of 0.16). | `outputs/variance_inflated_intervals.json` |
| **Availability-time censoring (ICU)** | Reporting latencies (storage minus observation time) from the open MIMIC-IV demo (median 1.3 h for chemistry, 0.4 h for vital signs) applied to the PhysioNet 2012 measurements: only 2.7% of measurements observed by 24 h are stored after it, and the deployable AUPRC changes by 0.0005 on average over 20 latency draws (range −0.005 to 0.008), so observation time is an adequate proxy for availability time in ICU records. | `outputs/lab_latency_sensitivity.json` |
| **Learner sensitivity (clinical)** | ICU mortality LAP: gradient-boosted trees +0.107, logistic regression +0.092 [0.065, 0.119], random forest +0.080 [0.062, 0.101]. Sepsis LAP: +0.089, random forest +0.070 [0.051, 0.100], logistic regression +0.016 [0.009, 0.029] (all exclude zero). | `outputs/learner_sensitivity_clinical.json` |
| **Known-truth comparison** (synthetic, 40 replicates per level) | Against the population LAP of each fitted model, LAP is nearly unbiased (−0.007 to +0.002) with 90–98% coverage; joint permutation importance exceeds the true LAP by 0.02–0.08 in 73–100% of replicates with leakage; a univariate screen never flags a post-decision column; removing the top-ranked column leaves 0–3% of the leakage when it sits in one column and 51–59% when it is spread across redundant columns. | `outputs/comparator_validation.json` |
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
| Revised CTO audit (availability tiers, exclusions) | `python scripts/audit_cto_v2.py` | ~4 min |
| Revised TrialBench audit (post-start features, joint and fixed-test comparisons) | `python scripts/audit_trialbench_v2.py` then `python scripts/audit_trialbench_v2_lap.py` (fetches enrollment type and site data from ClinicalTrials.gov on first run, cached in `data/trialbench/`) | ~11 min + 3 min |
| Learner and split-point sensitivity | `python scripts/robustness.py --out outputs/robustness.json` | ~2 min |
| Reviewer checks (rolling origin, label maturity, fixed-test split, batch bootstrap) | `python scripts/sensitivity_reviewer.py` | ~2 min |
| Language-model validity checks (masking audit, exclusions, curated approval strata; no model queries) | `python scripts/llm_validity_checks.py` | ~2 min |
| Interval comparison (fixed-model vs variance-inflated) | `python scripts/validate_refit_bootstrap.py` | ~45 min (6 workers) |
| Validation matched to the 20-disease benchmark | `python scripts/validate_matched.py` | ~35 min (4 workers) |
| Open-weight language models (exploratory) | `python scripts/llm_open_models.py` (local Ollama: `gemma3:12b`, `phi4:14b`); `--analyze-only` recomputes from `outputs/llm_raw/` | ~45 min per model |
| Patient-level demonstration (ICU records) | `python scripts/audit_physionet2012.py` (downloads the open PhysioNet 2012 challenge files on first run) | ~4 min |
| TOP chronology | `python scripts/audit_top.py` (fetches dates from ClinicalTrials.gov on first run, cached in `data/top/`) | ~2 min |
| Decision-level impact | `python scripts/decision_impact.py` | ~5 min |
| Comparison with common leakage checks | `python scripts/compare_methods.py` | ~1 min |
| Decision-curve analysis (ICU) | `python scripts/decision_curve_icu.py` | ~1 min |
| Sepsis onset (two hospital systems) | `python scripts/audit_physionet2019_sepsis.py` (downloads the open PhysioNet 2019 files on first run) | ~20 min |
| Variance-inflated intervals for the real audits | `python scripts/variance_inflated_intervals.py` | ~30 min |
| Numerical check of the formal properties (Appendix A) | `python scripts/check_propositions.py` | ~2 min |
| Availability-time censoring of the ICU records (MIMIC-IV demo latencies) | `python scripts/audit_lab_latency.py` (downloads the open demo files on first run) | ~5 min |
| Learner sensitivity of the clinical demonstrations | `python scripts/learner_sensitivity_clinical.py` | ~10 min |
| Known-truth comparison of the checks | `OMP_NUM_THREADS=1 python scripts/validate_comparators.py --workers 4` | ~5 min |
| All figures | `python scripts/make_figures.py --out figures` and `python scripts/make_figures_v2.py --out figures` | ~10 s |

**Data provenance.** Open Targets Platform release 26.06 (the connector warns if the live
release differs), ClinicalTrials.gov API v2, NCBI PubMed E-utilities, the PhysioNet/Computing in
Cardiology Challenge 2012 (Open Data Commons Attribution License v1.0); CTO from Hugging Face
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
versions; resolves to the latest). v1.4.0 adds the comparison with common leakage checks, the decision-curve analysis, the sepsis
demonstration, variance-inflated intervals for the real audits, availability-time censoring of the ICU records with MIMIC-IV latencies, clinical learner sensitivity, a known-truth comparison of the checks and a numerical check of the formal properties; v1.3.0 ([10.5281/zenodo.23032036](https://doi.org/10.5281/zenodo.23032036))
added the revised analyses (learner and split sensitivity, ICU records, TOP chronology, decision
impact, open-weight models); v1.2.0 ([10.5281/zenodo.22923231](https://doi.org/10.5281/zenodo.22923231))
produced the earlier version. See `CITATION.cff` (GitHub renders it as a "Cite this repository" button).

## License

MIT — see [LICENSE](LICENSE).
