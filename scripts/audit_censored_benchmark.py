"""Leakage, causal and decision audit of the as-of-time-censored drug-program benchmark.

Runs, on one dataset:
  * the temporal leakage-response curve (deployable vs. naive, peak LAP, half-life),
  * a per-source leakage decomposition (temporal placebo),
  * rolling-origin LAP across several cutoffs (replication),
  * the positivity-aware causal layer (AIPW ATE, overlap-weighted ATO, balance,
    permutation test, E-values; as-of-time features, training block only),
  * the decision gate (policy value at a budget vs. trivial rules).

Splits are derived from the dataset's info_time distribution (train <= 60th
percentile; target-disjoint later test block). All CIs are target-clustered bootstraps.

Usage:
    python scripts/audit_censored_benchmark.py --config config/benchmark_20disease.yaml \
        --out outputs/censored_benchmark_20disease.json
    python scripts/audit_censored_benchmark.py --config config/benchmark_41disease.yaml \
        --out outputs/censored_benchmark_41disease.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from temporal_leakage_audit import causal, decision, features, leakage, models, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/benchmark_20disease.yaml",
                    help="YAML config (sets data_dir, seed, bootstrap size)")
    ap.add_argument("--data-dir", default=None, help="override the config's data_dir")
    ap.add_argument("--out", default="outputs/censored_benchmark_20disease.json",
                    help="output JSON path")
    args = ap.parse_args()
    cfg = get_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    # Data are built from public APIs on demand, not committed. If the snapshot is
    # absent, say how to build it rather than crashing on a missing file.
    for f in ("programs.csv", "evidence.csv"):
        if not os.path.exists(os.path.join(data_dir, f)):
            raise SystemExit(
                f"{data_dir}/{f} not found. The dataset is built from public APIs and "
                f"is not stored in the repository. Build it first:\n"
                f"    python -m temporal_leakage_audit.data.connectors {args.config}")
    programs = pd.read_csv(os.path.join(data_dir, "programs.csv"))
    evidence = pd.read_csv(os.path.join(data_dir, "evidence.csv"))
    Xc, Xn, y, meta, groups = features.build(programs, evidence)
    yrs = programs["info_time"]

    print(f"dataset: {programs.shape} | label {programs['label'].value_counts().to_dict()} | "
          f"diseases {programs['indication'].nunique()} | targets {programs['target_id'].nunique()}")
    print(f"info_time: min {int(yrs.min())} median {int(yrs.median())} max {int(yrs.max())}")

    # Data-driven main split: train <= 60th pct of info_time, target-disjoint future test.
    tmax = int(yrs.quantile(0.60))
    hi = int(yrs.max())
    tr, te, info = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    print(f"main split: train<= {tmax}  test {tmax+1}-{hi}  n_train={info['n_train']} "
          f"n_test={info['n_test']} (dropped {info['test_dropped_shared_target']} shared-target)")
    if len(te) == 0 or y.loc[te].nunique() < 2:
        raise SystemExit(f"main split unusable (n_test={len(te)}, "
                         f"classes={sorted(set(y.loc[te]))}); widen the split window.")

    te_clusters = meta.loc[te, "target_id"].values
    lrc = leakage.leakage_response_curve(
        programs, evidence, tr, te, clusters=te_clusters,
        n_boot=cfg["bootstrap"]["n_boot"], budget_frac=cfg["decision"]["budget_frac"])
    psl = leakage.per_source_placebo_lap(Xc, Xn, y, tr, te, groups, clusters=te_clusters,
                                         n_boot=cfg["bootstrap"]["n_boot"])

    # Rolling-origin LAP across cutoffs for replication/power (each with a CI).
    cutoffs = sorted({int(yrs.quantile(x)) for x in (0.45, 0.55, 0.65, 0.75)})
    roll = []
    for c in cutoffs:
        tri, tei, i2 = splits.temporal_split(meta, c, (c + 1, hi), enforce_target_disjoint=True)
        if len(tei) < 20 or int(y.loc[tei].sum()) < 3:
            continue
        lap = leakage.lap_report(Xc, Xn, y, tri, tei, groups,
                                 clusters=meta.loc[tei, "target_id"].values,
                                 n_boot=cfg["bootstrap"]["n_boot"])
        roll.append({"cutoff": c, "n_test": i2["n_test"], "test_pos": int(y.loc[tei].sum()),
                     "test_base_rate": lap["test_base_rate"],
                     "naive_auprc": lap["naive"]["auprc"], "censored_auprc": lap["censored"]["auprc"],
                     "LAP_auprc": lap["LAP"]["auprc"], "LAP_ci": lap.get("LAP_auprc_ci"),
                     "LAP_p_gt_0": lap.get("LAP_auprc_p_gt_0")})

    conf, treat = groups["confounders"], cfg["task"]["treatment_col"]
    caus = causal.causal_report(Xc.loc[tr, conf], Xc.loc[tr, treat], y.loc[tr], seed=cfg["seed"],
                                clusters=meta.loc[tr, "target_id"].values, n_boot=300)

    clf = models.fit_gbdt(Xc.loc[tr], y.loc[tr].values, seed=cfg["seed"])
    p_model = models.predict_gbdt(clf, Xc.loc[te])
    heur = models.heuristic_genetic(Xc.loc[te])
    dec = decision.evaluate_policies(
        y.loc[te].values, {"model": p_model, "heuristic_genetic": heur},
        meta.loc[te, "target_id"].values, budget_frac=cfg["decision"]["budget_frac"],
        n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["seed"],
    )

    out = {
        "config": args.config,
        "data_dir": data_dir,
        "dataset": {"n": len(programs), "label": programs["label"].value_counts().to_dict(),
                    "diseases": int(programs["indication"].nunique()),
                    "targets": int(programs["target_id"].nunique())},
        "main_split": info,
        "leakage_response_curve": lrc,
        "per_source_placebo_lap": psl,
        "rolling_origin_LAP": roll,
        "causal": caus,
        "decision": dec,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("\n=== LEAKAGE-RESPONSE CURVE (target-clustered bootstrap CIs) ===")
    print(f"  test_base_rate={lrc['test_base_rate']:.3f} (AUPRC read against this)")
    for c in lrc["curve"]:
        h = "naive" if c["horizon"] is None else f"h={c['horizon']}y"
        ci = c.get("auprc_ci")
        cistr = f" 95%CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
        print(f"  {h:9s} AUPRC={c['auprc_mean']:.3f}{cistr}")
    print(f"  deployable={lrc['deployable_auprc']:.3f} (lift over base {lrc['deployable_lift_over_base']:+.3f})  "
          f"naive={lrc['naive_auprc']:.3f}")
    print(f"  LAP={lrc['total_LAP_auprc']:+.3f}  95%CI={lrc['total_LAP_auprc_ci']}  "
          f"P(LAP>0)={lrc['total_LAP_auprc_p_gt_0']}  half_life={lrc['leakage_half_life_years']}y")
    print(f"  peak_LAP={lrc['peak_LAP_auprc']:+.3f} at h={lrc['peak_horizon']} "
          f"(non-monotone curve; naive-minus-deployable understates the ceiling)")

    print("\n=== PER-SOURCE LEAKAGE (placebo: admit one source at full hindsight) ===")
    for g, v in psl["per_source"].items():
        ci = v.get("ci")
        print(f"  {g:16s} LAP={v['LAP']:+.3f}" + (f" CI[{ci[0]:+.3f},{ci[1]:+.3f}] P(>0)={v['p_gt_0']}" if ci else ""))

    print("\n=== ROLLING-ORIGIN LAP (with clustered CIs) ===")
    for r in roll:
        print(f"  cutoff {r['cutoff']}: n_test={r['n_test']} (+{r['test_pos']}, base {r['test_base_rate']:.2f})  "
              f"LAP={r['LAP_auprc']:+.3f} CI{r['LAP_ci']} P(>0)={r['LAP_p_gt_0']}  "
              f"(naive {r['naive_auprc']:.3f} vs censored {r['censored_auprc']:.3f})")

    print("\n=== CAUSAL (as-of-t, train; ATO = primary robust estimand) ===")
    a = caus["aipw_ate"]
    print(f"  naive assoc {caus['naive_association_diff']:+.3f}")
    print(f"  ATE      {a['ate']:+.3f}  IF-CI{a['ci95']}  clustered-CI{a.get('ate_ci_clustered')}")
    print(f"  n_treated={caus['n_treated']} n_control={caus['n_control']}  ATO_ESS={a.get('ato_ess')}")
    print(f"  ATO      {a.get('ato')}  clustered-CI{a.get('ato_ci_clustered')}  (positivity-robust)")
    print(f"  trimmed  {a.get('trimmed_ate')} (n={a.get('n_trimmed')}, e in {a.get('trim_range')})  "
          f"clustered-CI{a.get('trimmed_ate_ci_clustered')}")
    print(f"  raw propensity overlap {a['propensity_overlap']}  E-value(limit)={caus['sensitivity']['e_value_lower_ci']}")
    b = caus.get("covariate_balance", {})
    print(f"  balance: max|SMD| {b.get('max_abs_smd_unweighted')} -> {b.get('max_abs_smd_overlap_weighted')} "
          f"(overlap-weighted); {b.get('n_balanced_after_weighting')}/{b.get('n_covariates')} covariates |SMD|<0.1 after")
    ncd = caus.get("negative_control", {})
    print(f"  outcome-permutation test: real ATO {ncd.get('real_ato')} vs null "
          f"{ncd.get('null_mean')} CI{ncd.get('null_ci95')}  perm-p={ncd.get('permutation_p_value')}")

    print("\n=== DECISION ===")
    print(f"  gap {dec['gap_best_model_minus_best_trivial']:+.3f} CI{dec['gap_ci95']}  "
          f"VERDICT: {dec['VERDICT']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
