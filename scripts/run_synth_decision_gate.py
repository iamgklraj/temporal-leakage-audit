"""Decision gate on the synthetic testbed (fast).

Once leakage is controlled (as-of-time features) and confounding is adjusted,
does acting on the model beat acting on a trivial rule? Prints a GO / LEAN GO /
NO-GO verdict and writes outputs/synth_demo/decision_gate.json.

Usage:  python scripts/run_synth_decision_gate.py [--config config/default.yaml]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from temporal_leakage_audit import causal, decision, features, models, report, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402


def load_data(cfg):
    d = cfg["data_dir"]
    programs = pd.read_csv(os.path.join(d, "programs.csv"))
    evidence = pd.read_csv(os.path.join(d, "evidence.csv"))
    return programs, evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = get_config(args.config)
    seed = cfg["seed"]

    programs, evidence = load_data(cfg)
    X_cens, X_naive, y, meta, groups = features.build(programs, evidence)

    tr, te, split_info = splits.temporal_split(
        meta,
        train_max_year=cfg["split"]["train_max_year"],
        test_years=cfg["split"]["test_years"],
        enforce_target_disjoint=cfg["split"]["enforce_target_disjoint"],
    )

    # ---- Causal effect of the treatment, adjusting for confounders (as-of-t) ----
    conf_cols = groups["confounders"]
    treatment = cfg["task"]["treatment_col"]
    caus = causal.causal_report(
        X_cens.loc[tr, conf_cols], X_cens.loc[tr, treatment], y.loc[tr], seed=seed
    )

    # ---- Model scores on the test block (censored features only) ----
    clf = models.fit_gbdt(X_cens.loc[tr], y.loc[tr].values, seed=seed)
    p_model = models.predict_gbdt(clf, X_cens.loc[te])
    heur = models.heuristic_genetic(X_cens.loc[te])

    verdict = decision.evaluate_policies(
        y.loc[te].values,
        scores={"model": p_model, "heuristic_genetic": heur},
        clusters=meta.loc[te, "target_id"].values,
        budget_frac=cfg["decision"]["budget_frac"],
        n_boot=cfg["bootstrap"]["n_boot"],
        seed=seed,
    )

    out = {
        "split": split_info,
        "causal": caus,
        "decision": verdict,
    }
    report.write_json(out, os.path.join(cfg["output_dir"], "synth_demo", "decision_gate.json"))

    # ---- Console summary ----
    print("=" * 68)
    print("DECISION GATE (synthetic testbed)")
    print("=" * 68)
    print(f"train n={split_info['n_train']}  test n={split_info['n_test']}  "
          f"(dropped {split_info['test_dropped_shared_target']} test rows sharing a train target)")
    print()
    print(f"Treatment: {caus['treatment']}   treated={caus['n_treated']} control={caus['n_control']}")
    print(f"  naive association (confounded) : {caus['naive_association_diff']:+.4f}")
    print(f"  AIPW causal ATE (adjusted)     : {caus['aipw_ate']['ate']:+.4f}  "
          f"CI95 {caus['aipw_ate']['ci95']}")
    print(f"  refutations: placebo={caus['refutations']['placebo_treatment_ate']:+.4f} (want ~0)  "
          f"subset={caus['refutations']['subset_70pct_ate']:+.4f}  "
          f"rcc={caus['refutations']['random_common_cause_ate']:+.4f}")
    print()
    print(f"Policy value @ budget={verdict['budget_frac']:.0%}  (base rate {verdict['base_rate']:.3f})")
    for k, v in verdict["policy_values"].items():
        print(f"  {k:28s}: {v:.4f}")
    print(f"\n  gap (best model - best trivial): {verdict['gap_best_model_minus_best_trivial']:+.4f}  "
          f"CI95 {verdict['gap_ci95']}")
    print(f"\n>>> VERDICT: {verdict['VERDICT']}")
    print(f"    {verdict['guidance']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
