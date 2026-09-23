"""End-to-end demo on the synthetic testbed (runs in about a minute).

Runs the full pipeline on the sealed prospective test block of the synthetic data
(generate it first with scripts/make_synth.py): baselines with target-clustered
bootstrap CIs, the LAP leakage diagnostic with per-group attribution, the causal
report, and decision-curve / policy evaluation. Writes report.json, summary.txt,
decision_curve.png and reliability.png to outputs/synth_demo/.

Usage:  python scripts/make_synth.py && python scripts/run_synth_benchmark.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from temporal_leakage_audit import causal, decision, features, leakage, metrics as M, models, report, splits  # noqa: E402
from temporal_leakage_audit.config import get_config  # noqa: E402


def load_data(cfg):
    d = cfg["data_dir"]
    return (
        pd.read_csv(os.path.join(d, "programs.csv")),
        pd.read_csv(os.path.join(d, "evidence.csv")),
    )


def ci_for(y, p, clusters, n_boot, seed, budget):
    yy = np.asarray(y)

    def auprc_fn(rows):
        return M.auprc(yy[rows], p[rows])

    def pv_fn(rows):
        return M.policy_value_at_budget(yy[rows], p[rows], budget)

    return {
        "auprc": round(M.auprc(yy, p), 4),
        "auprc_ci95": [round(v, 4) for v in _ci(auprc_fn, yy, clusters, n_boot, seed)],
        "auroc": round(M.auroc(yy, p), 4),
        "brier": round(M.brier(yy, p), 4),
        "ece": round(M.ece(yy, p), 4),
        "policy_value": round(M.policy_value_at_budget(yy, p, budget), 4),
        "policy_value_ci95": [round(v, 4) for v in _ci(pv_fn, yy, clusters, n_boot, seed)],
    }


def _ci(fn, y, clusters, n_boot, seed):
    d = M.clustered_bootstrap_ci(fn, values=y, clusters=clusters, n_boot=n_boot, seed=seed)
    return [d["lo"], d["hi"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = get_config(args.config)
    seed = cfg["seed"]
    budget = cfg["decision"]["budget_frac"]
    n_boot = cfg["bootstrap"]["n_boot"]

    programs, evidence = load_data(cfg)
    X_cens, X_naive, y, meta, groups = features.build(programs, evidence)

    tr, te, split_info = splits.sealed_prospective(
        meta,
        seal_year=cfg["split"]["seal_year"],
        enforce_target_disjoint=cfg["split"]["enforce_target_disjoint"],
    )
    ytr, yte = y.loc[tr].values, y.loc[te].values
    clusters_te = meta.loc[te, "target_id"].values

    # ---- Baselines (all evaluated on the sealed test, censored features) ----
    baselines = {}
    full_clf = models.fit_gbdt(X_cens.loc[tr], ytr, seed=seed)
    p_full = models.predict_gbdt(full_clf, X_cens.loc[te])
    baselines["all_features_censored"] = ci_for(yte, p_full, clusters_te, n_boot, seed, budget)

    for name, cols in [
        ("genetic_only", groups["genetic"]),
        ("literature_only_censored", groups["literature"]),
        ("structural_only", groups["structural"]),
    ]:
        clf = models.fit_gbdt(X_cens.loc[tr], ytr, seed=seed, cols=cols)
        p = models.predict_gbdt(clf, X_cens.loc[te], cols=cols)
        baselines[name] = ci_for(yte, p, clusters_te, n_boot, seed, budget)

    # Leakage reference: literature-only but NAIVE (uncensored) -- should look inflated.
    clf_lit_n = models.fit_gbdt(X_naive.loc[tr], ytr, seed=seed, cols=groups["literature"])
    p_lit_n = models.predict_gbdt(clf_lit_n, X_naive.loc[te], cols=groups["literature"])
    baselines["literature_only_NAIVE"] = ci_for(yte, p_lit_n, clusters_te, n_boot, seed, budget)

    # Heuristic + base rate.
    heur = models.heuristic_genetic(X_cens.loc[te])
    baselines["heuristic_genetic"] = {
        "policy_value": round(M.policy_value_at_budget(yte, heur, budget), 4),
        "auroc": round(M.auroc(yte, heur), 4),
    }
    baselines["base_rate"] = {"policy_value": round(float(np.mean(yte)), 4)}

    # ---- Leakage diagnostic (LAP) ----
    lap = leakage.lap_report(
        X_cens, X_naive, y, tr, te, groups, seed=seed, budget_frac=budget
    )

    # ---- Causal report ----
    caus = causal.causal_report(
        X_cens.loc[tr, groups["confounders"]],
        X_cens.loc[tr, cfg["task"]["treatment_col"]],
        y.loc[tr],
        seed=seed,
    )

    # ---- Decision evaluation on the sealed test ----
    verdict = decision.evaluate_policies(
        yte,
        scores={"model": p_full, "heuristic_genetic": heur},
        clusters=clusters_te,
        budget_frac=budget,
        n_boot=n_boot,
        seed=seed,
    )
    nb = decision.net_benefit_summary(
        yte, p_full, cfg["decision"]["pt_for_netbenefit"], cfg["decision"]["pt_grid"]
    )

    out = {
        "split": split_info,
        "baselines": baselines,
        "leakage_LAP": lap,
        "causal": caus,
        "decision": verdict,
        "net_benefit": nb,
    }
    demo_dir = os.path.join(cfg["output_dir"], "synth_demo")
    report.write_json(out, os.path.join(demo_dir, "report.json"))
    report.plot_decision_curve(nb["decision_curve"], os.path.join(demo_dir, "decision_curve.png"))
    report.plot_reliability(yte, p_full, os.path.join(demo_dir, "reliability.png"))

    # ---- Human-readable summary ----
    L = []
    L.append("=" * 70)
    L.append("SYNTHETIC DEMO SUMMARY  (sealed prospective test)")
    L.append("=" * 70)
    L.append(f"train n={split_info['n_train']}  test n={split_info['n_test']}  "
             f"seal_year={split_info['seal_year']}  "
             f"dropped_shared_target={split_info['test_dropped_shared_target']}")
    L.append("")
    L.append("BASELINES (AUPRC [CI], AUROC, Brier, policy@budget):")
    for name, m in baselines.items():
        ap_ = m.get("auprc"); ci = m.get("auprc_ci95"); au = m.get("auroc")
        br = m.get("brier"); pvv = m.get("policy_value")
        row = f"  {name:28s} "
        if ap_ is not None:
            row += f"AUPRC={ap_:.3f} {ci}  "
        if au is not None:
            row += f"AUROC={au:.3f}  "
        if br is not None:
            row += f"Brier={br:.3f}  "
        if pvv is not None:
            row += f"policy={pvv:.3f}"
        L.append(row)
    L.append("")
    L.append("LEAKAGE (LAP = naive - censored):")
    L.append(f"  naive    AUPRC={lap['naive']['auprc']:.3f}  AUROC={lap['naive']['auroc']:.3f}")
    L.append(f"  censored AUPRC={lap['censored']['auprc']:.3f}  AUROC={lap['censored']['auroc']:.3f}")
    L.append(f"  LAP.auprc={lap['LAP']['auprc']:+.3f}  LAP.auroc={lap['LAP']['auroc']:+.3f}  "
             f"LAP.policy={lap['LAP']['policy_value']:+.3f}")
    L.append(f"  AUPRC drop-column importance by group (feature importance, not leakage "
             f"attribution): {lap['auprc_group_dropcol_importance']}")
    L.append("")
    L.append("CAUSAL:")
    L.append(f"  naive assoc (confounded) = {caus['naive_association_diff']:+.4f}")
    L.append(f"  AIPW ATE (adjusted)      = {caus['aipw_ate']['ate']:+.4f}  CI95 {caus['aipw_ate']['ci95']}")
    L.append(f"  refutations: placebo={caus['refutations']['placebo_treatment_ate']:+.4f} (want ~0)  "
             f"subset={caus['refutations']['subset_70pct_ate']:+.4f}  "
             f"rcc={caus['refutations']['random_common_cause_ate']:+.4f}")
    L.append("")
    L.append("DECISION:")
    for k, v in verdict["policy_values"].items():
        L.append(f"  policy[{k}]={v:.3f}")
    L.append(f"  gap(best model - best trivial)={verdict['gap_best_model_minus_best_trivial']:+.4f} "
             f"CI95 {verdict['gap_ci95']}")
    L.append(f"  net benefit @pt={nb['pt']}: model={nb['net_benefit_model']:+.4f} "
             f"vs advance-all={nb['net_benefit_treat_all']:+.4f}")
    L.append(f"  VERDICT: {verdict['VERDICT']}")
    L.append("=" * 70)
    summary = "\n".join(L)
    report.write_text(summary, os.path.join(demo_dir, "summary.txt"))
    print(summary)


if __name__ == "__main__":
    main()
