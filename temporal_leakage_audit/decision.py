"""Decision-aware evaluation and the go/no-go gate.

The operational question is not "what is the AUPRC" but "if we advance our
budget of programs using this model, do we do better than advancing them with a
trivial rule?" This module computes policy value at a budget and net benefit,
then compares the best model-based policy against the best trivial policy with a
target-clustered bootstrap CI on the *gap*. That gap, not accuracy, is the
go/no-go signal.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import metrics as M


def evaluate_policies(
    y: np.ndarray,
    scores: Dict[str, np.ndarray],
    clusters: np.ndarray,
    budget_frac: float = 0.2,
    n_boot: int = 500,
    seed: int = 0,
) -> Dict:
    """Compare model-based vs trivial policies and emit a go/no-go verdict.

    ``scores`` must include at least: 'model' (the censored model's scores),
    optionally 'cate_informed', and the trivial 'heuristic_genetic'. A
    treat-all / base-rate reference is derived from ``y`` directly.
    """
    y = np.asarray(y)
    base_rate = float(np.mean(y))

    def pv(rows, s):
        return M.policy_value_at_budget(y[rows], s[rows], budget_frac)

    all_rows = np.arange(len(y))

    model_candidates = {k: scores[k] for k in ("model", "cate_informed") if k in scores}
    trivial_candidates = {k: scores[k] for k in ("heuristic_genetic",) if k in scores}

    pv_points = {k: pv(all_rows, s) for k, s in {**model_candidates, **trivial_candidates}.items()}
    pv_points["treat_all_base_rate"] = base_rate

    # Choose the model policy and the trivial reference by point estimate.
    best_model_name = max(model_candidates, key=lambda k: pv(all_rows, model_candidates[k]))
    best_model_score = model_candidates[best_model_name]

    def trivial_pv(rows):
        vals = [pv(rows, s) for s in trivial_candidates.values()] + [float(np.mean(y[rows]))]
        return max(vals)

    def gap(rows):
        return pv(rows, best_model_score) - trivial_pv(rows)

    gap_point = gap(all_rows)
    ci = M.clustered_bootstrap_ci(gap, values=y, clusters=clusters, n_boot=n_boot, seed=seed)

    if ci["lo"] > 0:
        verdict = "GO"
        guidance = (
            "Model-based selection beats every trivial policy with a clustered "
            "bootstrap CI excluding 0."
        )
    elif gap_point > 0:
        verdict = "LEAN GO / UNDERPOWERED"
        guidance = (
            "The point estimate favours the model but the CI includes 0; more data "
            "or a wider evaluation window is needed before acting on the model."
        )
    else:
        verdict = "NO-GO"
        guidance = (
            "Model-based selection does not beat trivial policies at this budget; "
            "the model should not drive advancement decisions."
        )

    return {
        "budget_frac": budget_frac,
        "base_rate": round(base_rate, 4),
        "policy_values": {k: round(v, 4) for k, v in pv_points.items()},
        "chosen_model_policy": best_model_name,
        "gap_best_model_minus_best_trivial": round(gap_point, 4),
        "gap_ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
        "VERDICT": verdict,
        "guidance": guidance,
    }


def net_benefit_summary(y: np.ndarray, p_model: np.ndarray, pt: float, pt_grid) -> Dict:
    return {
        "pt": pt,
        "net_benefit_model": round(M.net_benefit(y, p_model, pt), 4),
        "net_benefit_treat_all": round(M.net_benefit_treat_all(y, pt), 4),
        "decision_curve": {
            k: [round(x, 4) for x in v] for k, v in M.decision_curve(y, p_model, pt_grid).items()
        },
    }
