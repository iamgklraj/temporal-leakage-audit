"""Evaluation metrics.

Discrimination (AUPRC/AUROC), calibration (Brier/ECE), and -- crucially for
this project -- decision metrics (net benefit, policy value) that ask whether
acting on the model beats acting on a trivial rule. All point estimates come
with target-clustered bootstrap confidence intervals, because pairs sharing a
gene are not independent.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        total += (m.sum() / n) * abs(acc - conf)
    return float(total)


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    """Decision-curve net benefit at threshold probability ``pt``."""
    n = len(y)
    pred = p >= pt
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    w = pt / (1 - pt) if pt < 1 else np.inf
    return float(tp / n - fp / n * w)


def net_benefit_treat_all(y: np.ndarray, pt: float) -> float:
    n = len(y)
    tp = int(np.sum(y == 1))
    fp = int(np.sum(y == 0))
    w = pt / (1 - pt) if pt < 1 else np.inf
    return float(tp / n - fp / n * w)


def decision_curve(y: np.ndarray, p: np.ndarray, pt_grid: Sequence[float]) -> Dict[str, list]:
    return {
        "pt": list(map(float, pt_grid)),
        "model": [net_benefit(y, p, pt) for pt in pt_grid],
        "treat_all": [net_benefit_treat_all(y, pt) for pt in pt_grid],
        "treat_none": [0.0 for _ in pt_grid],
    }


def policy_value_at_budget(y: np.ndarray, score: np.ndarray, budget_frac: float) -> float:
    """Mean label among the top ``budget_frac`` programs ranked by ``score``.

    This is the operational question: if you can advance this fraction of
    programs, how good are the ones this ranking picks?
    """
    n = len(y)
    k = max(1, int(round(budget_frac * n)))
    order = np.argsort(-score, kind="stable")
    top = order[:k]
    return float(np.mean(y[top]))


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    order = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y[order]))


def clustered_bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap a metric by resampling whole clusters (targets) with replacement.

    ``metric_fn`` receives the row indices of a bootstrap sample and returns a
    scalar. ``values`` is unused directly but documents intent; indices index
    into whatever arrays ``metric_fn`` closes over.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    # Map cluster -> row indices once.
    cl_to_rows = {c: np.where(clusters == c)[0] for c in uniq}
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([cl_to_rows[c] for c in chosen])
        try:
            stats.append(metric_fn(rows))
        except Exception:
            continue
    stats = np.array([s for s in stats if np.isfinite(s)])
    if len(stats) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "mean": float(np.mean(stats)),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
    }
