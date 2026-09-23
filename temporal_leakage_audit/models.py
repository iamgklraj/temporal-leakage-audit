"""Predictive baselines.

Uses sklearn's HistGradientBoostingClassifier so there is no hard dependency on
xgboost/lightgbm (both are fine drop-ins if you prefer -- see fit_gbdt). Also
includes the trivial rules the ML models must beat to be worth anything:
"advance if genetically supported" and the base rate.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def fit_gbdt(
    X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 0, cols: Optional[List[str]] = None
) -> HistGradientBoostingClassifier:
    if cols is not None:
        X_train = X_train[cols]
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=300,
        l2_regularization=1.0, random_state=seed,
    )
    clf.fit(X_train.values, y_train)
    return clf


def predict_gbdt(
    clf: HistGradientBoostingClassifier, X: pd.DataFrame, cols: Optional[List[str]] = None
) -> np.ndarray:
    if cols is not None:
        X = X[cols]
    return clf.predict_proba(X.values)[:, 1]


def heuristic_genetic(X: pd.DataFrame) -> np.ndarray:
    """Score = has_genetic_support (+ tiny genetic count tiebreaker)."""
    base = X["has_genetic_support"].values.astype(float)
    tie = X.get("ev_genetic_count", pd.Series(0.0, index=X.index)).values.astype(float)
    return base + 1e-3 * tie


def base_rate_scores(y_train: np.ndarray, n: int) -> np.ndarray:
    """Constant score = training base rate (a ranking-useless reference)."""
    return np.full(n, float(np.mean(y_train)))
