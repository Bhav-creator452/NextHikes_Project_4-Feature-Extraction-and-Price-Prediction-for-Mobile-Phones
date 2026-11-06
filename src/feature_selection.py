"""Day 5 — feature relevance helpers (correlation, mutual information, VIF)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


def correlation_with_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Pearson (linear) and Spearman (monotonic, rank-based) correlation of each column with y."""
    rows = []
    for col in X.columns:
        rows.append({
            "feature": col,
            "pearson": X[col].corr(y, method="pearson"),
            "spearman": X[col].corr(y, method="spearman"),
        })
    out = pd.DataFrame(rows).set_index("feature")
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values("abs_spearman", ascending=False)


def mutual_information(X: pd.DataFrame, y: pd.Series, discrete_mask=None,
                       random_state: int = 42, n_repeats: int = 5) -> pd.Series:
    """Mutual information (nats) averaged over several seeds — the kNN estimator is noisy."""
    scores = np.zeros(X.shape[1])
    for seed in range(random_state, random_state + n_repeats):
        scores += mutual_info_regression(X, y, discrete_features=discrete_mask if discrete_mask is not None else "auto",
                                         random_state=seed)
    return pd.Series(scores / n_repeats, index=X.columns).sort_values(ascending=False)


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R²_j), where R²_j regresses feature j on all other features."""
    vif = {}
    values = X.astype(float)
    for col in values.columns:
        others = values.drop(columns=col)
        r2 = LinearRegression().fit(others, values[col]).score(others, values[col])
        vif[col] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
    return pd.Series(vif).sort_values(ascending=False)
