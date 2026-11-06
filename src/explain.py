"""Day 10 — model explanations without extra dependencies.

`exact_shapley_values` computes interventional Shapley values over the *model input columns*
(model name, processor, storage, ...). With 9 inputs there are only 2^9 = 512 coalitions, so the
values are computed exactly rather than approximated. This is the same quantity SHAP's exact /
permutation explainers estimate; if the `shap` package is installed the notebook also uses it.

Explaining at the input level is deliberate: one-hot columns of `brand` are combined into a
single contribution, which is what a pricing analyst actually wants to read.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


def exact_shapley_values(model, x_row: pd.DataFrame, background: pd.DataFrame,
                         log_space: bool = False) -> pd.Series:
    """Shapley contribution of each input column to one prediction.

    Parameters
    ----------
    model : fitted pipeline with .predict on raw input columns.
    x_row : single-row DataFrame to explain.
    background : reference rows; "absent" features take their values from these rows.
    log_space : explain log(price) instead of price (additive in % terms).

    Returns
    -------
    pd.Series of contributions; sum(contributions) + base_value == prediction.
    The base value is stored in `.attrs['base_value']`.
    """
    features = list(x_row.columns)
    n = len(features)
    x = x_row.iloc[0]
    bg = background[features].reset_index(drop=True)

    # Evaluate all 2^n coalitions in one batched predict call.
    subsets = [s for k in range(n + 1) for s in combinations(features, k)]
    frames = []
    for s in subsets:
        data = bg.copy()
        for f in s:
            data[f] = x[f]
        frames.append(data)
    preds = np.asarray(model.predict(pd.concat(frames, ignore_index=True)), dtype=float)
    if log_space:
        preds = np.log(preds)
    means = preds.reshape(len(subsets), len(bg)).mean(axis=1)
    cache = {tuple(sorted(s)): m for s, m in zip(subsets, means)}

    def v(subset):
        return cache[tuple(sorted(subset))]

    phi = dict.fromkeys(features, 0.0)
    for i, f in enumerate(features):
        others = features[:i] + features[i + 1:]
        for k in range(n):
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for s in combinations(others, k):
                phi[f] += weight * (v(s + (f,)) - v(s))
    out = pd.Series(phi)
    out.attrs["base_value"] = v(())
    out.attrs["prediction"] = v(tuple(features))
    return out


def shapley_matrix(model, X: pd.DataFrame, background: pd.DataFrame, log_space: bool = True) -> pd.DataFrame:
    """Shapley values for several rows (one row per explained phone)."""
    rows = [exact_shapley_values(model, X.iloc[[i]], background, log_space) for i in range(len(X))]
    return pd.DataFrame(rows, index=X.index)
