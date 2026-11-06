"""Day 7–9 — metrics, cross-validation and diagnostic plots."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

from src.plot_style import PRIMARY

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R² and MAPE in rupees."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
    }


def cv_summary(estimator, X, y, cv, groups=None, name: str | None = None) -> dict:
    """Cross-validate and return mean train/validation metrics plus fit time."""
    res = cross_validate(clone(estimator), X, y, cv=cv, groups=groups, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    return {
        "model": name,
        "train_MAE": -res["train_mae"].mean(),
        "cv_MAE": -res["test_mae"].mean(),
        "cv_MAE_std": res["test_mae"].std(),
        "train_RMSE": -res["train_rmse"].mean(),
        "cv_RMSE": -res["test_rmse"].mean(),
        "train_R2": res["train_r2"].mean(),
        "cv_R2": res["test_r2"].mean(),
        "cv_R2_std": res["test_r2"].std(),
        "fit_time_s": res["fit_time"].mean(),
    }


def compare_models(models: dict, X, y, cv, groups=None) -> pd.DataFrame:
    rows = [cv_summary(est, X, y, cv, groups, name) for name, est in models.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("cv_MAE")


def holdout_evaluation(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Fit each model on the training set and score it on the untouched test set."""
    rows, fitted = [], {}
    for name, est in models.items():
        model = clone(est)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        train_m = regression_metrics(y_train, model.predict(X_train))
        test_m = regression_metrics(y_test, model.predict(X_test))
        rows.append({"model": name, "train_MAE": train_m["MAE"], "train_R2": train_m["R2"],
                     "test_MAE": test_m["MAE"], "test_RMSE": test_m["RMSE"], "test_R2": test_m["R2"],
                     "test_MAPE_%": test_m["MAPE_%"], "fit_time_s": fit_s})
        fitted[name] = model
    return pd.DataFrame(rows).set_index("model").sort_values("test_MAE"), fitted


# ----------------------------------------------------------------------------- plots
def _rupees(ax, axis="both"):
    fmt = plt.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def plot_actual_vs_predicted(y_true, y_pred, title: str, ax=None):
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    ax.scatter(y_true, y_pred, alpha=0.6, s=28, color=PRIMARY, edgecolor="white", linewidth=0.4)
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, ls="--", color="#555", lw=1, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price (₹)"); ax.set_ylabel("Predicted price (₹)")
    ax.set_title(title); ax.legend(loc="upper left"); _rupees(ax)
    return ax


def plot_residuals(y_true, y_pred, title: str, axes=None):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, resid, alpha=0.6, s=26, color=PRIMARY, edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color="#555", ls="--", lw=1)
    axes[0].set_xlabel("Predicted price (₹)"); axes[0].set_ylabel("Residual: actual − predicted (₹)")
    axes[0].set_title(f"{title}: residuals vs predicted"); _rupees(axes[0])
    axes[1].hist(resid, bins=30, color=PRIMARY, edgecolor="white")
    axes[1].axvline(0, color="#555", ls="--", lw=1)
    axes[1].set_xlabel("Residual (₹)"); axes[1].set_ylabel("Number of phones")
    axes[1].set_title(f"{title}: residual distribution"); _rupees(axes[1], "x")
    return axes
