"""Model definitions, data splitting and the end-to-end training script.

Run from the project root:

    python -m src.train

This reproduces the final model: load raw data -> clean -> grouped train/test split ->
fit the selected pipeline on the training set -> evaluate on the held-out test set ->
refit on all clean data -> save models/final_model.joblib and models/model_metadata.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import PROJECT_ROOT, load_raw_data
from src.evaluate import regression_metrics
from src.feature_engineering import extract_brand, make_preprocessor
from src.preprocessing import GROUP_KEY, MODEL_INPUT_COLUMNS, TARGET, clean_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Final model configuration, chosen in notebooks 08 (tuning) and 09 (evaluation):
# an equal-weight average of a Ridge regression and a tuned Gradient Boosting model,
# both learning log(price). Tuned values are rounded from the RandomizedSearchCV result.
FINAL_MODEL_NAME = "Ridge + Gradient Boosting (equal-weight average, log target)"
RIDGE_PARAMS = {"alpha": 0.01}
GB_PARAMS = {
    "n_estimators": 127,
    "learning_rate": 0.16,
    "max_depth": 3,
    "min_samples_leaf": 9,
    "subsample": 1.0,
}
HGB_PARAMS = {"max_iter": 388, "learning_rate": 0.084, "max_depth": 4,
              "min_samples_leaf": 9, "l2_regularization": 0.0062}
RF_PARAMS = {"n_estimators": 300, "max_depth": 8, "max_features": 0.3, "min_samples_leaf": 2}
FINAL_MODEL_PARAMS = {"ridge": RIDGE_PARAMS, "gradient_boosting": GB_PARAMS, "weights": [0.5, 0.5]}


# ----------------------------------------------------------------------------- data
def grouped_train_test_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                             random_state: int = RANDOM_STATE):
    """Split so that every variant of a phone model lands on the same side.

    A random split would put e.g. the blue and black versions of the same phone in train and
    test, letting the model memorise phones instead of learning how specifications drive price.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[GROUP_KEY]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def load_model_ready_data():
    """Clean data with its saved split column (created in notebook 04)."""
    path = PROCESSED_DIR / "phones_model_ready.csv"
    df = pd.read_csv(path)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    return train, test


def xy(df: pd.DataFrame):
    return df[MODEL_INPUT_COLUMNS], df[TARGET], df[GROUP_KEY]


def group_cv(n_splits: int = N_SPLITS) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


# ----------------------------------------------------------------------------- models
def build_pipeline(regressor, scale_numeric: bool = False, log_target: bool = True, **prep_kwargs) -> Pipeline:
    """Preprocessing + regressor. With log_target, the model learns log(price) and predicts
    in rupees (exp is applied automatically), which suits the right-skewed price."""
    model = regressor
    if log_target:
        model = TransformedTargetRegressor(regressor=regressor, func=np.log1p,
                                           inverse_func=np.expm1, check_inverse=False)
    return Pipeline([("prep", make_preprocessor(scale_numeric=scale_numeric, **prep_kwargs)),
                     ("model", model)])


def baseline_models(log_target: bool = True) -> dict:
    return {
        "Dummy (median)": build_pipeline(DummyRegressor(strategy="median"), log_target=False),
        "Linear Regression": build_pipeline(LinearRegression(), scale_numeric=True, log_target=log_target),
        "Decision Tree": build_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE), log_target=log_target),
        "Random Forest": build_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                                               n_jobs=-1), log_target=log_target),
    }


def advanced_models() -> dict:
    return {
        "Ridge": build_pipeline(Ridge(alpha=1.0), scale_numeric=True),
        "Random Forest": build_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "Gradient Boosting": build_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        "HistGradientBoosting": build_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


def tuned_models() -> dict:
    """Tuned candidates (parameters from notebook 08)."""
    return {
        "Ridge": build_pipeline(Ridge(**RIDGE_PARAMS), scale_numeric=True),
        "Random Forest (tuned)": build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **RF_PARAMS)),
        "Gradient Boosting (tuned)": build_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE, **GB_PARAMS)),
        "HistGradientBoosting (tuned)": build_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **HGB_PARAMS)),
        "Ridge + GB average": final_pipeline(),
    }


def final_pipeline() -> TransformedTargetRegressor:
    """Equal-weight average of Ridge (extrapolates, smooth) and Gradient Boosting (non-linear,
    interactions). Each member has its own preprocessing (Ridge needs scaling, trees do not).
    The average is taken in log-price space, then converted back to rupees."""
    ridge = Pipeline([("prep", make_preprocessor(scale_numeric=True)), ("ridge", Ridge(**RIDGE_PARAMS))])
    gb = Pipeline([("prep", make_preprocessor()),
                   ("gb", GradientBoostingRegressor(random_state=RANDOM_STATE, **GB_PARAMS))])
    return TransformedTargetRegressor(
        regressor=VotingRegressor([("ridge", ridge), ("gb", gb)], weights=[0.5, 0.5]),
        func=np.log1p, inverse_func=np.expm1, check_inverse=False)


# ----------------------------------------------------------------------------- script
def main() -> dict:
    raw = load_raw_data()
    clean, _, _ = clean_dataset(raw)
    train, test = grouped_train_test_split(clean)
    X_tr, y_tr, _ = xy(train)
    X_te, y_te, _ = xy(test)

    model = final_pipeline().fit(X_tr, y_tr)
    test_metrics = regression_metrics(y_te, model.predict(X_te))
    train_metrics = regression_metrics(y_tr, model.predict(X_tr))

    # Deployment model: same pipeline and hyperparameters, refit on all clean rows.
    X_all, y_all, _ = xy(clean)
    deploy = final_pipeline().fit(X_all, y_all)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(deploy, MODELS_DIR / "final_model.joblib")
    metadata = {
        "model": FINAL_MODEL_NAME,
        "params": FINAL_MODEL_PARAMS,
        "target": "price (INR), modelled as log1p(price)",
        "input_columns": MODEL_INPUT_COLUMNS,
        "scope": "smartphones (feature phones excluded)",
        "rows_train": len(train), "rows_test": len(test), "rows_deploy_fit": len(clean),
        "validation": "GroupShuffleSplit by model (test = unseen phone models)",
        "holdout_test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
        "training_ranges": {c: [float(clean[c].min()), float(clean[c].max())]
                            for c in ["storage_gb", "ram_gb", "battery_mah", "rear_camera_mp",
                                      "front_camera_mp", "display_size_cm"]},
        "known_processors": sorted(clean["processor"].unique().tolist()),
        "brands_seen": sorted(clean["model"].map(extract_brand).unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "random_state": RANDOM_STATE,
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Hold-out test: MAE ₹{test_metrics['MAE']:,.0f} | RMSE ₹{test_metrics['RMSE']:,.0f} | "
          f"R² {test_metrics['R2']:.3f}")
    print(f"Saved {MODELS_DIR / 'final_model.joblib'}")
    return metadata


if __name__ == "__main__":
    main()
