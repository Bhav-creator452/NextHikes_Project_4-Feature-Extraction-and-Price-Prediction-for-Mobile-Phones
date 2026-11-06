"""Inference: load the saved pipeline and estimate the price of a phone specification.

Example
-------
>>> from src.predict import predict_price
>>> predict_price({"model": "Samsung Galaxy M34 5G", "processor": "Exynos 1280",
...                "storage_gb": 128, "ram_gb": 6, "battery_mah": 6000,
...                "rear_camera_mp": 50, "front_camera_mp": 13, "ai_lens": 0,
...                "display_size_cm": 16.51})
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.data_loader import PROJECT_ROOT
from src.preprocessing import MODEL_INPUT_COLUMNS

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

NUMERIC_INPUTS = ["storage_gb", "ram_gb", "battery_mah", "rear_camera_mp",
                  "front_camera_mp", "ai_lens", "display_size_cm"]
HARD_LIMITS = {  # physically sensible bounds; anything outside is rejected
    "storage_gb": (1, 2048), "ram_gb": (1, 32), "battery_mah": (500, 10000),
    "rear_camera_mp": (0, 300), "front_camera_mp": (0, 100), "display_size_cm": (5, 25),
}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/final_model.joblib not found. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_spec(spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """Check one specification. Returns a one-row DataFrame and a list of warnings.

    Raises ValueError for missing fields or impossible values. Values that are valid but outside
    the range seen in training produce a warning (the estimate is an extrapolation).
    """
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in spec]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    row = {c: spec[c] for c in MODEL_INPUT_COLUMNS}
    for c in NUMERIC_INPUTS:
        try:
            row[c] = float(row[c])
        except (TypeError, ValueError):
            raise ValueError(f"{c} must be numeric, got {row[c]!r}") from None
    if row["ai_lens"] not in (0.0, 1.0):
        raise ValueError("ai_lens must be 0 or 1")
    for c, (lo, hi) in HARD_LIMITS.items():
        if not lo <= row[c] <= hi:
            raise ValueError(f"{c}={row[c]} is outside the plausible range [{lo}, {hi}]")
    for c in ["model", "processor"]:
        if not str(row[c]).strip():
            raise ValueError(f"{c} must not be empty")

    warnings = []
    ranges = load_metadata().get("training_ranges", {})
    for c, (lo, hi) in ranges.items():
        if not lo <= row[c] <= hi:
            warnings.append(f"{c}={row[c]:g} is outside the training range [{lo:g}, {hi:g}]; "
                            "the estimate is an extrapolation.")
    return pd.DataFrame([row]), warnings


def predict_price(spec: dict) -> dict:
    """Estimated price in INR plus any warnings."""
    X, warnings = validate_spec(spec)
    price = float(load_model().predict(X)[0])
    return {"predicted_price": round(price), "warnings": warnings}


def predict_frame(df: pd.DataFrame) -> pd.Series:
    """Batch prediction for a DataFrame with the model input columns."""
    return pd.Series(load_model().predict(df[MODEL_INPUT_COLUMNS]), index=df.index, name="predicted_price")
