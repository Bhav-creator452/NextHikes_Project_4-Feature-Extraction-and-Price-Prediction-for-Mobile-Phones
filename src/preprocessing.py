"""Day 3 — data cleaning.

`clean_dataset` turns the raw Flipkart file into the analysis-ready table. Every rule here is
documented in reports/data_cleaning_log.md and justified with evidence in
notebooks/01_data_exploration.ipynb and 03_data_preprocessing.ipynb.

The function never modifies its input; it returns a cleaned copy plus a log of what changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RENAME_MAP = {
    "Model": "model",
    "Colour": "colour",
    "Memory": "storage_gb",
    "RAM": "ram_gb",
    "Battery_": "battery_mah",
    "Rear Camera": "rear_camera_mp",
    "Front Camera": "front_camera_mp",
    "AI Lens": "ai_lens",
    "Mobile Height": "display_size_cm",
    "Processor_": "processor",
    "Prize": "price",
}

# Columns that are fed to the model pipeline (feature engineering happens inside the pipeline).
MODEL_INPUT_COLUMNS = [
    "model",
    "processor",
    "storage_gb",
    "ram_gb",
    "battery_mah",
    "rear_camera_mp",
    "front_camera_mp",
    "ai_lens",
    "display_size_cm",
]
TARGET = "price"
GROUP_KEY = "model_key"

# Mis-encoded text (UTF-8 bytes decoded as Latin-1) found on Day 1.
ENCODING_FIXES = {
    "7Â° Purple": "7° Purple",
    "QualcommÂ® Snapdragonâ_x0084_¢ 765G": "Qualcomm Snapdragon 765G",
}

# Model-name repairs supported by evidence inside the dataset (see 03 notebook).
MODEL_NAME_FIXES = {
    "vivo Y565G": "vivo Y56 5G",              # same colour/specs/price band as vivo Y56 5G rows
    "I Kall Z19Pro Flash blue": "I Kall Z19Pro",  # colour text leaked into the model name
    "micromax1": "micromax 1",
}

# Feature-phone definition: hardware that no smartphone in the data has.
FEATURE_PHONE_MAX_BATTERY = 2600   # mAh
FEATURE_PHONE_MAX_DISPLAY = 10.0   # cm


@dataclass
class CleaningLog:
    """Records each cleaning step and how many rows it affected."""

    steps: list = field(default_factory=list)

    def add(self, step: str, rows_affected: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def parse_price(series: pd.Series) -> pd.Series:
    """Convert strings like '7,299' to integers. Raises if any value cannot be parsed."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].unique()[:5]
        raise ValueError(f"Unparseable price values: {bad}")
    return numeric.astype(int)


def parse_megapixels(series: pd.Series) -> pd.Series:
    """Convert strings like '50MP' to integers. '0MP' (no camera) becomes 0."""
    extracted = series.astype(str).str.extract(r"^\s*(\d+(?:\.\d+)?)\s*MP\s*$", flags=re.I)[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.isna().any():
        bad = series[numeric.isna()].unique()[:5]
        raise ValueError(f"Unparseable camera values: {bad}")
    return numeric.astype(int)


def normalise_text(series: pd.Series) -> pd.Series:
    """Strip, collapse internal whitespace, and repair known encoding errors."""
    s = series.astype(str).replace(ENCODING_FIXES)
    return s.str.strip().str.replace(r"\s+", " ", regex=True)


def make_model_key(model: pd.Series) -> pd.Series:
    """Case-insensitive key used to group variants of the same phone (for grouped CV)."""
    return model.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def is_feature_phone(df: pd.DataFrame) -> pd.Series:
    """Rows whose battery or display is far below anything a smartphone in the data has."""
    return (df["battery_mah"] < FEATURE_PHONE_MAX_BATTERY) | (
        df["display_size_cm"] < FEATURE_PHONE_MAX_DISPLAY
    )


def clean_dataset(raw: pd.DataFrame, exclude_feature_phones: bool = True):
    """Apply all Day 3 cleaning rules.

    Returns
    -------
    clean : pd.DataFrame
        Analysis-ready rows (smartphones only by default).
    excluded : pd.DataFrame
        Rows removed by scope or duplicate rules, with a `reason` column.
    log : CleaningLog
    """
    log = CleaningLog()
    df = raw.copy()

    # 1. Drop the leftover index, rename columns.
    df = df.drop(columns=["Unnamed: 0"]).rename(columns=RENAME_MAP)
    log.add("Drop 'Unnamed: 0' and rename columns", len(df), "index artefact; snake_case names")

    # 2. Type conversions.
    df["price"] = parse_price(df["price"])
    df["rear_camera_mp"] = parse_megapixels(df["rear_camera_mp"])
    df["front_camera_mp"] = parse_megapixels(df["front_camera_mp"])
    log.add("Convert price and camera strings to integers", len(df), "'7,299' -> 7299; '50MP' -> 50")

    # 3. Text hygiene.
    before = df[["model", "colour", "processor"]].copy()
    for col in ["model", "colour", "processor"]:
        df[col] = normalise_text(df[col])
    df["model"] = df["model"].replace(MODEL_NAME_FIXES)
    changed = (before != df[["model", "colour", "processor"]]).any(axis=1).sum()
    log.add("Normalise whitespace, fix encoding and 3 model-name errors", changed)

    # Colour: unify case; the 'M3)' fragment is not a colour.
    df["colour"] = df["colour"].str.title()
    n_garbled = (df["colour"] == "M3)").sum()
    df.loc[df["colour"] == "M3)", "colour"] = "Unknown"
    log.add("Title-case colours; 'M3)' -> 'Unknown'", n_garbled)

    # 4. Mislabelled model: a 'Google Pixel 6a' row whose six spec fields equal 'nothing phone 1'.
    spec_cols = ["storage_gb", "ram_gb", "battery_mah", "rear_camera_mp",
                 "front_camera_mp", "display_size_cm", "processor"]
    nothing = df[df["model"].str.lower() == "nothing phone 1"][spec_cols].drop_duplicates()
    pixel = df["model"].str.lower() == "google pixel 6a"
    mislabelled = pixel & df[spec_cols].apply(tuple, axis=1).isin(nothing.apply(tuple, axis=1))
    df.loc[mislabelled, "model"] = "nothing phone 1"
    log.add("Relabel 'Google Pixel 6a' row with Nothing Phone 1 specs", mislabelled.sum(),
            "all spec fields identical to nothing phone 1 rows")

    # 5. Display-size error: 41.94 cm on MOTOROLA G62 5G; the model's other rows say 16.64.
    bad_display = df["display_size_cm"] > 20
    for idx in df.index[bad_display]:
        same_model = (df["model"] == df.at[idx, "model"]) & ~bad_display
        if same_model.any():
            df.at[idx, "display_size_cm"] = df.loc[same_model, "display_size_cm"].mode().iloc[0]
    log.add("Replace impossible display size (41.94 cm) with same-model value", bad_display.sum())

    # 6. Model key for grouping (case-insensitive).
    df["model_key"] = make_model_key(df["model"])

    excluded_parts = []

    # 7. Exact duplicate listings.
    dup = df.duplicated(keep="first")
    excluded_parts.append(df[dup].assign(reason="duplicate listing"))
    df = df[~dup]
    log.add("Drop duplicate listings (all fields identical)", dup.sum())

    # 8. Scope: feature phones (storage/RAM/colour copied from neighbouring rows by the scraper).
    if exclude_feature_phones:
        fp = is_feature_phone(df)
        excluded_parts.append(df[fp].assign(reason="feature phone (out of scope)"))
        df = df[~fp]
        log.add("Exclude feature phones (battery < 2600 mAh or display < 10 cm)", fp.sum(),
                "unreliable storage/RAM; different product category")

    # 9. Implausible RAM -> missing (imputed later inside the training pipeline).
    #    Rule A: a 256 GB smartphone recorded with <= 4 GB RAM (suspected '12 GB' truncation).
    #    Rule B: Apple rows — the same iPhone at the same price is recorded with 3, 4 and 6 GB.
    #    Neither rule uses price, so no target information enters the features.
    rule_a = (df["storage_gb"] == 256) & (df["ram_gb"] <= 4)
    rule_b = df["model"].str.upper().str.startswith("APPLE")
    suspect = rule_a | rule_b
    df["ram_gb"] = df["ram_gb"].astype(float)
    df.loc[suspect, "ram_gb"] = np.nan
    log.add("Set implausible RAM to missing (256 GB with <= 4 GB RAM; Apple rows)", suspect.sum(),
            "imputed with training median inside the pipeline")

    df = df.reset_index(drop=True)
    excluded = pd.concat(excluded_parts, ignore_index=True) if excluded_parts else pd.DataFrame()
    return df, excluded, log
