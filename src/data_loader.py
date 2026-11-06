"""Loading and schema validation for the raw Flipkart mobile-phone dataset.

The raw CSV is treated as read-only. Every function here returns a *new*
DataFrame, so the file on disk is never modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Processed_Flipdata.csv"

# Column names exactly as they appear in the raw file (verified on Day 1).
EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "Model",
    "Colour",
    "Memory",
    "RAM",
    "Battery_",
    "Rear Camera",
    "Front Camera",
    "AI Lens",
    "Mobile Height",
    "Processor_",
    "Prize",
]

TARGET_RAW = "Prize"  # The target column is spelled "Prize" in the source file.


def load_raw_data(path: str | Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Load the raw dataset exactly as stored on disk.

    Parameters
    ----------
    path : str or Path, optional
        Location of the CSV. Defaults to ``data/raw/Processed_Flipdata.csv``.
    validate : bool, default True
        If True, check that the expected columns are present.

    Returns
    -------
    pd.DataFrame
        Unmodified raw data. ``Prize`` stays a string (e.g. ``"7,299"``);
        conversion happens in the preprocessing stage, not here.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If ``validate`` is True and the schema differs from the expected one.
    """
    csv_path = Path(path) if path is not None else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Place Processed_Flipdata.csv in data/raw/."
        )

    df = pd.read_csv(csv_path)

    if validate:
        validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if columns are missing or unexpected ones appear."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}. Unexpected: {extra}.")
