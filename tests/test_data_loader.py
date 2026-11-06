"""Tests for src.data_loader."""

import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, RAW_DATA_PATH, load_raw_data, validate_schema


def test_raw_file_exists():
    assert RAW_DATA_PATH.exists(), "Raw dataset must be at data/raw/Processed_Flipdata.csv"


def test_load_raw_data_shape_and_columns():
    df = load_raw_data()
    assert df.shape == (541, 12)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_raw_target_is_unconverted_string():
    # Loading must not silently alter the raw data.
    df = load_raw_data()
    assert df["Prize"].iloc[0] == "7,299"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nope.csv")


def test_schema_mismatch_raises():
    bad = pd.DataFrame({"Model": ["x"], "Unexpected": [1]})
    with pytest.raises(ValueError):
        validate_schema(bad)
