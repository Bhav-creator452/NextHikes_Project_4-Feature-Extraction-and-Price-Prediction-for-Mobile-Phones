"""Tests for feature extraction and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (ENGINEERED_COLUMNS, SpecFeatureBuilder, extract_brand, is_5g_model,
                                     is_premium_variant, make_preprocessor, processor_tier, processor_vendor)


@pytest.mark.parametrize("model, brand", [
    ("SAMSUNG Galaxy F13", "Samsung"), ("REDMI Note 12 Pro 5G", "Redmi"), ("I Kall Z19Pro", "I Kall"),
    ("micromax 1", "Micromax"), ("IQOO Neo 7 5G", "iQOO"), ("Brandnew X1", "Brandnew"),
])
def test_extract_brand(model, brand):
    assert extract_brand(model) == brand


@pytest.mark.parametrize("proc, brand, tier, vendor", [
    ("Mediatek Helio G35", None, 0, "MediaTek"),
    ("Unisoc T612", None, 0, "Unisoc"),
    ("Qualcomm Snapdragon 680", None, 1, "Qualcomm"),
    ("Mediatek Dimensity 1080", None, 2, "MediaTek"),
    ("Qualcomm Snapdragon 8+ Gen 1", None, 3, "Qualcomm"),
    ("Snapdragon 8s Gen 3", None, 3, "Qualcomm"),
    ("Google Tensor G2", None, 3, "Google Tensor"),
    ("6 Core", "Apple", 3, "Apple"),
    ("Meditek Helio G37", None, 0, "MediaTek"),   # typo in source data
])
def test_processor_mapping(proc, brand, tier, vendor):
    assert processor_tier(proc, brand) == tier
    assert processor_vendor(proc, brand) == vendor


def test_unknown_processor_is_nan():
    assert np.isnan(processor_tier("Oxygen OS"))
    assert processor_vendor("Oxygen OS") == "Unknown"


def test_name_flags():
    assert is_5g_model("POCO X5 5G") == 1 and is_5g_model("POCO X5") == 0
    assert is_premium_variant("REDMI Note 12 Pro+ 5G") == 1
    assert is_premium_variant("realme C55") == 0


def _sample():
    return pd.DataFrame([
        {"model": "POCO X5 5G", "processor": "Qualcomm Snapdragon 695", "storage_gb": 128, "ram_gb": 6,
         "battery_mah": 5000, "rear_camera_mp": 48, "front_camera_mp": 13, "ai_lens": 0, "display_size_cm": 16.94},
        {"model": "Infinix SMART 7", "processor": "Unisoc SC9863A1", "storage_gb": 64, "ram_gb": np.nan,
         "battery_mah": 6000, "rear_camera_mp": 13, "front_camera_mp": 5, "ai_lens": 1, "display_size_cm": 16.76},
    ])


def test_builder_output_columns():
    out = SpecFeatureBuilder().fit_transform(_sample())
    assert list(out.columns) == ENGINEERED_COLUMNS
    assert out.loc[0, "is_5g"] == 1 and out.loc[0, "processor_tier"] == 1


def test_preprocessor_handles_unseen_and_missing():
    prep = make_preprocessor().fit(pd.concat([_sample()] * 5, ignore_index=True))
    new = _sample().assign(model=["Acme Z 5G", "Other Phone"], processor=["Mystery", "Mystery"])
    out = prep.transform(new)
    assert not out.isna().any().any()
    assert out.shape[0] == 2
