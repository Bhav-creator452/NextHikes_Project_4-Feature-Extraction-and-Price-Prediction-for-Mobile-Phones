"""Tests for the cleaning rules in src.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data
from src.preprocessing import (MODEL_INPUT_COLUMNS, clean_dataset, is_feature_phone, normalise_text,
                               parse_megapixels, parse_price)


@pytest.fixture(scope="module")
def cleaned():
    return clean_dataset(load_raw_data())


def test_parse_price():
    assert parse_price(pd.Series(["7,299", "920", "80,999"])).tolist() == [7299, 920, 80999]


def test_parse_price_rejects_garbage():
    with pytest.raises(ValueError):
        parse_price(pd.Series(["7,299", "N/A"]))


def test_parse_megapixels():
    assert parse_megapixels(pd.Series(["50MP", "0MP", "108MP"])).tolist() == [50, 0, 108]
    with pytest.raises(ValueError):
        parse_megapixels(pd.Series(["dual camera"]))


def test_normalise_text_fixes_encoding_and_spaces():
    out = normalise_text(pd.Series(["Exynos 1330, Octa Core ", "7Â° Purple"]))
    assert out.tolist() == ["Exynos 1330, Octa Core", "7° Purple"]


def test_clean_shape_and_exclusions(cleaned):
    clean, excluded, _ = cleaned
    assert len(clean) == 516
    assert excluded["reason"].value_counts().to_dict() == {"feature phone (out of scope)": 15, "duplicate listing": 10}


def test_clean_invariants(cleaned):
    clean, _, _ = cleaned
    assert not clean.duplicated().any()
    assert clean["display_size_cm"].max() < 20                 # 41.94 cm error fixed
    assert not is_feature_phone(clean).any()                  # scope
    assert clean["price"].dtype.kind == "i" and (clean["price"] > 0).all()
    assert set(MODEL_INPUT_COLUMNS) <= set(clean.columns)
    assert clean.isna().sum().drop("ram_gb").sum() == 0       # only RAM may be missing


def test_implausible_ram_set_to_missing(cleaned):
    clean, _, _ = cleaned
    assert clean["ram_gb"].isna().sum() == 15
    assert clean.loc[clean["model"].str.upper().str.startswith("APPLE"), "ram_gb"].isna().all()
    smart_256 = clean[(clean["storage_gb"] == 256)]
    assert not (smart_256["ram_gb"] <= 4).any()


def test_mislabelled_pixel_relabelled(cleaned):
    clean, _, _ = cleaned
    pixel = clean[clean["model"] == "Google Pixel 6a"]
    assert (pixel["processor"] == "Google Tensor").all()


def test_raw_input_not_modified():
    raw = load_raw_data()
    before = raw.copy()
    clean_dataset(raw)
    pd.testing.assert_frame_equal(raw, before)
