"""Tests for the trained model, the split and the inference API."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_raw_data
from src.predict import MODEL_PATH, load_model, predict_frame, predict_price, validate_spec
from src.preprocessing import GROUP_KEY, clean_dataset
from src.train import grouped_train_test_split, load_model_ready_data

SPEC = {"model": "Samsung Galaxy M34 5G", "processor": "Exynos 1280", "storage_gb": 128, "ram_gb": 6,
        "battery_mah": 6000, "rear_camera_mp": 50, "front_camera_mp": 13, "ai_lens": 0, "display_size_cm": 16.51}


def test_grouped_split_has_no_model_overlap():
    clean, _, _ = clean_dataset(load_raw_data())
    train, test = grouped_train_test_split(clean)
    assert not set(train[GROUP_KEY]) & set(test[GROUP_KEY])
    assert len(train) + len(test) == len(clean)


def test_saved_split_matches_code():
    clean, _, _ = clean_dataset(load_raw_data())
    _, test_code = grouped_train_test_split(clean)
    _, test_saved = load_model_ready_data()
    assert set(test_code[GROUP_KEY]) == set(test_saved[GROUP_KEY])


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_model_loads_and_predicts_plausibly():
    result = predict_price(SPEC)
    assert 5_000 < result["predicted_price"] < 60_000
    assert result["warnings"] == []


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_more_storage_never_lowers_price():
    low = predict_price({**SPEC, "storage_gb": 64})["predicted_price"]
    high = predict_price({**SPEC, "storage_gb": 256})["predicted_price"]
    assert high > low


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_batch_prediction_matches_single():
    df = pd.DataFrame([SPEC, {**SPEC, "ram_gb": 8}])
    batch = predict_frame(df)
    assert abs(batch.iloc[0] - predict_price(SPEC)["predicted_price"]) <= 1


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_spec({k: v for k, v in SPEC.items() if k != "processor"})
    with pytest.raises(ValueError):
        validate_spec({**SPEC, "battery_mah": -5})
    with pytest.raises(ValueError):
        validate_spec({**SPEC, "ram_gb": "lots"})
    with pytest.raises(ValueError):
        validate_spec({**SPEC, "ai_lens": 3})


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_out_of_range_input_warns():
    _, warnings = validate_spec({**SPEC, "ram_gb": 12})
    assert any("ram_gb" in w for w in warnings)
