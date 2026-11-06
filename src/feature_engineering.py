"""Day 4 — feature engineering and the reusable preprocessing pipeline.

`SpecFeatureBuilder` turns the cleaned input columns (including free-text `model` and
`processor`) into numeric/categorical features. It is the first step of every model pipeline,
so training, evaluation and the Streamlit app all use exactly the same code.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------- brand
BRAND_ALIASES = {
    "i kall": "I Kall", "iqoo": "iQOO", "oneplus": "OnePlus", "realme": "realme",
    "redmi": "Redmi", "poco": "POCO", "vivo": "vivo", "oppo": "OPPO", "samsung": "Samsung",
    "motorola": "Motorola", "infinix": "Infinix", "tecno": "Tecno", "google": "Google",
    "apple": "Apple", "nokia": "Nokia", "lava": "LAVA", "itel": "itel", "micromax": "Micromax",
    "nothing": "Nothing",
}


def extract_brand(model: str) -> str:
    """First word of the model name, mapped to a canonical brand ('I Kall' is two words)."""
    m = str(model).strip().lower()
    if m.startswith("i kall"):
        return "I Kall"
    first = m.split()[0] if m else "unknown"
    first = re.sub(r"\d+$", "", first)  # 'micromax1' -> 'micromax'
    return BRAND_ALIASES.get(first, first.title())


def is_5g_model(model: str) -> int:
    """Whether the listing name advertises 5G."""
    return int(bool(re.search(r"5\s*g\b", str(model), flags=re.I)))


def is_premium_variant(model: str) -> int:
    """Name suffixes retailers use for higher-spec variants: Pro, Plus/+, Ultra, Max, Fusion."""
    return int(bool(re.search(r"\b(pro|plus|ultra|max|fusion)\b|\+", str(model), flags=re.I)))


# ----------------------------------------------------------------------------- processor
# Tier: 0 = entry, 1 = mainstream, 2 = upper-mid, 3 = flagship. Based on the chip family's
# public positioning (vendor naming conventions), not on prices in this dataset.
_TIER_RULES = [
    # flagship
    (r"snapdragon\s*8\s*[+s]?\s*gen|snapdragon\s*888|snapdragon\s*7\+\s*gen|dimensity\s*(8020|8100|8200|9\d{3})|tensor|\bios\b|6 core", 3),
    # upper-mid
    (r"snapdragon\s*(7[3-9]\d|8[4-6]\d)|dimensity\s*(10\d\d|1[1-3]\d\d|7200)|exynos\s*1380", 2),
    # mainstream
    (r"snapdragon\s*(6\d\d|720|4\s*gen)|dimensity\s*(6\d{3}|[7-9]\d\d)|helio\s*g(7\d|8\d|9\d)|\bg9\d\b|exynos\s*(850|1330)|s5e8535|unisoc\s*t7\d\d|octa-core\(exynos\)|exynos octa", 1),
    # entry
    (r"helio\s*(a2\d|p3\d|g2\d|g3\d)|\bg3\d\b|unisoc|spreadtrum|sc98|sc65|snapdragon\s*4[0-4]\d|mt62|mtk62", 0),
]

_VENDOR_RULES = [
    (r"snapdragon|qualcomm", "Qualcomm"),
    (r"dimensity|helio|mediatek|meditek|\bmtk|\bmt\d|^g\d\d$", "MediaTek"),
    (r"exynos|s5e8535", "Samsung Exynos"),
    (r"unisoc|spreadtrum|sc98|sc65|\bt6\d\d", "Unisoc"),
    (r"tensor", "Google Tensor"),
    (r"\bios\b|6 core", "Apple"),
]


def processor_vendor(processor: str, brand: str | None = None) -> str:
    p = str(processor).lower()
    for pattern, vendor in _VENDOR_RULES:
        if re.search(pattern, p):
            return vendor
    if brand == "Apple":
        return "Apple"
    return "Unknown"


def processor_tier(processor: str, brand: str | None = None) -> float:
    """Ordinal chip tier, or NaN when the text does not identify a chip (e.g. 'Oxygen OS')."""
    p = str(processor).lower()
    if brand == "Apple":
        return 3.0
    for pattern, tier in _TIER_RULES:
        if re.search(pattern, p):
            return float(tier)
    return np.nan


# ----------------------------------------------------------------------------- transformer
NUMERIC_FEATURES = [
    "storage_gb", "ram_gb", "battery_mah", "rear_camera_mp", "front_camera_mp",
    "display_size_cm", "ai_lens", "processor_tier", "is_5g", "is_premium_variant",
]
CATEGORICAL_FEATURES = ["brand", "processor_vendor"]
ENGINEERED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SpecFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless transformer: cleaned inputs -> engineered feature frame.

    Learns nothing from data, so it cannot leak information between train and test.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        brand = X["model"].map(extract_brand)
        out = pd.DataFrame(index=X.index)
        for col in ["storage_gb", "ram_gb", "battery_mah", "rear_camera_mp",
                    "front_camera_mp", "display_size_cm", "ai_lens"]:
            out[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        out["processor_tier"] = [processor_tier(p, b) for p, b in zip(X["processor"], brand)]
        out["is_5g"] = X["model"].map(is_5g_model).astype(float)
        out["is_premium_variant"] = X["model"].map(is_premium_variant).astype(float)
        out["brand"] = brand.astype(object)
        out["processor_vendor"] = [processor_vendor(p, b) for p, b in zip(X["processor"], brand)]
        out["processor_vendor"] = out["processor_vendor"].astype(object)
        return out[ENGINEERED_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_COLUMNS, dtype=object)


def make_column_transformer(scale_numeric: bool = False, min_brand_frequency: int = 5,
                            numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """Imputation + encoding, fitted on training data only (it lives inside the pipeline).

    - Numeric: median imputation (RAM and processor tier have missing values); optional scaling
      for linear models only — tree models do not need it.
    - Categorical: one-hot with rare categories grouped into 'infrequent' and unseen categories
      at prediction time handled safely.
    """
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric_features)]
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=min_brand_frequency, sparse_output=False)),
            ]),
            categorical_features,
        ))
    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def make_preprocessor(scale_numeric: bool = False, **kwargs) -> Pipeline:
    """Full preprocessing: SpecFeatureBuilder -> imputation/encoding."""
    return Pipeline([
        ("features", SpecFeatureBuilder()),
        ("encode", make_column_transformer(scale_numeric=scale_numeric, **kwargs)),
    ])
