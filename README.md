# Feature Extraction and Price Prediction for Mobile Phones

NextHikes Data Science Internship — Project 4.

A mobile-phone retailer wants to know **which specifications drive smartphone prices** and needs a
**regression model that estimates a phone's price from its specifications**. This repository covers the full
workflow: data understanding, cleaning, feature engineering, feature extraction/selection, modelling, evaluation,
explainability, and business recommendations.

> **Status:** Day 1 of 12 complete — dataset inspection, data dictionary, initial data-quality report.

## Dataset
`data/raw/Processed_Flipdata.csv` — 541 Flipkart India listings × 12 columns (model, colour, storage, RAM, battery,
rear/front camera, AI lens flag, display size, processor, price in INR). The raw file is never modified.
See [`reports/data_dictionary.md`](reports/data_dictionary.md).

## Repository structure (current)
```
mobile-price-prediction/
├── data/
│   ├── raw/Processed_Flipdata.csv       # original data, read-only
│   └── processed/                       # created from Day 3
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   ├── data_dictionary.md
│   └── day01_data_quality_report.md
├── src/
│   ├── __init__.py
│   └── data_loader.py                   # load + schema validation
├── tests/
│   └── test_data_loader.py
├── requirements.txt
├── pytest.ini
└── .gitignore
```

## Setup
```bash
git clone <your-repo-url>
cd mobile-price-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest           # 5 tests should pass
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Progress
| Day | Stage | Status |
|---|---|---|
| 1 | Project setup & dataset understanding | ✅ |
| 2 | Exploratory data analysis | ⏳ |
| 3 | Data cleaning & preprocessing | — |
| 4 | Feature engineering & encoding | — |
| 5 | Feature extraction & selection | — |
| 6 | Initial feature importance | — |
| 7 | Baseline models | — |
| 8 | Advanced models & tuning | — |
| 9 | Evaluation & validation | — |
| 10 | Final explainability | — |
| 11 | Visualisations & recommendations | — |
| 12 | Documentation & submission | — |

## Key Day 1 findings
- Target `Prize` is a comma-formatted string (₹920–₹80,999, strongly right-skewed).
- 541 rows describe only ~181 distinct models, so validation must be **grouped by model** to measure performance on
  unseen phones.
- Known data problems: feature-phone rows that inherit storage/RAM/colour from the previous row, suspected truncation
  of 12 GB RAM, a 41.94 cm display-size error, chipset names with many spellings. Full list in
  [`reports/day01_data_quality_report.md`](reports/day01_data_quality_report.md).
