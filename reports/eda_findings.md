# Day 2 — EDA Findings

Notebook: `notebooks/02_exploratory_data_analysis.ipynb` · Data: raw file with types converted only (541 rows).
Figures: `reports/figures/eda_*.png`.

## Target
- Price is right-skewed (skewness 2.43); 77 % of listings are under ₹20k. Log price is close to symmetric (−0.68).
  → Model **log(price)**; report errors in rupees.

## Associations with price (Spearman, smartphones only)
| Feature | ρ | Note |
|---|---|---|
| Processor tier (derived) | 0.86 | Medians ₹8.3k → ₹15k → ₹25k → ₹33k from entry to flagship |
| Storage | 0.81 | Median roughly doubles per step: 32 → 64 → 128 → 256 GB |
| Front camera | 0.79 | Upgraded together with overall tier |
| RAM | 0.77 | Capped at 8 GB in data; premium end understated |
| Rear camera | 0.59 | MP is not quality: 12 MP iPhones/Pixels median ~₹43k |
| Display size | 0.14 | Almost no variation among smartphones |
| AI Lens | −0.23 | Infinix-only flag → brand effect, not a value effect |
| Battery | −0.46 | Budget phones ship 6,000 mAh; flagships 4,000–4,500 mAh → segment marker |

## Other findings
- **Brand** separates price bands strongly, but premium brands have few listings (Apple 5, Google 9, OnePlus 5).
- **Model names** carry signal: “5G” listings median ₹19.7k vs ₹10k; Pro/Plus/Ultra/Max ₹21.5k vs ₹12.5k.
- **Colour**: 275 values, 137 appear once → no generalisable information.
- **Multicollinearity**: storage, RAM, front camera, processor tier inter-correlate at 0.6–0.75 (specs are upgraded
  as a bundle) → importance will be shared among them.
- **Outliers**: IQR flags 152 battery values, but they are normal market variation; real errors are only the
  41.94 cm display values and feature-phone storage/RAM.
- **Feature phones** form a separate category; removing them strengthens storage/RAM correlations.
- **Battery vs price coloured by processor tier** shows the negative battery correlation is driven by market segment.

## Decisions carried into Day 3–4
1. Exclude feature phones (scope = smartphones).
2. Keep genuine outliers (premium phones, 7,000 mAh battery).
3. Engineer processor tier/vendor, brand, 5G and premium-variant flags; drop colour.
4. Use log(price) as the modelling target; prefer models/validation that handle correlated specs.
