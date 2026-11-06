# Findings Summary — Mobile Phone Price Prediction

All numbers come from the executed notebooks (`notebooks/01`–`11`). Test-set figures are for **35 phone models never
seen during training, tuning or feature selection**.

## 1. Data
- 541 Flipkart India listings → **516 clean smartphone rows** (172 models) after removing 10 duplicates and
  15 feature phones whose storage/RAM/colour were copied from neighbouring rows by the scraper.
- Fixes: price/camera parsing, a 41.94 cm display error, a mislabelled "Pixel 6a" row, encoding errors, 3 model-name
  errors. 15 implausible RAM values (premium 256 GB phones at ≤ 4 GB; inconsistent iPhone values) set to missing.
- Price is right-skewed (₹5k–₹81k, median ₹14k) → modelled as log(price).

## 2. What drives price (association, not causation)
| Rank | Input | Evidence on unseen phones | Effect in the model (other specs fixed) |
|---|---|---|---|
| 1 | **Processor** (tier + vendor) | +₹897 MAE when shuffled; Shapley ±12 % | mainstream → flagship chip **+49 %**; entry → mainstream +14 % |
| 2 | **Brand / model name** (brand, 5G, Pro) | +₹701 | Apple +60 %, Google +34 % … Infinix −25 % (vs average) |
| 3 | **RAM** | +₹469 | 4 → 8 GB **+21 %** (likely understated: RAM capped at 8 GB in data) |
| 4 | **Front camera** | +₹359 | 8 → 16 MP +6 % |
| 5 | **Battery** | +₹323 | 5,000 → 6,000 mAh **−10 %** (large batteries mark budget phones) |
| 6 | **Storage** | +₹172 | each doubling +11–12 %; strongly correlated (ρ 0.81) but overlaps RAM/processor |
| — | Rear camera MP, display size, AI lens | ≈ 0 or negative | MP count is not a quality signal (12 MP flagships vs 50 MP budget) |

Interactions are weak (≤ ±1.6 % deviation from additive-in-log). Spec effects behave multiplicatively.

## 3. Model
- **Final model:** equal-weight average of Ridge regression and tuned Gradient Boosting, both on log(price), with a
  shared feature-engineering pipeline (brand, processor tier/vendor, 5G/Pro flags; one-hot with rare-category pooling).
- **Validation:** grouped by phone model. Random splits overstated tree-model accuracy by up to 4× (Decision Tree
  CV MAE ₹847 random vs ₹3,470 grouped).

| Model | Grouped CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Dummy (median) | ₹7,411 | ₹6,163 | ₹10,469 | −0.01 |
| Linear Regression | ₹2,330 | ₹2,936 | ₹5,785 | 0.69 |
| Random Forest (tuned) | ₹2,807 | ₹3,178 | ₹6,192 | 0.65 |
| Gradient Boosting (tuned) | ₹2,469 | ₹2,969 | ₹5,349 | 0.74 |
| HistGradientBoosting (tuned) | ₹2,516 | ₹2,878 | ₹5,158 | 0.76 |
| **Ridge + GB average (final)** | **₹2,171** | **₹2,796** | ₹5,190 | 0.75 |

- 64 % of unseen phones within ±₹2,000; 73 % within ±20 %. Test MAE 95 % bootstrap interval ₹1.7k–₹4.2k.
- **Reliable below ₹10k** (90 % within ±20 %); usable to ₹20k; **under-predicts ≥ ₹30k** (43 % within ±20 %).
- Largest errors: older flagship chips sold cheaply (realme X3, Galaxy F23) and premium Samsung phones (S23, A54).
- Learning curve still falling → more data (especially premium phones) would help.

## 4. Feature selection & extraction
- Consensus core across Pearson/Spearman, MI, F-test, RFE: processor tier, RAM, storage, front camera.
- PCA: PC1 = "premium-ness" (14.9 % of variance); 17 components for 90 %. Not adopted (no reliable gain, loses
  interpretability).
- Selection helped only untuned boosting; after tuning all features were best.

## 5. Recommendations
1. Set the price band by **processor tier**, then adjust for **RAM/storage** variants and **brand** positioning.
2. Don't justify premiums with rear-camera megapixels or battery size; market big batteries as budget value.
3. Use the model for **preliminary estimates below ~₹20k**; require human review for premium phones, older
   flagships, new brands and out-of-range specs.
4. Add launch date, true RAM (12 GB+), and more premium listings; retrain quarterly.
