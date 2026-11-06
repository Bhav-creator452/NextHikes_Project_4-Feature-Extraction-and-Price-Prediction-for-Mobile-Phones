# Day 1 — Initial Data-Quality Report

**Notebook:** `notebooks/01_data_exploration.ipynb` · **Data:** `data/raw/Processed_Flipdata.csv` (unchanged)

## 1. Dataset overview
- 541 rows × 12 columns; schema matches the brief.
- One row = one product listing variant (model × colour × storage/RAM).
- 541 rows cover only **~181 distinct models** and **~280 distinct model/storage/RAM configurations**.
- Target `Prize` (INR) is stored as a comma-formatted string; all 541 values convert cleanly.
- Price: min ₹920 · median ₹13,499 · mean ₹16,228 · max ₹80,999 · skewness 2.43 (strongly right-skewed).

## 2. Findings

| ID | Issue | Rows | Evidence | Classification | Proposed handling (decide Day 3) |
|---|---|---|---|---|---|
| Q1 | Target stored as text | 541 | `"7,299"` | Formatting | Strip commas → int |
| Q2 | Camera specs stored as text | 541 | `"50MP"` | Formatting | Strip `MP` → int |
| Q3 | Duplicate listings (ignoring index) | 10 pairs (20 rows) | Identical on all 11 real fields | Duplicate | Drop the repeat |
| Q4 | Same variant at different prices | 4 variants | e.g. POCO C50 32/2 at ₹5,649 and ₹6,999 | Genuine price noise | Keep; note as irreducible error |
| Q5 | Feature-phone rows inherit colour/memory/RAM from the row above | ~16 | In all 16 rows, colour equals previous row; e.g. micromax 1 at ₹1,299 recorded as 256 GB / 8 GB | Scraping error | Choose scope: exclude, or keep with storage/RAM flagged |
| Q6 | `Mobile Height` = 41.94 | 2 | Same model (MOTOROLA G62 5G) = 16.64 elsewhere | Data-entry error | Replace with 16.64 (within-dataset evidence) |
| Q7 | RAM never exceeds 8 GB; premium 256 GB variants recorded as 2–4 GB | ~10 | Note 12 Pro+ 256 GB: 8 GB at ₹29,999 vs 2/4 GB at ₹32,999 | Suspected truncation of 12 GB | Flag; decide correct/keep/exclude with justification |
| Q8 | iPhone RAM inconsistent | 5 | iPhone 14 Plus same price with RAM 6/3/4 | Data error | Flag |
| Q9 | Mislabelled model | 1 | “Google Pixel 6a” (index 334) has Nothing Phone 1 specs | Labelling error | Flag / relabel with justification |
| Q10 | `Processor_` not a chipset | ~25 | `Oxygen OS`, `ios`, `Octa Core`, `MediaTek` | Disguised missing | “Unknown” category in feature engineering |
| Q11 | Chipset spelling variants | many | *Snapdragon 680* written 5 ways; typo *Meditek* | Inconsistency | Normalise on Day 4 |
| Q12 | Case variants in `Model`/`Colour` | 6 / 4 | `OPPO A17k` vs `OPPO A17K` | Inconsistency | Normalise case/whitespace |
| Q13 | Mis-encoded characters | 2 | `7Â° Purple`, `QualcommÂ® Snapdragonâ_x0084_¢ 765G` | Encoding | Repair text |
| Q14 | Garbled colour `M3)` | 5 | redmi min/max/mono rows | Parsing error | Treat as unknown |
| Q15 | `Unnamed: 0` index with 102 gaps | 541 | 0–642 | Artefact | Drop |
| Q16 | `0MP` cameras | 10 rear / 16 front | Feature phones | **Valid** (no camera) | Keep as 0 |
| Q17 | Premium price outliers | 25 above IQR fence (₹34,265) | iPhone 14 Plus, Galaxy S23, Pixel 7 | **Genuine extremes** | Keep; consider log target |
| Q18 | 7000 mAh battery | 1 | Tecno Pova 3 | **Valid extreme** | Keep |
| Q19 | `AI Lens` only on Infinix | 34 | No other brand has 1 | Interpretation risk | Watch for brand confounding |
| Q20 | `Mobile Height` is display size | all | Values match Flipkart “16.76 cm (6.6 inch)” display spec | Naming | Rename `display_size_cm` |

## 3. Implications for later days
- **Validation (Day 7):** with many colour-only variants, a random split leaks phone identity into the test set. Use
  **group-aware splitting by normalised model name**, and report random-split results only for comparison.
- **Target (Days 7–8):** right skew → evaluate a log-transformed target; report MAE alongside RMSE.
- **Encoding (Day 4):** `Model`, `Colour`, `Processor_` are too granular to one-hot encode directly; derive brand and
  processor vendor/tier instead.
- **Interpretation (Days 5–10):** RAM's premium-end relationship with price is probably understated (Q7); `AI Lens`
  may act as a brand indicator (Q19).

## 4. Open decision for Day 3
**Scope:** should the model cover feature phones (~16 rows, ~3 % of data, unreliable storage/RAM) or smartphones only?
Recommendation to evaluate on Day 3: model smartphones only and state that clearly as a limitation, because the feature-phone
rows' storage/RAM values are copied from other phones and would teach the model wrong relationships.
