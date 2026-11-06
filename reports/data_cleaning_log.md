# Day 3 — Data-Cleaning Log

Implementation: `src/preprocessing.py → clean_dataset()` · Evidence: `notebooks/01_…` and `03_data_preprocessing.ipynb`
Result: **541 raw rows → 516 clean smartphone rows** (172 distinct models). Raw file untouched.

| # | Step | Classification | Rows | Justification |
|---|---|---|---|---|
| 1 | Drop `Unnamed: 0`; rename to snake_case (`Prize`→`price`, `Mobile Height`→`display_size_cm`, …) | Artefact / readability | all | Index with 102 gaps; names made explicit |
| 2 | `price` "7,299" → 7299; cameras "50MP" → 50 | Formatting | all | Parsers raise on failure, so nothing is silently lost |
| 3 | Whitespace collapse; fix 2 mis-encoded strings; 3 model-name fixes; title-case colour; `M3)` → `Unknown` | Formatting / entry error | 23 + 5 | `vivo Y565G` matches vivo Y56 5G on colour, specs and price |
| 4 | Relabel one "Google Pixel 6a" row as "nothing phone 1" | Labelling error | 1 | All 7 spec fields identical to nothing phone 1; price within its range |
| 5 | Display 41.94 cm → 16.64 cm | Entry error | 2 | Same model's other rows show 16.64 |
| 6 | Remove duplicate listings | Duplicate | 10 | Identical on every field |
| 7 | Exclude feature phones (battery < 2,600 mAh or display < 10 cm) | Scope / unreliable values | 15 | Storage/RAM/colour copied from preceding listing by the scraper |
| 8 | RAM → NaN for 256 GB phones with ≤ 4 GB RAM and for Apple rows | Scraping error | 15 | Suspected "12 GB" truncation; iPhone RAM mutually inconsistent. Imputed with training median inside the pipeline. Rules do not use price. |

**Kept on purpose:** 25 premium-price phones (genuine extremes), batteries of 4,000–4,500 and 6,000–7,000 mAh
(valid market variation flagged by IQR), 108/200 MP cameras, repeated listings with different prices (real price variation).

**Outputs:** `data/processed/phones_clean.csv`, `excluded_rows.csv`, `cleaning_log.csv`.

**Limitations introduced:** model does not cover feature phones; RAM information is partially missing for premium
phones, so RAM's measured importance is probably understated.
