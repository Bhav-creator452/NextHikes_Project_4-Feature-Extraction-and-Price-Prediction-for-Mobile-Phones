# Data Dictionary — `Processed_Flipdata.csv`

**Source:** NextHikes Project 4 assignment (Flipkart India listings).
**Size:** 541 rows × 12 columns. **Unit of observation:** one product listing variant (model × colour × storage/RAM).
**Verified:** Day 1, `notebooks/01_data_exploration.ipynb`.

| # | Raw column | Raw type | Meaning | Unit | Range / values | Quality notes | Planned treatment |
|---|---|---|---|---|---|---|---|
| 1 | `Unnamed: 0` | int | Leftover row index from an earlier export | — | 0–642 (102 numbers missing) | Not a feature | Drop |
| 2 | `Model` | text | Brand + model name | — | 187 raw / ~181 case-normalised values | 6 case variants; garbled feature-phone names (`redmi min/max/mono`, `micromax1`); one mislabelled row (“Pixel 6a” with Nothing Phone 1 specs) | Normalise; extract brand; group key for validation |
| 3 | `Colour` | text | Marketing colour name | — | 275 values | 4 case variants; `M3)` fragment (5 rows); copied from previous row on feature phones | Probably excluded as a feature |
| 4 | `Memory` | int | Internal storage | GB | 16, 32, 64, 128, 256 | Unreliable on ~16 feature-phone rows (copied from previous row) | Numeric (ordinal) |
| 5 | `RAM` | int | RAM | GB | 2, 3, 4, 6, 8 | No 12 GB; ~10 premium 256 GB rows suspected truncated; iPhone values inconsistent; feature-phone rows copied | Numeric; flag suspect rows |
| 6 | `Battery_` | int | Battery capacity | mAh | 800–7000 (median 5000) | Values < 2600 are feature phones only | Numeric |
| 7 | `Rear Camera` | text | Primary rear camera | MP | 0–200 (`"50MP"` format) | `0MP` = no camera (10 rows) | Parse to int |
| 8 | `Front Camera` | text | Front camera | MP | 0–60 | `0MP` = no camera (16 rows) | Parse to int |
| 9 | `AI Lens` | int | Listing mentions an AI lens | 0/1 | 1 on 34 rows | Only Infinix phones → possible brand proxy | Binary |
| 10 | `Mobile Height` | float | **Display size (diagonal)** as listed — despite the name | cm | 4.5–17.65 (+ two errors) | 41.94 on two MOTOROLA G62 5G rows; same model elsewhere = 16.64 | Numeric; correct error; rename |
| 11 | `Processor_` | text | Chipset description | — | 123 raw / 113 normalised | Many spellings per chip; ~25 rows give OS/core count/vendor only; 1 mis-encoded; 18 trailing spaces | Engineer vendor / family / tier |
| 12 | `Prize` | text | **Target — selling price** | INR (₹) | 920–80,999 (median 13,499) | Comma-formatted string; all 541 parse cleanly; skew ≈ 2.43 | Convert to int; rename `price` |

## Proposed column renames (applied on Day 3)

| Raw | Clean |
|---|---|
| `Model` | `model` |
| `Colour` | `colour` |
| `Memory` | `storage_gb` |
| `RAM` | `ram_gb` |
| `Battery_` | `battery_mah` |
| `Rear Camera` | `rear_camera_mp` |
| `Front Camera` | `front_camera_mp` |
| `AI Lens` | `ai_lens` |
| `Mobile Height` | `display_size_cm` |
| `Processor_` | `processor` |
| `Prize` | `price` |
