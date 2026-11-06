# Day 5 — Feature Relevance Report

Notebook: `notebooks/05_feature_extraction.ipynb` · Data: training split only (419 rows, 29 encoded features).
Tables: `reports/feature_relevance.csv`, `reports/feature_set_comparison.csv`.

## Extraction vs selection
- **Extraction** (new features): processor → tier/vendor, model name → brand/5G/premium flag (Day 4); PCA (tested).
- **Selection** (subset of existing features): Pearson/Spearman, mutual information, variance threshold, F-test,
  SelectKBest, RFE/RFECV.

## Consensus ranking (mean rank over Spearman, MI, F-test, RF-RFE)
| Rank | Feature | Spearman ρ (log price) | MI | F |
|---|---|---|---|---|
| 1 | processor_tier | 0.859 | 0.780 | 1,105 |
| 2 | ram_gb | 0.796 | 0.723 | 551 |
| 3 | storage_gb | 0.814 | 0.740 | 453 |
| 4 | front_camera_mp | 0.789 | 0.714 | 232 |
| 5 | rear_camera_mp | 0.573 | 0.573 | 155 |
| 6 | is_5g | 0.597 | 0.326 | 155 |
| 7 | battery_mah | −0.458 | 0.324 | 75 |

## Key observations
- Against log price, Pearson ≈ Spearman → the log transform linearises the relationships.
- Display size: high MI (0.62) but low correlation (0.22). Exact sizes fingerprint phone families → memorisation risk.
- Variance threshold (0.01 on binary columns) removes nothing; rarest is brand_Apple (0.0118) — rare but important.
- RFECV (Ridge, grouped CV) keeps 22 of 29 features.
- PCA: PC1 = "premium-ness" (storage, RAM, tier, front camera, 5G; battery negative), 14.9 % of variance;
  90 % needs 17 components, 95 % needs 20. Not adopted (gain within noise, loses interpretability).

## Feature-set comparison (grouped 5-fold CV MAE, ₹)
| Feature set | Ridge | Gradient Boosting (default) |
|---|---|---|
| All features (29) | **2,408** | 2,810 |
| SelectKBest MI (k=8) | 3,098 | **2,583** |
| SelectKBest F (k=8) | 3,015 | 2,718 |
| RFE-RF (10) | 3,023 | 2,708 |
| RFE-Ridge (22) | 2,571 | 3,124 |
| PCA (20 comp.) | 2,466 | 2,762 |

Selection helps only the *untuned* boosting model (acts as regularisation). Day 8 shows that after tuning,
all features are best for boosting too (MAE ₹2,469 all vs ₹2,525–2,533 with 8–10 MI features).
**Decision: use all engineered features.**
