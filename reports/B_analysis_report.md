# B Analysis Report

## Overview

This report documents the exploratory data analysis, feature engineering, and model evaluation results for the crop yield prediction task.

## Data Summary

| Dataset | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| Training set | 15,090 | 15 | Plots with yield and agronomic attributes |
| Test set | 3,750 | 14 | Leaderboard test plots |
| Weather | 230 | 5 | Monthly climate by region |
| Market prices | 100 | 4 | Crop prices by region and year |

## Feature Engineering

### Weather-Derived Features (from `regional_weather.csv`)

Each plot's 4-month growing season (planting month + 3 months) was joined to the weather table on `(region, year, month)`:

| Feature | Description |
| --- | --- |
| `season_mean_temp` | Mean temperature across the growing window |
| `season_rainfall_total` | Cumulative seasonal rainfall |
| `season_rainfall_mean` | Mean seasonal rainfall |
| `season_extreme_heat_days` | Total days above heat threshold |
| `season_temp_std` | Temperature variability across the season |
| `missing_season_months` | Count of missing month rows in the join |

### Derived Agronomic Features

| Feature | Derivation |
| --- | --- |
| `fertilizer_per_labor_day` | `fertilizer_kg_per_ha / labor_days_per_ha` |
| `fertilizer_x_improved_seed` | `fertilizer_kg_per_ha * improved_seed_used` |
| `soil_x_fertilizer` | `soil_quality_index * fertilizer_kg_per_ha` |
| `rainfall_to_fertilizer` | `rainfall_mm_season / fertilizer_kg_per_ha` |
| `altitude_per_farm_size` | `altitude_m / farm_size_ha` |
| `planting_month_sin`, `planting_month_cos` | Sinusoidal encoding of planting month |

## Model Comparison

The following table ranks all 18 candidate models by validation RMSE:

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Stacking Ensemble | 0.4599 | 0.3368 | 0.8940 |
| XGBoost | 0.4693 | 0.3441 | 0.8897 |
| Hist Gradient Boosting | 0.4719 | 0.3429 | 0.8885 |
| LightGBM | 0.4770 | 0.3478 | 0.8860 |
| MLP Neural Net | 0.4844 | 0.3536 | 0.8825 |
| Gradient Boosting | 0.4962 | 0.3619 | 0.8767 |
| Extra Trees | 0.5303 | 0.3882 | 0.8591 |
| Random Forest | 0.5370 | 0.3948 | 0.8556 |
| SVR | 0.7299 | 0.5199 | 0.7331 |
| Decision Tree | 0.7452 | 0.5526 | 0.7219 |
| Linear Regression | 0.8896 | 0.6650 | 0.6036 |
| Ridge | 0.8896 | 0.6650 | 0.6036 |
| SGD | 0.8932 | 0.6688 | 0.6004 |
| Elastic Net | 0.8942 | 0.6651 | 0.5995 |
| Lasso | 0.8955 | 0.6654 | 0.5983 |
| KNN | 0.9337 | 0.6875 | 0.5633 |
| Mean Baseline | 1.4131 | 1.1042 | -0.0001 |

The Stacking Ensemble (RF + HGB + ET + XGB/LGB with Ridge meta-learner) was selected as the best model.

## Feature Importance

The top features from the final model are:

| Feature | Importance |
| --- | ---: |
| season_mean_temp | 0.2219 |
| altitude_m | 0.1770 |
| crop_type_maize | 0.1737 |
| soil_x_fertilizer | 0.0720 |
| season_extreme_heat_days | 0.0717 |
| pest_disease_flag | 0.0690 |
| fertilizer_x_improved_seed | 0.0390 |
| region_Somali | 0.0363 |
| crop_type_teff | 0.0282 |
| soil_quality_index | 0.0235 |

Weather-derived features (`season_mean_temp`, `season_extreme_heat_days`) are among the most important predictors, confirming the value of the climate join.

## Validation Metrics

| Metric | Value |
| --- | ---: |
| RMSE | 0.4599 |
| MAE | 0.3368 |
| R² | 0.8940 |
| 5-fold CV RMSE (mean) | 0.5576 |
| 5-fold CV RMSE (std) | 0.0046 |

## Key Findings

1. **Ensemble methods outperform single models** — The Stacking Ensemble achieves the best balance of bias and variance.
2. **Climate features matter** — Weather-derived features (`season_mean_temp`, `season_extreme_heat_days`) are top predictors, validating the seasonal climate join.
3. **Crop type is highly predictive** — One-hot encoded crop type (`maize`, `teff`) carries substantial signal.
4. **Altitude is important** — Elevation (`altitude_m`) is the second most important feature.
5. **Model stability** — Low 5-fold CV RMSE standard deviation (0.0046) indicates stable performance across data splits.