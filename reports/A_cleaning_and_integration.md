# A Cleaning and Integration Report

## Overview

This report documents the data cleaning, standardization, and integration process applied to the competition inputs. All raw files remain immutable; cleaned and merged outputs are stored under `data/processed/`.

## Source Data

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `crop_yield_train.csv` | 15,090 | 15 | Training plots with yield and agronomic attributes |
| `crop_yield_leaderboard_test.csv` | 3,750 | 14 | Leaderboard test plots (no yield column) |
| `regional_weather.csv` | 230 | 5 | Monthly climate by region |
| `market_prices.csv` | 100 | 4 | Crop prices by region and year |

## Cleaning Log

The following issues were detected and addressed during cleaning. The complete machine-readable log is stored in `reports/A_cleaning_log.csv`.

| File | Column(s) | Issue Type | Affected Rows | Percentage | Fix | Justification |
| --- | --- | --- | ---: | ---: | --- | --- |
| plot_train.csv | rainfall_mm_season | Missing values | 449 | 2.98% | Median imputation by crop-region-year strata | Self-reported rainfall had a small amount of missingness; median preserves original scale. |
| plot_train.csv | fertilizer_kg_per_ha | Missing values | 748 | 4.96% | Median imputation by crop and region | Fertilizer is skewed, so median is robust and preserves typical practice levels. |
| plot_train.csv | soil_quality_index | Missing values | 608 | 4.03% | Median imputation within crop-region groups | Soil quality is a moderate-quality field characteristic with limited missingness. |
| plot_train.csv | labor_days_per_ha | -999 sentinel | 488 | 3.23% | Replace -999 with median | This is an impossible labor value and should be treated as missing rather than a real observation. |
| plot_test.csv | rainfall_mm_season | Missing values | 125 | 3.33% | Median imputation by crop-region-year strata | Self-reported rainfall had a small amount of missingness; median preserves original scale. |
| plot_test.csv | fertilizer_kg_per_ha | Missing values | 189 | 5.04% | Median imputation by crop and region | Fertilizer is skewed, so median is robust and preserves typical practice levels. |
| plot_test.csv | soil_quality_index | Missing values | 125 | 3.33% | Median imputation within crop-region groups | Soil quality is a moderate-quality field characteristic with limited missingness. |
| plot_test.csv | labor_days_per_ha | -999 sentinel | 100 | 2.67% | Replace -999 with median | This is an impossible labor value and should be treated as missing rather than a real observation. |
| regional_weather.csv | avg_temp_c | Missing values | 1 | 0.43% | Fill with regional-month median | Temperature gaps are small and can be estimated from the same region-month context. |
| regional_weather.csv | monthly_rainfall_mm | Missing values | 2 | 0.86% | Fill with regional-month median | Monthly rainfall is highly seasonal, so regional medians are a sensible missing-data rule. |
| market_prices.csv | price_birr_per_quintal | Missing values | 4 | 4.00% | Fill with median crop-region price | Missing price records are too sparse to drop wholesale and likely reflect incomplete observation, not missing phenomenon. |

## Standardization Rules

### Region Labels
All region labels were harmonized to canonical values:
- `amh`, `amhara`, `amhara ` → `Amhara`
- `orom`, `oromia`, `oromia ` → `Oromia`
- `snnp`, `snnpr`, `snnpr ` → `SNNPR`
- `som`, `somali`, `somali ` → `Somali`
- `tig`, `tigray`, `tigray ` → `Tigray`

### Crop Labels
Crop labels were canonicalized:
- `tef` → `teff`
- All labels lowercased and whitespace stripped

### Sentinel Values
- `-999` sentinels were treated as missing and imputed (not assumed valid)
- Negative yield values were clipped to zero

## Integration

The cleaned tables were merged as follows:

1. **Weather features** — Each plot's 4-month growing season (planting month + 3 months) was joined to the weather table on `(region, year, month)`. This produced 6 seasonal climate features: `season_mean_temp`, `season_rainfall_total`, `season_rainfall_mean`, `season_extreme_heat_days`, `season_temp_std`, and `missing_season_months`.

2. **Market price** — Price data was attached by `(crop_type, region, year)` for economic analysis only. The `price_birr_per_quintal` column was excluded from model features per competition rules.

3. **Feature engineering** — Derived features include `fertilizer_per_labor_day`, `fertilizer_x_improved_seed`, `soil_x_fertilizer`, `rainfall_to_fertilizer`, `altitude_per_farm_size`, and sinusoidal encoding of `planting_month`.

## Outputs

| File | Description |
| --- | --- |
| `data/processed/master_train.csv` | Cleaned, merged training feature table |
| `data/processed/master_test.csv` | Cleaned, merged test feature table |
| `data/processed/data_dictionary_master.csv` | Data dictionary for the master table |
| `reports/A_cleaning_log.csv` | Machine-readable cleaning log |