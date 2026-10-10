# Team 11 — Crop Yield Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-success.svg)](https://scikit-learn.org/)

End-to-end crop yield prediction pipeline for the hackathon task. The repository keeps raw data immutable, cleans and merges agronomic with climate and market-price data, engineers weather-derived seasonal features, compares and tunes candidate regression models, and exports the final leaderboard submission.

## Problem Statement

Predict crop yield (tons per hectare) from agronomic field attributes enriched with regional seasonal climate variables, while satisfying the competition rules:

- Keep all raw files unchanged (immutable)
- Clean and merge data into `data/processed`
- Derive at least one genuine feature from `regional_weather.csv`
- Keep `price_birr_per_quintal` out of the model features (price is used for economic analysis only)
- Fit only on the training data and evaluate with validation splits plus an out-of-time check by year

## Repository Layout

```
data/raw/          # Immutable competition inputs (never modified by the pipeline)
data/processed/    # Cleaned, merged master tables and data dictionary
src/
    cleaning.py    # Data cleaning, standardization, and issue logging
    features.py    # Weather-feature engineering and derived agronomic features
    train.py       # Training, validation, tuning, and artifact export
    predict.py     # Standalone inference pipeline for the submission
app/app.py         # Streamlit demo interface
figures/           # Exploratory and evaluation visualizations
models/            # Serialized model artifacts
reports/           # Cleaning log, model summary, comparison table
submission/        # Final leaderboard export
notebooks/         # Exploratory analysis and presentation
presentation/      # Slide deck
```

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

## Reproduce the Pipeline

```bash
# Train and generate all artifacts
python src/train.py

# Generate predictions for the leaderboard test set
python src/predict.py

# Run the Streamlit demo
streamlit run app/app.py
```

## Approach

### 1. Data Cleaning (`src/cleaning.py`)

Each input table is normalized independently and documented in a machine-readable cleaning log:

- **Region labels** — harmonized to canonical values (`Amhara`, `Oromia`, `SNNPR`, `Somali`, `Tigray`), resolving case and spacing drift between tables.
- **Crop labels** — canonicalized (`tef` → `teff`), with case and whitespace normalization.
- **Sentinel values** — `-999` sentinels treated as missing and imputed rather than assumed valid.
- **Numeric coercion** — fields converted to numeric with error coercion; impossible negatives (e.g. yield) clipped to zero.
- **Imputation strategy** — stratified medians chosen to respect biological and seasonal structure (rainfall by crop × region × year; fertilizer and soil by crop × region; temperature by region × month).
- **Missingness reporting** — every issue type, affected row count, percentage, fix, and justification is exported to `reports/A_cleaning_log.csv`.

### 2. Feature Engineering (`src/features.py`)

**Weather-derived growing-season features (from `regional_weather.csv`):**

| Feature | Description |
| --- | --- |
| `season_mean_temp` | Mean temperature across the 4-month growing window anchored on planting month |
| `season_rainfall_total` | Cumulative seasonal rainfall |
| `season_rainfall_mean` | Mean seasonal rainfall |
| `season_extreme_heat_days` | Total days above the heat threshold during the season |
| `season_temp_std` | Variability of temperatures across the season |
| `missing_season_months` | Count of missing month rows in the join (debugging aid) |

The growing season is computed as the planting month plus the next three months, with year wrapping, so each plot joins the correct seasonal climate history.

**Derived agronomic interactions:**

- `fertilizer_per_labor_day`, `fertilizer_x_improved_seed`, `soil_x_fertilizer`, `rainfall_to_fertilizer`, `altitude_per_farm_size`, plus sinusoidal encoding of `planting_month`.

### 3. Modeling (`src/train.py`)

- **Preprocessing** — median imputation + standard scaling for numeric columns; most-frequent imputation + one-hot encoding for `region` and `crop_type`.
- **Train/validation split** — 80/20 with `random_state=42`; all preprocessing fitted on the train fold only.
- **Candidate models** — 18 regression algorithms: mean baseline, Ridge, Linear Regression, Lasso, Elastic Net, SGD, KNN, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, Gradient Boosting, MLP, SVR, XGBoost, LightGBM, plus a Stacking Ensemble (RF + HGB + ET + XGB/LGB with Ridge meta-learner).
- **Ensemble + tuning** — Random Forest with RandomizedSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `min_samples_split`; 5-fold cross-validated RMSE and an out-of-time evaluation restricted to `survey_year <= 2023` (train) vs `> 2023` (validation) to guard against leakage across years.
- **Ablation** — model trained without any weather features to quantify the weather contribution.

### 4. Outputs

| Artifact | Description |
| --- | --- |
| `submission/team_11_submission.csv` | Leaderboard predictions (`plot_id`, `predicted_yield_tons_per_ha`) |
| `models/final_model.joblib` | Fitted preprocessing + model pipeline |
| `reports/A_cleaning_log.csv` | Record of every cleaning decision |
| `reports/D_model_summary.json` | Validation, CV, OOT, and ablation metrics plus best params |
| `reports/D_model_comparison.csv` | Rank-ordered candidate model table |
| `reports/validation_predictions.csv` | Validation actuals vs predictions and absolute errors |
| `reports/feature_importance.csv` | Global feature importances from the final model |
| `figures/*.png` | Missingness, cleaning impact, distributions, climate trends, model comparison, residuals, and feature importance plots |

## Results

On the held-out validation split, the fitted Stacking Ensemble regressor achieved:

| Metric | Value |
| --- | --- |
| RMSE | 0.4599 |
| MAE | 0.3368 |
| R² | 0.8940 |

The out-of-time evaluation (year-based holdout) reports RMSE, MAE, and R² in `reports/D_model_summary.json`. The weather-ablation run reports the same metrics without weather features, letting you verify that the climate variables improve predictions.

### Model comparison

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

The Stacking Ensemble was selected as the best model, combining the strengths of Random Forest, HistGradientBoosting, Extra Trees, and optionally XGBoost/LightGBM with a Ridge meta-learner.

## Data Dictionary

`data/processed/data_dictionary_master.csv` describes every column in the master table, its dtype, source, and derivation.

## Reproducibility

- Fixed random seeds everywhere (`random_state=42`); use the same versions of Python, pandas, NumPy, and scikit-learn.
- Raw inputs are copied to `data/raw/` at the start of `src/train.py` and never written back to the competition-supplied locations.
- Re-running `src/train.py` regenerates `data/processed/`, `figures/`, `models/`, and `reports/` deterministically.

## Team

- Oli Bakala Beyena, qiyas-2026-005295, Olibekele50@gmail.com
- Bereket G/Alif, qiyas-2026-004845, bereketgalif21@gmail.com
- Ermiyas Zewdu, qiyas-2026-004085, Ermiyaszewdu266@gmail.com
- Samuel Kahsay, qiyas-2026-001108, Samuelkahsay76@gmail.com
