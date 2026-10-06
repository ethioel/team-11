# Team 11 Crop Yield Challenge

This project delivers an end-to-end crop-yield prediction pipeline for the hackathon task. It keeps the raw data immutable, builds a cleaned and integrated analysis dataset, engineers meaningful climate features from `regional_weather.csv`, compares candidate models, and exports a final submission that follows the competition rules.

## Objective

Predict crop yield in tons per hectare using agronomic and climate signals while remaining fully compliant with the challenge constraints:

- keep all raw files unchanged
- write cleaned and merged outputs into `data/processed`
- derive at least one genuine feature from `regional_weather.csv`
- exclude `price_birr_per_quintal` from the model feature set
- fit only on training data and validate using held-out performance checks

## Repository structure

- `data/raw`: immutable source files
- `data/processed`: cleaned and integrated datasets
- `src/cleaning.py`: standardization, validation, and missing-data handling
- `src/features.py`: feature engineering and data dictionary generation
- `src/train.py`: training pipeline, validation, tuning, and artifact export
- `src/predict.py`: reusable prediction workflow for saved model outputs
- `app/app.py`: Streamlit dashboard for interactive yield and revenue exploration
- `models/`: trained model artifacts
- `reports/`: model summary, comparison metrics, and cleaning log
- `figures/`: figures and visual analysis pack
- `submission/`: final prediction export
- `presentation/`: five-slide project presentation
- `notebooks/`: analysis and modeling notebooks

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

## Reproduction workflow

```bash
python src/train.py
python src/predict.py
streamlit run app/app.py
```

## Key outputs

- `models/final_model.joblib`
- `submission/team_beso_submission.csv`
- `reports/D_model_summary.json`
- `reports/D_model_comparison.csv`
- `reports/A_cleaning_log.csv`
- `reports/project_summary.pdf`

## Model analytics and performance

The final selected model is a RandomForest regressor trained on cleaned agronomic features and weather-derived seasonal climate variables from `regional_weather.csv`.

### Validation performance

On the held-out validation split:

- RMSE: 0.5376
- MAE: 0.3950
- R²: 0.8553

This means the model explains approximately 85.5% of the variation in crop yield while keeping prediction error low and stable.

### Model comparison

The candidate model comparison showed the following validation performance:

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Mean baseline | 1.4131 | 1.1042 | -0.0001 |
| Linear regression | 0.8896 | 0.6650 | 0.6036 |
| Random forest | 0.5376 | 0.3950 | 0.8553 |

The RandomForest model was selected because it delivered the strongest predictive quality and the best balance of accuracy and robustness.

### Supporting evidence

- `reports/D_model_summary.json`
- `reports/D_model_comparison.csv`
- `figures/fig10_model_comparison.png`
- `figures/fig11_predicted_vs_actual_residuals.png`
- `figures/fig12_feature_importance.png`

## Final outcome

The final solution combines rule-compliant data handling, weather-aware feature engineering, and a strong predictive model. It produces a high-quality yield forecast while preserving the integrity constraints of the challenge and creating a clear, reproducible project package for review and presentation.
