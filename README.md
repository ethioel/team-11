# Team 11 Crop Yield Challenge

This repository reproduces the end-to-end crop-yield prediction pipeline for the hackathon task. The project keeps raw data immutable, cleans and consolidates it into processed tables, engineers weather-derived features, compares candidate models, and exports the final submission.

## Project goal

Predict crop yield (tons/ha) using agronomic and climate features while obeying the competition rules:

- keep raw files unchanged
- clean into `data/processed`
- derive at least one genuine feature from `regional_weather.csv`
- keep `price_birr_per_quintal` out of the model features
- fit only on train data and evaluate with validation splits

## Repository layout

- `data/raw`: immutable source files
- `data/processed`: cleaned and merged dataset exports
- `src/cleaning.py`: data cleaning and standardization
- `src/features.py`: engineered weather and agronomic features
- `src/train.py`: training, validation, model comparison, and artifact export
- `src/predict.py`: prediction pipeline for saved model
- `app/app.py`: Streamlit demo
- `models/`: saved model artifacts
- `reports/`: cleaning log, summary JSON, comparison CSV
- `figures/`: model and exploratory visualizations
- `submission/`: final prediction export
- `presentation/`: pptx slide deck
- `notebooks/`: analysis notebook

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

## Reproduce the pipeline

```bash
python src/train.py
python src/predict.py
streamlit run app/app.py
```

## Key outputs

- `models/final_model.joblib`
- `submission/team_11_submission.csv`
- `reports/D_model_summary.json`
- `reports/D_model_comparison.csv`
- `reports/A_cleaning_log.csv`

## Model Analytics & Performance

The final model is a RandomForest regressor trained on cleaned agronomic features plus weather-derived seasonal climate variables from `regional_weather.csv`.

### Validation metrics

On the held-out validation split:

- RMSE: 0.5376
- MAE: 0.3950
- R²: 0.8553

This indicates the model explains about 85.5% of the variance in crop yield, while maintaining low absolute error.

### Model comparison

The average validation performance across candidate models was:

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Mean baseline | 1.4131 | 1.1042 | -0.0001 |
| Linear regression | 0.8896 | 0.6650 | 0.6036 |
| Random forest | 0.5376 | 0.3950 | 0.8553 |

The RandomForest model was selected because it gave the best balance of predictive accuracy and stability across validation checks.

### Supporting artifacts

- `reports/D_model_summary.json`
- `reports/D_model_comparison.csv`
- `figures/fig10_model_comparison.png`
- `figures/fig11_predicted_vs_actual_residuals.png`
- `figures/fig12_feature_importance.png`

## Result

The selected model is a RandomForest regressor trained on cleaned agronomic features and weather-derived seasonal climate variables. Final validation achieved strong performance with low RMSE and high R-squared on the held-out validation split.

## Group Members

- Oli Bakala Beyena, qiyas-2026-005295, Olibekele50@gmail.com

- Bereket G/Alif, qiyas-2026-004845,  bereketgalif21@gmail.com

- Ermiyas Zewdu, qiyas-2026-004085, Ermiyaszewdu266@gmail.com

- Samuel Kahsay, qiyas-2026-001108, Samuelkahsay76@gmail.com
