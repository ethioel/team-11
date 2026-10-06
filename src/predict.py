from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from cleaning import clean_market_price, clean_plot_data, clean_weather_data, standardize_crop, standardize_region
from features import add_weather_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_PATH = ROOT / 'models' / 'final_model.joblib'


def load_inputs():
    train_df = pd.read_csv(DATA_DIR / 'crop_yield_train.csv')
    test_df = pd.read_csv(DATA_DIR / 'crop_yield_leaderboard_test.csv')
    weather_df = pd.read_csv(RAW_DIR / 'regional_weather.csv')
    price_df = pd.read_csv(RAW_DIR / 'market_prices.csv')
    clean_train, clean_test, _ = clean_plot_data(train_df, test_df)
    weather_clean, _ = clean_weather_data(weather_df)
    price_clean, _ = clean_market_price(price_df)
    train_master = add_weather_features(clean_train, weather_clean)
    test_master = add_weather_features(clean_test, weather_clean)
    return train_master, test_master, price_clean


def predict_submission():
    model = joblib.load(MODEL_PATH)
    _, test_df, _ = load_inputs()
    feature_cols = [
        'survey_year', 'planting_month', 'altitude_m', 'rainfall_mm_season', 'farm_size_ha',
        'fertilizer_kg_per_ha', 'soil_quality_index', 'labor_days_per_ha', 'distance_to_market_km',
        'season_mean_temp', 'season_rainfall_total', 'season_rainfall_mean', 'season_extreme_heat_days',
        'season_temp_std', 'fertilizer_per_labor_day', 'fertilizer_x_improved_seed', 'soil_x_fertilizer',
        'planting_month_sin', 'planting_month_cos', 'rainfall_to_fertilizer', 'altitude_per_farm_size',
        'region', 'crop_type', 'improved_seed_used', 'pest_disease_flag'
    ]
    X = test_df[feature_cols].copy()
    pred = model.predict(X)
    out = pd.DataFrame({'plot_id': test_df['plot_id'], 'predicted_yield_tons_per_ha': pred})
    out.to_csv(ROOT / 'submission' / 'team_beso_submission.csv', index=False)
    print(out.head())
    print('rows', len(out))


if __name__ == '__main__':
    predict_submission()
