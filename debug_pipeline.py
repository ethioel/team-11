import pandas as pd
from pathlib import Path

from src.cleaning import clean_plot_data, clean_weather_data, clean_market_price
from src.features import add_weather_features, attach_price_analysis, build_feature_columns

root = Path(r'c:\Users\Sami\Documents\GitHub\team-beso')
train = pd.read_csv(root / 'data' / 'raw' / 'crop_yield_train.csv')
test = pd.read_csv(root / 'data' / 'raw' / 'crop_yield_leaderboard_test.csv')
weather = pd.read_csv(root / 'data' / 'raw' / 'regional_weather.csv')
price = pd.read_csv(root / 'data' / 'raw' / 'market_prices.csv')

train_clean, test_clean, log = clean_plot_data(train, test)
print('clean plot ok')
print(train_clean[['improved_seed_used', 'pest_disease_flag', 'labor_days_per_ha']].isna().sum())
weather_clean, wlog = clean_weather_data(weather)
print('clean weather ok')
print(weather_clean.isna().sum())
price_clean, plog = clean_market_price(price)
print('clean price ok')
print(price_clean.isna().sum())

train_feat = add_weather_features(train_clean, weather_clean)
test_feat = add_weather_features(test_clean, weather_clean)
train_feat['year'] = train_feat['survey_year']
test_feat['year'] = test_feat['survey_year']
print('weather features ok')
print(train_feat[['season_mean_temp', 'season_rainfall_total', 'fertilizer_per_labor_day']].isna().sum())
train_master = attach_price_analysis(train_feat, price_clean)
test_master = attach_price_analysis(test_feat, price_clean)
print('price attach ok')
print(train_master[['price_birr_per_quintal']].isna().sum())

for df in [train_master, test_master]:
    print('sample flags before conversion')
    print(df[['improved_seed_used', 'pest_disease_flag']].head())
    df['improved_seed_used'] = pd.to_numeric(df['improved_seed_used'], errors='coerce').fillna(0).astype(int)
    df['pest_disease_flag'] = pd.to_numeric(df['pest_disease_flag'], errors='coerce').fillna(0).astype(int)

feature_cols = build_feature_columns()
print('feature cols', len(feature_cols))
print('train shape', train_master.shape)
print('test shape', test_master.shape)

print('done')
