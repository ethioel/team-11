from __future__ import annotations

import pandas as pd
import numpy as np


MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _season_months(planting_month, survey_year):
    months = []
    for delta in range(4):
        month = planting_month + delta
        year = survey_year
        while month > 12:
            month -= 12
            year += 1
        months.append((year, month))
    return months


def add_weather_features(plot_df, weather_df):
    weather = weather_df.copy()
    weather = weather.dropna(subset=['region', 'year', 'month']).copy()
    weather['year'] = weather['year'].astype(int)
    weather['month'] = weather['month'].astype(int)
    results = []

    for _, row in plot_df.iterrows():
        region = row['region']
        year = int(row['survey_year'])
        month = int(row['planting_month'])
        season = _season_months(month, year)
        region_weather = weather[weather['region'] == region].copy()
        if region_weather.empty:
            mean_temp = np.nan
            rainfall_total = np.nan
            rainfall_mean = np.nan
            heat_days = np.nan
            temp_std = np.nan
            expected = len(season)
            actual = 0
        else:
            season_years = {y for y, _ in season}
            season_months = {m for _, m in season}
            season_rows = region_weather[
                region_weather['year'].isin(season_years) & region_weather['month'].isin(season_months)
            ].copy()
            expected = len(season)
            actual = len(season_rows)
            if actual == 0:
                mean_temp = np.nan
                rainfall_total = np.nan
                rainfall_mean = np.nan
                heat_days = np.nan
                temp_std = np.nan
            else:
                mean_temp = season_rows['avg_temp_c'].mean()
                rainfall_total = season_rows['monthly_rainfall_mm'].sum()
                rainfall_mean = season_rows['monthly_rainfall_mm'].mean()
                heat_days = season_rows['extreme_heat_days'].sum()
                temp_std = season_rows['avg_temp_c'].std(ddof=0)

        results.append({
            'plot_id': row['plot_id'],
            'season_mean_temp': mean_temp,
            'season_rainfall_total': rainfall_total,
            'season_rainfall_mean': rainfall_mean,
            'season_extreme_heat_days': heat_days,
            'season_temp_std': temp_std,
            'missing_season_months': expected - actual,
        })

    features = pd.DataFrame(results)
    out = plot_df.merge(features, on='plot_id', how='left')
    out['season_temp_std'] = out['season_temp_std'].fillna(0)
    out['season_extreme_heat_days'] = out['season_extreme_heat_days'].fillna(0)
    out['season_rainfall_total'] = out['season_rainfall_total'].fillna(0)
    out['season_rainfall_mean'] = out['season_rainfall_mean'].fillna(0)
    out['season_mean_temp'] = out['season_mean_temp'].fillna(out['avg_temp_c'].median()) if 'avg_temp_c' in out.columns else out['season_mean_temp'].fillna(22)

    out['fertilizer_per_labor_day'] = out['fertilizer_kg_per_ha'] / (out['labor_days_per_ha'].replace(0, np.nan))
    out['fertilizer_per_labor_day'] = out['fertilizer_per_labor_day'].fillna(0)
    out['fertilizer_x_improved_seed'] = out['fertilizer_kg_per_ha'] * out['improved_seed_used']
    out['soil_x_fertilizer'] = out['soil_quality_index'] * out['fertilizer_kg_per_ha']
    out['planting_month_sin'] = np.sin(2 * np.pi * out['planting_month'] / 12)
    out['planting_month_cos'] = np.cos(2 * np.pi * out['planting_month'] / 12)
    out['rainfall_to_fertilizer'] = out['rainfall_mm_season'] / out['fertilizer_kg_per_ha'].replace(0, np.nan)
    out['rainfall_to_fertilizer'] = out['rainfall_to_fertilizer'].fillna(0)
    out['altitude_per_farm_size'] = out['altitude_m'] / out['farm_size_ha'].replace(0, np.nan)
    out['altitude_per_farm_size'] = out['altitude_per_farm_size'].fillna(0)
    return out


def attach_price_analysis(df, price_df):
    merged = df.merge(price_df[['crop_type', 'region', 'year', 'price_birr_per_quintal']],
                      on=['crop_type', 'region', 'year'],
                      how='left',
                      suffixes=('', '_price'))
    if 'yield_tons_per_ha' in merged.columns:
        merged['revenue_per_ha_birr'] = merged['yield_tons_per_ha'] * 10 * merged['price_birr_per_quintal']
    else:
        merged['revenue_per_ha_birr'] = 0.0
    merged['revenue_per_ha_birr'] = merged['revenue_per_ha_birr'].fillna(0)
    return merged


def build_feature_columns():
    numeric = [
        'survey_year', 'planting_month', 'altitude_m', 'rainfall_mm_season', 'farm_size_ha',
        'fertilizer_kg_per_ha', 'soil_quality_index', 'labor_days_per_ha',
        'distance_to_market_km', 'season_mean_temp', 'season_rainfall_total',
        'season_rainfall_mean', 'season_extreme_heat_days', 'season_temp_std',
        'fertilizer_per_labor_day', 'fertilizer_x_improved_seed', 'soil_x_fertilizer',
        'planting_month_sin', 'planting_month_cos', 'rainfall_to_fertilizer',
        'altitude_per_farm_size'
    ]
    categorical = ['region', 'crop_type', 'improved_seed_used', 'pest_disease_flag']
    return numeric + categorical


def make_data_dictionary(df):
    rows = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        rows.append({
            'column': c,
            'type': dtype,
            'source': 'processed feature table',
            'description': c,
            'derivation': 'engineered or source feature'
        })
    return pd.DataFrame(rows)
