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
    """Vectorized weather feature engineering.

    For each plot, we compute 4-month growing-season statistics by joining
    the plot's planting month and survey year to the weather table. This
    replaces the previous O(n*m) iterrows() loop with a vectorized merge.
    """
    weather = weather_df.copy()
    weather = weather.dropna(subset=['region', 'year', 'month']).copy()
    weather['year'] = weather['year'].astype(int)
    weather['month'] = weather['month'].astype(int)

    # Build a season-key table: one row per (plot, month-in-season).
    # Each plot gets 4 rows (months 0..3 after planting month).
    plots = plot_df[['plot_id', 'region', 'survey_year', 'planting_month']].copy()
    plots['survey_year'] = plots['survey_year'].astype(int)
    plots['planting_month'] = plots['planting_month'].astype(int)

    # For each plot, generate the 4 (year, month) pairs in the season.
    season_pairs = []
    for delta in range(4):
        df = plots.copy()
        df['month'] = df['planting_month'] + delta
        df['year'] = df['survey_year']
        # Wrap year/month: month > 12 -> subtract 12, increment year
        df['year'] = df['year'] + (df['month'] - 1) // 12
        df['month'] = ((df['month'] - 1) % 12) + 1
        df['delta'] = delta
        season_pairs.append(df[['plot_id', 'region', 'year', 'month', 'delta']])
    season_long = pd.concat(season_pairs, ignore_index=True)

    # Merge with weather on (region, year, month).
    merged = season_long.merge(
        weather[['region', 'year', 'month', 'avg_temp_c', 'monthly_rainfall_mm', 'extreme_heat_days']],
        on=['region', 'year', 'month'],
        how='left'
    )
    merged['has_data'] = merged['avg_temp_c'].notna()

    # Aggregate per plot_id.
    agg = merged.groupby('plot_id').agg(
        season_mean_temp=('avg_temp_c', 'mean'),
        season_rainfall_total=('monthly_rainfall_mm', 'sum'),
        season_rainfall_mean=('monthly_rainfall_mm', 'mean'),
        season_extreme_heat_days=('extreme_heat_days', 'sum'),
        season_temp_std=('avg_temp_c', lambda s: s.std(ddof=0) if len(s) > 1 else 0.0),
        actual_months=('has_data', 'sum'),
    ).reset_index()
    agg['missing_season_months'] = 4 - agg['actual_months']
    agg = agg.drop(columns=['actual_months'])

    out = plot_df.merge(agg, on='plot_id', how='left')
    out['season_temp_std'] = out['season_temp_std'].fillna(0)
    out['season_extreme_heat_days'] = out['season_extreme_heat_days'].fillna(0)
    out['season_rainfall_total'] = out['season_rainfall_total'].fillna(0)
    out['season_rainfall_mean'] = out['season_rainfall_mean'].fillna(0)
    if 'avg_temp_c' in out.columns:
        out['season_mean_temp'] = out['season_mean_temp'].fillna(out['avg_temp_c'].median())
    else:
        out['season_mean_temp'] = out['season_mean_temp'].fillna(22)

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
