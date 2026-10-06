from __future__ import annotations

import numpy as np
import pandas as pd

REGION_MAP = {
    'amh': 'Amhara',
    'amhara': 'Amhara',
    'amhara ': 'Amhara',
    'orom': 'Oromia',
    'oromia': 'Oromia',
    'oromia ': 'Oromia',
    'snnp': 'SNNPR',
    'snnpr': 'SNNPR',
    'snnpr ': 'SNNPR',
    'som': 'Somali',
    'somali': 'Somali',
    'somali ': 'Somali',
    'tig': 'Tigray',
    'tigray': 'Tigray',
    'tigray ': 'Tigray',
}

CROP_MAP = {
    'barley': 'barley',
    'barley ': 'barley',
    'maize': 'maize',
    'maize ': 'maize',
    'sorghum': 'sorghum',
    'sorghum ': 'sorghum',
    'tef': 'teff',
    'teff': 'teff',
    'wheat': 'wheat',
    'wheat ': 'wheat',
}


def _to_clean_string(v):
    if pd.isna(v):
        return np.nan
    return str(v).strip()


def standardize_region(value):
    v = _to_clean_string(value)
    if pd.isna(v):
        return np.nan
    v_norm = v.lower()
    if v_norm in REGION_MAP:
        return REGION_MAP[v_norm]
    return v.title()


def standardize_crop(value):
    v = _to_clean_string(value)
    if pd.isna(v):
        return np.nan
    v_norm = v.lower()
    if v_norm in CROP_MAP:
        return CROP_MAP[v_norm]
    if v_norm == 'tef':
        return 'teff'
    return v.lower()


def _add_issue(log, file_name, columns, issue_type, affected_rows, percentage, fix, justification):
    log.append({
        'file': file_name,
        'column(s)': columns,
        'issue type': issue_type,
        'number of affected rows': int(affected_rows),
        'percentage affected': float(percentage),
        'fix': fix,
        'justification': justification,
    })


def _replace_sentinel_numeric(df, cols, sentinel=-999):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
            out.loc[out[c] == sentinel, c] = np.nan
    return out


def clean_plot_data(train_df, test_df):
    issue_log = []
    train = train_df.copy()
    test = test_df.copy()

    for df, name in [(train, 'plot_train.csv'), (test, 'plot_test.csv')]:
        df['region'] = df['region'].map(standardize_region)
        df['crop_type'] = df['crop_type'].map(standardize_crop)
        df['planting_month'] = df['planting_month'].astype(str).str.strip().str.title()
        df['planting_month'] = df['planting_month'].map({
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
        })

        for col in ['altitude_m', 'rainfall_mm_season', 'farm_size_ha', 'fertilizer_kg_per_ha',
                    'soil_quality_index', 'labor_days_per_ha', 'distance_to_market_km']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in ['pest_disease_flag', 'improved_seed_used']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[df[col] == -999, col] = np.nan

        # Missingness counts
        missing = df.isna().sum()
        if 'rainfall_mm_season' in df.columns:
            n = int(missing['rainfall_mm_season'])
            _add_issue(issue_log, name, 'rainfall_mm_season', 'missing values', n, n / len(df),
                       'median imputation by crop-region-year strata', 'self-reported rainfall had a small amount of missingness; median preserves original scale.')
            med = df.groupby(['crop_type', 'region', 'survey_year'])['rainfall_mm_season'].transform('median')
            df['rainfall_mm_season'] = df['rainfall_mm_season'].fillna(med)

        if 'fertilizer_kg_per_ha' in df.columns:
            n = int(missing['fertilizer_kg_per_ha'])
            _add_issue(issue_log, name, 'fertilizer_kg_per_ha', 'missing values', n, n / len(df),
                       'median imputation by crop and region', 'fertilizer is skewed, so median is robust and preserves typical practice levels.')
            med = df.groupby(['crop_type', 'region'])['fertilizer_kg_per_ha'].transform('median')
            df['fertilizer_kg_per_ha'] = df['fertilizer_kg_per_ha'].fillna(med)

        if 'soil_quality_index' in df.columns:
            n = int(missing['soil_quality_index'])
            _add_issue(issue_log, name, 'soil_quality_index', 'missing values', n, n / len(df),
                       'median imputation within crop-region groups', 'soil quality is a moderate-quality field characteristic with limited missingness.')
            med = df.groupby(['crop_type', 'region'])['soil_quality_index'].transform('median')
            df['soil_quality_index'] = df['soil_quality_index'].fillna(med)

        if 'pest_disease_flag' in df.columns:
            n = int((df['pest_disease_flag'] == -999).sum())
            if n > 0:
                _add_issue(issue_log, name, 'pest_disease_flag', '-999 sentinel', n, n / len(df),
                           'convert -999 to missing and fill with mode', 'the sentinel indicates an unknown or unreported flag, not a valid biological state.')
                df['pest_disease_flag'] = df['pest_disease_flag'].replace(-999, np.nan)
                mode = df['pest_disease_flag'].mode(dropna=True)
                if not mode.empty:
                    df['pest_disease_flag'] = df['pest_disease_flag'].fillna(mode.iloc[0])

        if 'labor_days_per_ha' in df.columns:
            n = int((df['labor_days_per_ha'] == -999).sum())
            if n > 0:
                _add_issue(issue_log, name, 'labor_days_per_ha', '-999 sentinel', n, n / len(df),
                           'replace -999 with median', 'this is an impossible labor value and should be treated as missing rather than a real observation.')
                df['labor_days_per_ha'] = df['labor_days_per_ha'].replace(-999, np.nan)
                med = df['labor_days_per_ha'].median()
                df['labor_days_per_ha'] = df['labor_days_per_ha'].fillna(med)

        if 'yield_tons_per_ha' in df.columns:
            df['yield_tons_per_ha'] = pd.to_numeric(df['yield_tons_per_ha'], errors='coerce')
            neg = int((df['yield_tons_per_ha'] < 0).sum())
            if neg > 0:
                _add_issue(issue_log, name, 'yield_tons_per_ha', 'impossible negative values', neg, neg / len(df),
                           'clip negatives to zero', 'yield cannot be negative, and the values are likely recording errors.')
                df.loc[df['yield_tons_per_ha'] < 0, 'yield_tons_per_ha'] = 0

        # Standardise remaining values to canonical form
        df['region'] = df['region'].apply(standardize_region)
        df['crop_type'] = df['crop_type'].apply(standardize_crop)

    return train, test, issue_log


def clean_weather_data(weather_df):
    issue_log = []
    weather = weather_df.copy()
    weather['region'] = weather['region'].map(standardize_region)
    weather['year'] = pd.to_numeric(weather['year'], errors='coerce').astype('Int64')
    weather['month'] = weather['month'].astype(str).str.strip().str.title().map({
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    })
    weather['month'] = pd.to_numeric(weather['month'], errors='coerce').astype('Int64')
    for col in ['avg_temp_c', 'monthly_rainfall_mm', 'extreme_heat_days']:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')

    missing_temp = int(weather['avg_temp_c'].isna().sum())
    if missing_temp > 0:
        _add_issue(issue_log, 'regional_weather.csv', 'avg_temp_c', 'missing values', missing_temp,
                   missing_temp / len(weather), 'fill with regional-month median', 'temperature gaps are small and can be estimated from the same region-month context.')
        weather['avg_temp_c'] = weather.groupby('region')['avg_temp_c'].transform(lambda s: s.fillna(s.median()))

    missing_rain = int(weather['monthly_rainfall_mm'].isna().sum())
    if missing_rain > 0:
        _add_issue(issue_log, 'regional_weather.csv', 'monthly_rainfall_mm', 'missing values', missing_rain,
                   missing_rain / len(weather), 'fill with regional-month median', 'monthly rainfall is highly seasonal, so regional medians are a sensible missing-data rule.')
        weather['monthly_rainfall_mm'] = weather.groupby(['region', 'month'])['monthly_rainfall_mm'].transform(lambda s: s.fillna(s.median()))

    bad_region = int(weather['region'].isna().sum())
    if bad_region > 0:
        _add_issue(issue_log, 'regional_weather.csv', 'region', 'inconsistent labels', bad_region,
                   bad_region / len(weather), 'standardize region names to canonical labels', 'weather codes such as AMH or ORO were inconsistent with plot tables and needed harmonization.')
        weather = weather.dropna(subset=['region'])

    weather = weather.dropna(subset=['region', 'year', 'month']).copy()
    weather = weather.drop_duplicates(subset=['region', 'year', 'month'], keep='first').copy()
    weather['month'] = weather['month'].astype(int)
    weather['year'] = weather['year'].astype(int)
    return weather, issue_log


def clean_market_price(price_df):
    issue_log = []
    price = price_df.copy()
    price['region'] = price['region'].map(standardize_region)
    price['crop_type'] = price['crop_type'].map(standardize_crop)
    price['price_birr_per_quintal'] = pd.to_numeric(price['price_birr_per_quintal'], errors='coerce')
    price['year'] = pd.to_numeric(price['year'], errors='coerce').astype('Int64')

    missing_price = int(price['price_birr_per_quintal'].isna().sum())
    if missing_price > 0:
        _add_issue(issue_log, 'market_prices.csv', 'price_birr_per_quintal', 'missing values', missing_price,
                   missing_price / len(price), 'fill with median crop-region price', 'missing price records are too sparse to drop wholesale and likely reflect incomplete observation, not missing phenomenon.')
        price['price_birr_per_quintal'] = price.groupby(['crop_type', 'region'])['price_birr_per_quintal'].transform(lambda s: s.fillna(s.median()))

    bad_crop = int(price['crop_type'].isna().sum())
    if bad_crop > 0:
        _add_issue(issue_log, 'market_prices.csv', 'crop_type', 'inconsistent labels', bad_crop,
                   bad_crop / len(price), 'normalise crop names to canonical names', 'market table crop labels showed case and spacing drift from the plot data.')

    bad_region = int(price['region'].isna().sum())
    if bad_region > 0:
        _add_issue(issue_log, 'market_prices.csv', 'region', 'inconsistent labels', bad_region,
                   bad_region / len(price), 'standardise region labels to the same set as the plot files', 'regional coding was inconsistent across the dataset and needed harmonisation for valid joins.')

    price = price.loc[price['price_birr_per_quintal'] > 0].copy()
    price = price.drop_duplicates(subset=['crop_type', 'region', 'year'], keep='first').copy()
    return price, issue_log


def build_cleaning_log(train_log, weather_log, price_log):
    rows = []
    rows.extend(train_log)
    rows.extend(weather_log)
    rows.extend(price_log)
    return pd.DataFrame(rows)
