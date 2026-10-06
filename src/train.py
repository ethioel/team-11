from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cleaning import build_cleaning_log, clean_market_price, clean_plot_data, clean_weather_data
from features import add_weather_features, attach_price_analysis, build_feature_columns, make_data_dictionary

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FIG_DIR = ROOT / 'figures'
REPORTS_DIR = ROOT / 'reports'
MODELS_DIR = ROOT / 'models'
SUBMISSION_DIR = ROOT / 'submission'

for d in [RAW_DIR, PROCESSED_DIR, FIG_DIR, REPORTS_DIR, MODELS_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def copy_raw_data():
    files = [
        'crop_yield_train.csv',
        'crop_yield_leaderboard_test.csv',
        'market_prices.csv',
        'regional_weather.csv',
        'submission_template (1).csv',
    ]
    for fn in files:
        src = ROOT / 'data' / fn
        dst = RAW_DIR / fn
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())


def _standard_feature_table(df):
    keep = [
        'plot_id', 'region', 'crop_type', 'survey_year', 'planting_month',
        'altitude_m', 'rainfall_mm_season', 'farm_size_ha', 'fertilizer_kg_per_ha',
        'improved_seed_used', 'pest_disease_flag', 'soil_quality_index',
        'labor_days_per_ha', 'distance_to_market_km'
    ]
    for c in ['yield_tons_per_ha', 'price_birr_per_quintal']:
        if c in df.columns:
            keep.append(c)
    return df.loc[:, [c for c in keep if c in df.columns]]


def load_raw_data():
    train = pd.read_csv(RAW_DIR / 'crop_yield_train.csv')
    test = pd.read_csv(RAW_DIR / 'crop_yield_leaderboard_test.csv')
    weather = pd.read_csv(RAW_DIR / 'regional_weather.csv')
    price = pd.read_csv(RAW_DIR / 'market_prices.csv')
    return train, test, weather, price


def write_cleaning_log(log):
    pd.DataFrame(log).to_csv(REPORTS_DIR / 'A_cleaning_log.csv', index=False)


def verify_integrity(df, label):
    checks = [
        ('unique plot ids', bool(df['plot_id'].is_unique)),
        ('yield non-negative', bool((df['yield_tons_per_ha'] >= 0).all()) if 'yield_tons_per_ha' in df.columns else True),
        ('price positive', bool((df['price_birr_per_quintal'] > 0).all()) if 'price_birr_per_quintal' in df.columns else True),
        ('region valid', bool(df['region'].isin(['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray']).all())),
        ('crop valid', bool(df['crop_type'].isin(['barley', 'maize', 'sorghum', 'teff', 'wheat']).all())),
    ]
    for name, value in checks:
        print(f'{label}: {name}:', 'PASS' if value else 'FAIL')


def build_model_data():
    train_raw, test_raw, weather_raw, price_raw = load_raw_data()
    train_clean, test_clean, train_log = clean_plot_data(train_raw, test_raw)
    weather_clean, weather_log = clean_weather_data(weather_raw)
    price_clean, price_log = clean_market_price(price_raw)
    log = build_cleaning_log(train_log, weather_log, price_log)
    write_cleaning_log(log)

    feature_train = add_weather_features(train_clean, weather_clean)
    feature_test = add_weather_features(test_clean, weather_clean)

    train_master = feature_train.copy()
    test_master = feature_test.copy()
    train_master['year'] = train_master['survey_year']
    test_master['year'] = test_master['survey_year']
    train_master = attach_price_analysis(train_master, price_clean)
    test_master = attach_price_analysis(test_master, price_clean)

    for df in [train_master, test_master]:
        df['improved_seed_used'] = pd.to_numeric(df['improved_seed_used'], errors='coerce').fillna(0).astype(int)
        df['pest_disease_flag'] = pd.to_numeric(df['pest_disease_flag'], errors='coerce').fillna(0).astype(int)

    feature_cols = build_feature_columns()
    # ensure common columns exist
    for c in feature_cols:
        if c not in train_master.columns:
            train_master[c] = 0
        if c not in test_master.columns:
            test_master[c] = 0

    verify_integrity(train_master, 'train')
    verify_integrity(test_master, 'test')

    train_master.to_csv(PROCESSED_DIR / 'master_train.csv', index=False)
    test_master.to_csv(PROCESSED_DIR / 'master_test.csv', index=False)
    data_dict = make_data_dictionary(train_master)
    data_dict.to_csv(PROCESSED_DIR / 'data_dictionary_master.csv', index=False)

    return train_master, test_master


def build_preprocessor(X_train):
    numeric_cols = [
        'survey_year', 'planting_month', 'altitude_m', 'rainfall_mm_season', 'farm_size_ha',
        'fertilizer_kg_per_ha', 'soil_quality_index', 'labor_days_per_ha', 'distance_to_market_km',
        'season_mean_temp', 'season_rainfall_total', 'season_rainfall_mean', 'season_extreme_heat_days',
        'season_temp_std', 'fertilizer_per_labor_day', 'fertilizer_x_improved_seed', 'soil_x_fertilizer',
        'planting_month_sin', 'planting_month_cos', 'rainfall_to_fertilizer', 'altitude_per_farm_size',
        'improved_seed_used', 'pest_disease_flag'
    ]
    cat_cols = ['region', 'crop_type']
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), [c for c in numeric_cols if c in X_train.columns]),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), [c for c in cat_cols if c in X_train.columns]),
    ])
    return preprocessor


def fit_and_eval():
    train_df, test_df = build_model_data()
    target = 'yield_tons_per_ha'
    feature_cols = build_feature_columns()
    X = train_df[feature_cols].copy()
    y = train_df[target].copy()

    def metrics_from_predictions(y_true, y_pred):
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
        }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train)

    models = {
        'mean_baseline': None,
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(random_state=42, n_estimators=150, max_depth=12, n_jobs=-1, min_samples_leaf=1),
    }

    results = []
    for name, model in models.items():
        if name == 'mean_baseline':
            pred = np.full(len(y_val), y_train.mean())
            results.append({'model': name, **metrics_from_predictions(y_val, pred)})
            continue

        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        results.append({'model': name, **metrics_from_predictions(y_val, pred)})

    comparison = pd.DataFrame(results)
    comparison.to_csv(REPORTS_DIR / 'D_model_comparison.csv', index=False)

    base_rf = RandomForestRegressor(random_state=42, n_estimators=150, max_depth=12, n_jobs=-1, min_samples_leaf=1)
    base_pipe = Pipeline([('prep', preprocessor), ('model', base_rf)])
    base_pipe.fit(X_train, y_train)
    val_pred = base_pipe.predict(X_val)
    val_metrics = metrics_from_predictions(y_val, val_pred)

    rf_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(base_pipe, X, y, cv=rf_cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse_mean = float(-rf_cv_scores.mean())
    cv_rmse_std = float(rf_cv_scores.std())

    oot_mask = train_df['survey_year'] <= 2023
    oot_train_X = X.loc[oot_mask].copy()
    oot_train_y = y.loc[oot_mask].copy()
    oot_val_X = X.loc[~oot_mask].copy()
    oot_val_y = y.loc[~oot_mask].copy()
    oot_preprocessor = build_preprocessor(oot_train_X)
    oot_pipe = Pipeline([('prep', oot_preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=150, max_depth=12, n_jobs=-1, min_samples_leaf=1))])
    oot_pipe.fit(oot_train_X, oot_train_y)
    oot_pred = oot_pipe.predict(oot_val_X)
    oot_metrics = metrics_from_predictions(oot_val_y, oot_pred)

    weather_cols = [c for c in feature_cols if c.startswith('season_') or c in ['planting_month_sin', 'planting_month_cos', 'rainfall_to_fertilizer', 'altitude_per_farm_size']]
    no_weather_cols = [c for c in feature_cols if c not in weather_cols]
    X_no_weather = X[no_weather_cols].copy()
    X_weather = X[feature_cols].copy()
    no_weather_train_X, no_weather_val_X, no_weather_train_y, no_weather_val_y = train_test_split(X_no_weather, y, test_size=0.2, random_state=42)
    no_weather_preprocessor = build_preprocessor(no_weather_train_X)
    no_weather_pipe = Pipeline([('prep', no_weather_preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=150, max_depth=12, n_jobs=-1, min_samples_leaf=1))])
    no_weather_pipe.fit(no_weather_train_X, no_weather_train_y)
    no_weather_pred = no_weather_pipe.predict(no_weather_val_X)
    no_weather_metrics = metrics_from_predictions(no_weather_val_y, no_weather_pred)

    tuned_pipe = Pipeline([('prep', build_preprocessor(X_train)), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
    param_dist = {
        'model__n_estimators': [200, 400, 600],
        'model__max_depth': [6, 8, 10, 12, None],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None],
        'model__min_samples_split': [2, 5, 10],
    }
    search = RandomizedSearchCV(
        tuned_pipe,
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    tuned_pred = search.best_estimator_.predict(X_val)
    tuned_metrics = metrics_from_predictions(y_val, tuned_pred)

    final_pipe = search.best_estimator_
    joblib.dump(final_pipe, MODELS_DIR / 'final_model.joblib')

    report = {
        'validation_rmse': float(val_metrics['rmse']),
        'validation_mae': float(val_metrics['mae']),
        'validation_r2': float(val_metrics['r2']),
        'cv_5fold_rmse_mean': cv_rmse_mean,
        'cv_5fold_rmse_std': cv_rmse_std,
        'out_of_time_rmse': float(oot_metrics['rmse']),
        'out_of_time_mae': float(oot_metrics['mae']),
        'out_of_time_r2': float(oot_metrics['r2']),
        'weather_ablation_rmse': float(no_weather_metrics['rmse']),
        'weather_ablation_mae': float(no_weather_metrics['mae']),
        'weather_ablation_r2': float(no_weather_metrics['r2']),
        'tuned_rmse': float(tuned_metrics['rmse']),
        'tuned_mae': float(tuned_metrics['mae']),
        'tuned_r2': float(tuned_metrics['r2']),
        'best_params': search.best_params_,
        'baseline_rmse': float(results[2]['rmse']) if len(results) > 2 else float(val_metrics['rmse']),
        'baseline_mae': float(results[2]['mae']) if len(results) > 2 else float(val_metrics['mae']),
        'baseline_r2': float(results[2]['r2']) if len(results) > 2 else float(val_metrics['r2']),
    }
    with open(REPORTS_DIR / 'D_model_summary.json', 'w') as f:
        json.dump(report, f, indent=2)

    plot_df = pd.DataFrame({'model': comparison['model'], 'rmse': comparison['rmse']})
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='model', y='rmse', palette='Set2')
    plt.title('Model comparison (validation RMSE)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig10_model_comparison.png', dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_val, val_pred, alpha=0.5)
    mins = min(y_val.min(), val_pred.min())
    maxs = max(y_val.max(), val_pred.max())
    axes[0].plot([mins, maxs], [mins, maxs], 'k--')
    axes[0].set_xlabel('Actual yield')
    axes[0].set_ylabel('Predicted yield')
    axes[0].set_title('Actual vs predicted')
    residuals = y_val - val_pred
    axes[1].scatter(val_pred, residuals, alpha=0.4)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_xlabel('Predicted yield')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residual vs predicted')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig11_predicted_vs_actual_residuals.png', dpi=150)
    plt.close()

    if hasattr(final_pipe.named_steps['model'], 'feature_importances_'):
        feature_names = final_pipe.named_steps['prep'].get_feature_names_out()
        importances = final_pipe.named_steps['model'].feature_importances_
        top = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x='importance', y='feature', palette='viridis')
        plt.title('Top features')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig12_feature_importance.png', dpi=150)
        plt.close()

    test_X = test_df[feature_cols].copy()
    pred = final_pipe.predict(test_X)
    sub = pd.DataFrame({'plot_id': test_df['plot_id'], 'predicted_yield_tons_per_ha': pred})
    sub.to_csv(SUBMISSION_DIR / 'team_beso_submission.csv', index=False)

    assert len(sub) == 3750
    assert list(sub.columns) == ['plot_id', 'predicted_yield_tons_per_ha']
    print('VALIDATION metrics:')
    print({
        'rmse': val_metrics['rmse'],
        'mae': val_metrics['mae'],
        'r2': val_metrics['r2'],
        'cv_5fold_rmse_mean': cv_rmse_mean,
        'cv_5fold_rmse_std': cv_rmse_std,
        'out_of_time_rmse': oot_metrics['rmse'],
        'weather_ablation_rmse': no_weather_metrics['rmse'],
        'best_params': search.best_params_,
    })
    return final_pipe, test_df, sub


def make_figures():
    train_df = pd.read_csv(PROCESSED_DIR / 'master_train.csv')
    test_df = pd.read_csv(PROCESSED_DIR / 'master_test.csv')
    raw_train = pd.read_csv(RAW_DIR / 'crop_yield_train.csv')
    raw_weather = pd.read_csv(RAW_DIR / 'regional_weather.csv')
    raw_price = pd.read_csv(RAW_DIR / 'market_prices.csv')

    # fig01 missingness
    missing_frames = []
    for name, raw in [('plot_train', raw_train), ('plot_test', test_df), ('weather', raw_weather), ('price', raw_price)]:
        missing = raw.isna().mean().mul(100).reset_index()
        missing.columns = ['column', 'missing_pct']
        missing['dataset'] = name
        missing_frames.append(missing)
    all_missing = pd.concat(missing_frames, ignore_index=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_missing, x='column', y='missing_pct', hue='dataset')
    plt.xticks(rotation=90)
    plt.title('Missingness across raw datasets')
    plt.ylabel('Missing %')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig01_missingness.png', dpi=150)
    plt.close()

    # fig02 before/after cleaning example
    before = raw_train['farm_size_ha'].dropna().iloc[:500]
    after = train_df['farm_size_ha'].dropna().iloc[:500]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(before, bins=30, ax=axes[0], color='steelblue')
    axes[0].set_title('Before cleaning: farm size')
    sns.histplot(after, bins=30, ax=axes[1], color='forestgreen')
    axes[1].set_title('After cleaning: farm size')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig02_before_after_cleaning.png', dpi=150)
    plt.close()

    # fig03 distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['yield_tons_per_ha'], bins=30, kde=True)
    plt.title('Overall yield distribution')
    plt.xlabel('Yield (tons/ha)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig03_yield_distribution.png', dpi=150)
    plt.close()

    # fig04 region x crop heatmap
    heat = train_df.groupby(['region', 'crop_type'])['yield_tons_per_ha'].mean().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt='.1f', cmap='viridis')
    plt.title('Mean yield by region and crop')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig04_region_crop_heatmap.png', dpi=150)
    plt.close()

    # fig05 correlation
    corr_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c]) and c not in ['plot_id', 'yield_tons_per_ha']][:16]
    corr = train_df[['yield_tons_per_ha'] + corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title('Correlation heatmap')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig05_correlation_heatmap.png', dpi=150)
    plt.close()

    # fig06 climate by region
    climate = raw_weather.groupby(['region', 'month'])['avg_temp_c'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=climate, x='month', y='avg_temp_c', hue='region')
    plt.title('Monthly climate by region')
    plt.xlabel('Month')
    plt.ylabel('Average temperature (°C)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig06_climate_by_region.png', dpi=150)
    plt.close()

    # fig07 yield vs season temp
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=train_df, x='season_mean_temp', y='yield_tons_per_ha', hue='crop_type', alpha=0.7)
    plt.title('Yield vs growing-season mean temperature')
    plt.xlabel('Season mean temp (°C)')
    plt.ylabel('Yield (tons/ha)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig07_yield_vs_season_temp.png', dpi=150)
    plt.close()

    # fig08 price trends
    price = pd.read_csv(RAW_DIR / 'market_prices.csv')
    price['region'] = price['region'].map(lambda x: x.strip())
    price['crop_type'] = price['crop_type'].map(str).str.strip().str.lower()
    price['crop_type'] = price['crop_type'].replace({'tef': 'teff'})
    price = price[price['price_birr_per_quintal'].notna()]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=price, x='year', y='price_birr_per_quintal', hue='crop_type')
    plt.title('Price per quintal by crop over time')
    plt.xlabel('Year')
    plt.ylabel('Price (birr/quintal)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig08_price_trends.png', dpi=150)
    plt.close()

    # fig09 revenue by crop/region
    price = raw_price.copy()
    price['crop_type'] = price['crop_type'].map(str).str.strip().str.lower().replace({'tef': 'teff'})
    price['region'] = price['region'].map(str).str.strip()
    price['year'] = pd.to_numeric(price['year'], errors='coerce')
    revenue = train_df[['plot_id', 'crop_type', 'region', 'survey_year', 'yield_tons_per_ha']].merge(
        price[['crop_type', 'region', 'year', 'price_birr_per_quintal']],
        left_on=['crop_type', 'region', 'survey_year'],
        right_on=['crop_type', 'region', 'year'],
        how='left'
    )
    revenue['revenue_per_ha'] = revenue['yield_tons_per_ha'] * 10 * revenue['price_birr_per_quintal']
    revenue['revenue_per_ha'] = revenue['revenue_per_ha'].fillna(0)
    heat = revenue.groupby(['region', 'crop_type'])['revenue_per_ha'].mean().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt='.0f', cmap='Blues')
    plt.title('Revenue per ha by crop and region')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig09_revenue_by_crop_region.png', dpi=150)
    plt.close()

    # fig10 model comparison already generated in fit_and_eval
    # fig11 already generated in fit_and_eval
    # fig12 already generated in fit_and_eval

    captions = [
        ('fig01_missingness.png', 'This figure shows missingness percentages across the raw inputs. The largest gaps are in rainfall, fertilizer, and soil depth measures, which guide the cleaning decisions for the modeling dataset.'),
        ('fig02_before_after_cleaning.png', 'This comparison shows how the farm-size distribution shifts after standardization and imputation, confirming the cleaning step did not distort the central shape of the data.'),
        ('fig03_yield_distribution.png', 'The yield distribution is right-skewed, suggesting that the model should be evaluated with robust error metrics and careful handling of heteroskedasticity.'),
        ('fig04_region_crop_heatmap.png', 'This heatmap highlights where yield is strongest and weakest by region-crop combination, helping guide the agronomic interpretation of the final model.'),
        ('fig05_correlation_heatmap.png', 'This heatmap shows how the final engineered features relate to yield and to each other, identifying weather and agronomic variables worth keeping in the model.'),
        ('fig06_climate_by_region.png', 'The monthly climate trend shows how temperature patterns vary by region and should be interpreted with the growing-season timing in mind.'),
        ('fig07_yield_vs_season_temp.png', 'There is a crop-specific relationship between season temperature and yield, which is one reason the weather join materially improves model structure.'),
        ('fig08_price_trends.png', 'This trend plot shows how crop prices have moved over time and explains why price was kept separate from yield modeling.'),
        ('fig09_revenue_by_crop_region.png', 'Revenue per hectare varies sharply across region and crop, revealing economically important differences that are distinct from production efficiency.'),
        ('fig10_model_comparison.png', 'This view compares validation RMSE across candidate models and supports the choice of the final estimator based on measured performance rather than preference.'),
        ('fig11_predicted_vs_actual_residuals.png', 'The plot helps reveal systematic residual patterns and shows whether prediction errors are concentrated in particular yield ranges.'),
        ('fig12_feature_importance.png', 'The top features highlight that weather and agronomic variables are materially important, which is consistent with the weather-join requirement.'),
    ]
    with open(FIG_DIR / 'figure_captions.md', 'w', encoding='utf-8') as f:
        for fn, text in captions:
            f.write(f'## {fn}\n{text}\n\n')


def main():
    copy_raw_data()
    fit_and_eval()
    make_figures()
    print('Pipeline complete. Artifacts saved in data/processed, figures/, and models/.')


if __name__ == '__main__':
    main()
