from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'final_model.joblib'
PRICE_PATH = ROOT / 'data' / 'raw' / 'market_prices.csv'
WEATHER_PATH = ROOT / 'data' / 'raw' / 'regional_weather.csv'

FEATURE_COLUMNS = [
    'survey_year', 'planting_month', 'altitude_m', 'rainfall_mm_season', 'farm_size_ha',
    'fertilizer_kg_per_ha', 'soil_quality_index', 'labor_days_per_ha', 'distance_to_market_km',
    'season_mean_temp', 'season_rainfall_total', 'season_rainfall_mean', 'season_extreme_heat_days',
    'season_temp_std', 'fertilizer_per_labor_day', 'fertilizer_x_improved_seed', 'soil_x_fertilizer',
    'planting_month_sin', 'planting_month_cos', 'rainfall_to_fertilizer', 'altitude_per_farm_size',
    'region', 'crop_type', 'improved_seed_used', 'pest_disease_flag'
]

T = {
    'en': {
        'language': 'Language',
        'title': '🌱 Crop Yield & Revenue Planner',
        'subtitle': 'Regional climate-aware forecasting dashboard',
        'farm_inputs': '🌱 Farm Inputs',
        'region': 'Region',
        'crop_type': 'Crop type',
        'survey_year': 'Survey year',
        'planting_month': 'Planting month',
        'altitude': 'Altitude (m)',
        'farm_size': 'Farm size (ha)',
        'fertilizer': 'Fertilizer (kg/ha)',
        'soil_quality': 'Soil quality index',
        'labor': 'Labor days/ha',
        'distance': 'Distance to market (km)',
        'improved_seed': 'Improved seed used',
        'pest_flag': 'Pest/disease flag',
        'predicted_yield': 'Predicted yield',
        'estimated_revenue': 'Estimated revenue',
        'market_price': 'Market price',
        'farm_assumptions': '📋 Farm assumptions',
        'season_outlook': '🌦️ Season outlook',
        'yield_comparison': '📈 Yield comparison',
        'scenario': 'Scenario',
        'model_forecast': 'Model forecast',
        'regional_baseline': 'Regional baseline',
        'yield_t_ha': 'Yield (t/ha)',
        'caption': 'Values reflect region-season climate patterns and management inputs used by the trained prediction model.',
        'yes': 'Yes',
        'no': 'No',
    },
    'am': {
        'language': 'ቋንቋ',
        'title': '🌱 የምርት ምርት እና ገቢ ተቀያሪ',
        'subtitle': 'አካባቢያዊ የአየር ንብረት እውቀት ያለው ትንበያ ዳሽቦርድ',
        'farm_inputs': '🌱 የእርሻ ግብዓቶች',
        'region': 'ክልል',
        'crop_type': 'የእፅዋት ዓይነት',
        'survey_year': 'የጥናት ዓመት',
        'planting_month': 'የመዝራት ወር',
        'altitude': 'ከፍታ (ሜትር)',
        'farm_size': 'የእርሻ መጠን (ሄክታር)',
        'fertilizer': 'ማዳበሪያ (ኪ.ግ/ሄክታር)',
        'soil_quality': 'የአፈር ጥራት መረጃ',
        'labor': 'የስራ ቀናት/ሄክታር',
        'distance': 'ከገበያ የርቀት (ኪ.ሜ)',
        'improved_seed': 'የማሻሻያ ዘር ጥቅም ላይ ውሏል',
        'pest_flag': 'በተባይ እና በበሽታ ጉዳት ላይ የሚመለከት',
        'predicted_yield': 'የተነበበ ምርት',
        'estimated_revenue': 'የተገመተ ገቢ',
        'market_price': 'የገበያ ዋጋ',
        'farm_assumptions': '📋 የእርሻ ግምቶች',
        'season_outlook': '🌦️ የወቅቱ እይታ',
        'yield_comparison': '📈 የምርት ንጽጽር',
        'scenario': 'ሁኔታ',
        'model_forecast': 'የሞዴል ትንበያ',
        'regional_baseline': 'የክልል መሠረታዊ እውቀት',
        'yield_t_ha': 'ምርት (ቶን/ሄክታር)',
        'caption': 'እሴቶቹ በክልል እና በወቅት የአየር ንብረት እና የእርሻ ግብዓቶች ላይ የተመሰረቱ ናቸው።',
        'yes': 'አዎ',
        'no': 'አይ',
    },
}

REGION_LABELS = {
    'en': {
        'Amhara': 'Amhara',
        'Oromia': 'Oromia',
        'SNNPR': 'SNNPR',
        'Somali': 'Somali',
        'Tigray': 'Tigray',
    },
    'am': {
        'Amhara': 'አማራ',
        'Oromia': 'ኦሮሚያ',
        'SNNPR': 'ደቡብ',
        'Somali': 'ሶማሊ',
        'Tigray': 'ትግራይ',
    },
}

CROP_LABELS = {
    'en': {
        'barley': 'barley',
        'maize': 'maize',
        'sorghum': 'sorghum',
        'teff': 'teff',
        'wheat': 'wheat',
    },
    'am': {
        'barley': 'ገብስ',
        'maize': 'በቆሎ',
        'sorghum': 'ማሽላ',
        'teff': 'ጤፍ',
        'wheat': 'ስንዴ',
    },
}


def text(lang, key):
    return T.get(lang, T['en']).get(key, T['en'][key])


def region_label(lang, value):
    return REGION_LABELS.get(lang, REGION_LABELS['en']).get(value, value)


def crop_label(lang, value):
    return CROP_LABELS.get(lang, CROP_LABELS['en']).get(value, value)


st.set_page_config(
    page_title='Crop Yield Planner',
    page_icon='🌾',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5fff7 0%, #eef8ff 40%, #fff9ef 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #1d3557 100%);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #173f5f;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(245,250,255,0.92));
        border: 1px solid rgba(32, 117, 90, 0.12);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 25px rgba(18, 67, 91, 0.08);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800;
        color: #0f766e;
    }
    [data-testid="stMetricLabel"] {
        color: #3a5368;
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(17, 24, 39, 0.06);
    }
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #14b8a6, #3b82f6);
        color: white;
        font-weight: 700;
    }
    .stAlert {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_region_name(value):
    """Map raw region labels to the canonical names used in the app."""
    if value is None or pd.isna(value):
        return ""

    normalized = str(value).strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    aliases = {
        'amhara': 'Amhara',
        'amh': 'Amhara',
        'oromia': 'Oromia',
        'oro': 'Oromia',
        'snnpr': 'SNNPR',
        'snnp': 'SNNPR',
        'somali': 'Somali',
        'som': 'Somali',
        'tigray': 'Tigray',
        'tig': 'Tigray',
    }
    return aliases.get(normalized, str(value).strip())


def normalize_crop_name(value):
    """Standardize crop labels so app filters match the raw data format."""
    if value is None or pd.isna(value):
        return ""

    normalized = str(value).strip().lower().replace('-', '').replace('_', ' ')
    normalized = ' '.join(normalized.split())
    aliases = {'tef': 'teff', 'teff': 'teff'}
    return aliases.get(normalized, normalized)


def load_data():
    """Load pricing, weather, and trained model inputs."""
    prices = pd.read_csv(PRICE_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    model = joblib.load(MODEL_PATH)
    return prices, weather, model


def season_features(region, year, planting_month, weather):
    """Return seasonal climate metrics for a region and planting window.

    If the exact season is absent from the source data, return missing values rather than
    fabricating a seasonal estimate from unrelated rows.
    """
    season_months = []
    for delta in range(4):
        month = planting_month + delta
        season_year = year
        while month > 12:
            month -= 12
            season_year += 1
        season_months.append((season_year, month))

    normalized_region = normalize_region_name(region)
    weather = weather.copy()
    weather['__region_norm'] = weather['region'].map(normalize_region_name)
    region_rows = weather[weather['__region_norm'] == normalized_region].copy()

    exact_rows = region_rows[
        (region_rows['year'].isin([y for y, _ in season_months]))
        & (region_rows['month'].isin([m for _, m in season_months]))
    ].copy()

    if exact_rows.empty:
        return {
            'season_mean_temp': float('nan'),
            'season_rainfall_total': float('nan'),
            'season_rainfall_mean': float('nan'),
            'season_extreme_heat_days': float('nan'),
            'season_temp_std': float('nan'),
        }

    return {
        'season_mean_temp': float(exact_rows['avg_temp_c'].mean()),
        'season_rainfall_total': float(exact_rows['monthly_rainfall_mm'].sum()),
        'season_rainfall_mean': float(exact_rows['monthly_rainfall_mm'].mean()),
        'season_extreme_heat_days': float(exact_rows['extreme_heat_days'].sum()),
        'season_temp_std': float(exact_rows['avg_temp_c'].std(ddof=0)),
    }


def build_feature_row(region, crop, survey_year, planting_month, altitude, farm_size, fertilizer, improved_seed, pest_flag, soil_quality, labor, distance, weather):
    """Build a single feature row matching the trained model schema."""
    season = season_features(region, survey_year, planting_month, weather)
    rainfall_total = season['season_rainfall_total']
    if pd.isna(rainfall_total):
        rainfall_total = 0.0

    row = {
        'survey_year': survey_year,
        'planting_month': planting_month,
        'altitude_m': altitude,
        'rainfall_mm_season': rainfall_total,
        'farm_size_ha': farm_size,
        'fertilizer_kg_per_ha': fertilizer,
        'soil_quality_index': soil_quality,
        'labor_days_per_ha': labor,
        'distance_to_market_km': distance,
        'season_mean_temp': season['season_mean_temp'] if not pd.isna(season['season_mean_temp']) else 22.0,
        'season_rainfall_total': season['season_rainfall_total'] if not pd.isna(season['season_rainfall_total']) else 0.0,
        'season_rainfall_mean': season['season_rainfall_mean'] if not pd.isna(season['season_rainfall_mean']) else 0.0,
        'season_extreme_heat_days': season['season_extreme_heat_days'] if not pd.isna(season['season_extreme_heat_days']) else 0.0,
        'season_temp_std': season['season_temp_std'] if not pd.isna(season['season_temp_std']) else 0.0,
        'fertilizer_per_labor_day': fertilizer / labor if labor > 0 else 0.0,
        'fertilizer_x_improved_seed': fertilizer * improved_seed,
        'soil_x_fertilizer': soil_quality * fertilizer,
        'planting_month_sin': math.sin(2 * math.pi * planting_month / 12),
        'planting_month_cos': math.cos(2 * math.pi * planting_month / 12),
        'rainfall_to_fertilizer': rainfall_total / fertilizer if fertilizer > 0 else 0.0,
        'altitude_per_farm_size': altitude / farm_size if farm_size > 0 else 0.0,
        'region': region,
        'crop_type': crop,
        'improved_seed_used': int(improved_seed),
        'pest_disease_flag': int(pest_flag),
    }
    frame = pd.DataFrame([row])
    frame = frame.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return frame


def main():
    """Run the Streamlit dashboard and compute forecast scenarios."""
    prices, weather, model = load_data()
    lang = st.sidebar.selectbox(text('en', 'language'), ['English', 'አማርኛ'])
    current_lang = 'am' if lang == 'አማርኛ' else 'en'

    with st.sidebar:
        st.title(text(current_lang, 'farm_inputs'))
        region = st.selectbox(
            text(current_lang, 'region'),
            ['Amhara', 'Oromia', 'SNNPR', 'Somali', 'Tigray'],
            format_func=lambda value: region_label(current_lang, value),
        )
        crop = st.selectbox(
            text(current_lang, 'crop_type'),
            ['barley', 'maize', 'sorghum', 'teff', 'wheat'],
            format_func=lambda value: crop_label(current_lang, value),
        )
        survey_year = st.selectbox(text(current_lang, 'survey_year'), [2021, 2022, 2023, 2024])
        planting_month = st.selectbox(text(current_lang, 'planting_month'), list(range(1, 13)), index=6)

        st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)
        altitude = st.number_input(text(current_lang, 'altitude'), value=2000.0, min_value=0.0, step=50.0)
        farm_size = st.number_input(text(current_lang, 'farm_size'), value=1.0, min_value=0.01, step=0.1)
        fertilizer = st.number_input(text(current_lang, 'fertilizer'), value=30.0, min_value=0.0, step=5.0)
        soil_quality = st.number_input(text(current_lang, 'soil_quality'), value=0.5, min_value=0.0, max_value=1.0, step=0.05)
        labor = st.number_input(text(current_lang, 'labor'), value=40.0, min_value=0.0, step=5.0)
        distance = st.number_input(text(current_lang, 'distance'), value=12.0, min_value=0.0, step=1.0)
        improved_seed = st.selectbox(text(current_lang, 'improved_seed'), [0, 1])
        pest_flag = st.selectbox(text(current_lang, 'pest_flag'), [0, 1])

    price_frame = prices.copy()
    price_frame['__region_norm'] = price_frame['region'].map(normalize_region_name)
    price_frame['__crop_norm'] = price_frame['crop_type'].map(normalize_crop_name)
    price_row = price_frame[
        (price_frame['__region_norm'] == normalize_region_name(region))
        & (price_frame['__crop_norm'] == normalize_crop_name(crop))
        & (price_frame['year'] == survey_year)
    ].copy()
    price_row = price_row.dropna(subset=['price_birr_per_quintal'])
    price_value = float(price_row['price_birr_per_quintal'].mean()) if not price_row.empty else 0.0
    feature_df = build_feature_row(
        region, crop, survey_year, planting_month, altitude, farm_size, fertilizer,
        improved_seed, pest_flag, soil_quality, labor, distance, weather
    )

    prediction = float(model.predict(feature_df)[0])
    revenue = prediction * farm_size * 10 * price_value
    season = season_features(region, survey_year, planting_month, weather)

    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg,#0f766e,#10b981); padding: 1.5rem 1.5rem; border-radius: 22px; margin-bottom: 1rem; box-shadow: 0 18px 35px rgba(16,185,129,0.2);'>
            <h1 style='color:white; margin:0; font-size:2.2rem;'>{text(current_lang, 'title')}</h1>
            <p style='color:rgba(255,255,255,0.88); margin:0.4rem 0 0 0;'>{text(current_lang, 'subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(text(current_lang, 'predicted_yield'), f'{prediction:.2f} t/ha')
    col2.metric(text(current_lang, 'estimated_revenue'), f'{revenue:,.0f} birr')
    col3.metric(text(current_lang, 'market_price'), f'{price_value:,.0f} birr/quintal')

    st.markdown('---')

    left, right = st.columns(2)
    with left:
        st.subheader(text(current_lang, 'farm_assumptions'))
        summary = pd.DataFrame([
            {'Metric': text(current_lang, 'region'), 'Value': region_label(current_lang, region)},
            {'Metric': text(current_lang, 'crop_type'), 'Value': crop_label(current_lang, crop)},
            {'Metric': text(current_lang, 'planting_month'), 'Value': planting_month},
            {'Metric': text(current_lang, 'survey_year'), 'Value': survey_year},
            {'Metric': text(current_lang, 'farm_size'), 'Value': f'{farm_size:.1f} ha'},
            {'Metric': text(current_lang, 'fertilizer'), 'Value': f'{fertilizer:.1f} kg/ha'},
            {'Metric': text(current_lang, 'soil_quality'), 'Value': f'{soil_quality:.2f}'},
            {'Metric': text(current_lang, 'improved_seed'), 'Value': text(current_lang, 'yes') if improved_seed else text(current_lang, 'no')},
        ])
        st.dataframe(summary, hide_index=True, use_container_width=True)

    with right:
        st.subheader(text(current_lang, 'season_outlook'))
        seasonal = pd.DataFrame([
            {'Metric': 'Mean temp', 'Value': f'{season["season_mean_temp"]:.1f} °C' if not pd.isna(season["season_mean_temp"]) else 'No data'},
            {'Metric': 'Rainfall total', 'Value': f'{season["season_rainfall_total"]:.1f} mm' if not pd.isna(season["season_rainfall_total"]) else 'No data'},
            {'Metric': 'Rainfall mean', 'Value': f'{season["season_rainfall_mean"]:.1f} mm' if not pd.isna(season["season_rainfall_mean"]) else 'No data'},
            {'Metric': 'Extreme heat days', 'Value': f'{season["season_extreme_heat_days"]:.0f}' if not pd.isna(season["season_extreme_heat_days"]) else 'No data'},
            {'Metric': 'Temp variability', 'Value': f'{season["season_temp_std"]:.1f} °C' if not pd.isna(season["season_temp_std"]) else 'No data'},
        ])
        st.dataframe(seasonal, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.subheader(text(current_lang, 'yield_comparison'))
    comparison = pd.DataFrame({
        'Scenario': [text(current_lang, 'model_forecast'), text(current_lang, 'regional_baseline')],
        'Yield (t/ha)': [prediction, 2.4],
    })
    st.bar_chart(comparison.set_index('Scenario'))

    st.caption(text(current_lang, 'caption'))


if __name__ == '__main__':
    main()
