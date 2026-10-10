from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
MODEL_PATH = ASSETS_DIR / 'final_model.joblib'
PRICE_PATH = ASSETS_DIR / 'cleaned_price.csv'
WEATHER_PATH = ASSETS_DIR / 'cleaned_weather.csv'

# Region aliases for normalizing inconsistent labels found in cleaned assets
REGION_ALIASES = {
    'amhara': 'Amhara', 'amh': 'Amhara', 'amhara ': 'Amhara',
    'oromia': 'Oromia', 'oro': 'Oromia', 'oromia ': 'Oromia',
    'snnpr': 'SNNPR', 'snnp': 'SNNPR', 'snnp': 'SNNPR', 'snnpr ': 'SNNPR',
    'somali': 'Somali', 'som': 'Somali', 'somali ': 'Somali',
    'tigray': 'Tigray', 'tig': 'Tigray', 'tigray ': 'Tigray',
}

# Crop aliases for normalizing inconsistent labels found in cleaned assets
CROP_ALIASES = {
    'barley': 'barley', 'barley ': 'barley',
    'maize': 'maize', 'maize ': 'maize',
    'sorghum': 'sorghum', 'sorghum ': 'sorghum',
    'tef': 'teff', 'teff': 'teff', 'teff ': 'teff',
    'wheat': 'wheat', 'wheat ': 'wheat',
}

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
        'mean_temp': 'Mean temp',
        'rainfall_total': 'Rainfall total',
        'rainfall_mean': 'Rainfall mean',
        'extreme_heat_days': 'Extreme heat days',
        'temp_variability': 'Temp variability',
    },
    'am': {
        'language': 'ቋንቋ',
        'title': '🌱 የምርት ውጤት እና የገቢ እቅድ',
        'subtitle': 'አካባቢያዊ የአየር ሁኔታ የተመለከተ ትንበያ ዳሽቦርድ',
        'farm_inputs': '🌱 የእርሻ ግብዓቶች',
        'region': 'ክልል',
        'crop_type': 'የእፅዋት ዓይነት',
        'survey_year': 'የጥናት ዓመት',
        'planting_month': 'የመዝራት ወር',
        'altitude': 'ከፍታ (ሜትር)',
        'farm_size': 'የእርሻ መጠን (ሄክታር)',
        'fertilizer': 'ማዳበሪያ (ኪ.ግ/ሄክታር)',
        'soil_quality': 'የአፈር ጥራት መረጃ',
        'labor': 'የስራ ቀናት/በሄክታር',
        'distance': 'ከገበያ የርቀት (ኪ.ሜ)',
        'improved_seed': 'የማሻሻያ ዘር ጥቅም ላይ ውሏል',
        'pest_flag': 'የተባይ እና በሽታ ጉዳት',
        'predicted_yield': 'የተነበበ ምርት',
        'estimated_revenue': 'የተገመተ ገቢ',
        'market_price': 'የገበያ ዋጋ',
        'farm_assumptions': '📋 የእርሻ ግምቶች',
        'season_outlook': '🌦️ የወቅቱ እይታ',
        'yield_comparison': '📈 የምርት ምዘና',
        'scenario': 'ሁኔታ',
        'model_forecast': 'የሞዴል ትንበያ',
        'regional_baseline': 'የክልል መሠረታዊ እውቀት',
        'yield_t_ha': 'ምርት (ቶን/ሄክታር)',
        'caption': 'እሴቶቹ በክልል እና በወቅት የአየር ንብረት እና የእርሻ ግብዓቶች ላይ የተመሰረቱ ናቸው።',
        'yes': 'አዎ',
        'no': 'አይ',
        'mean_temp': 'የበረታዊ ጡና ልምድ',
        'rainfall_total': 'የጧንታ አነጋገር',
        'rainfall_mean': 'የጧንታ ቃልል',
        'extreme_heat_days': 'ከፍተኛ ስንዴ ቀናት',
        'temp_variability': 'የጡን቞ ሁኔታ',
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
    return REGION_ALIASES.get(normalized, str(value).strip())


def normalize_crop_name(value):
    """Standardize crop labels so app filters match the raw data format."""
    if value is None or pd.isna(value):
        return ""

    normalized = str(value).strip().lower().replace('-', '').replace('_', ' ')
    normalized = ' '.join(normalized.split())
    return CROP_ALIASES.get(normalized, normalized)


def load_data():
    """Load pricing, weather, and trained model inputs from assets.

    The cleaned assets may contain inconsistent region/crop labels (e.g. 'SOM', 'AMH', 'ORO', 'TIG').
    We normalize these on load so downstream lookups work reliably.
    """
    prices = pd.read_csv(PRICE_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    model = joblib.load(MODEL_PATH)

    # Normalize region labels
    prices['region'] = prices['region'].map(normalize_region_name)
    weather['region'] = weather['region'].map(normalize_region_name)

    # Normalize crop labels
    prices['crop_type'] = prices['crop_type'].map(normalize_crop_name)
    weather['crop_type'] = weather['crop_type'].map(normalize_crop_name) if 'crop_type' in weather.columns else None

    # Convert month names to integers if needed
    if weather['month'].dtype == object:
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
        }
        weather['month'] = weather['month'].map(month_map)

    # Ensure year is integer
    weather['year'] = weather['year'].astype(int)
    prices['year'] = prices['year'].astype(int)

    # Drop rows with missing critical values
    weather = weather.dropna(subset=['region', 'year', 'month'])
    prices = prices.dropna(subset=['region', 'crop_type', 'year'])

    return prices, weather, model


def season_features(region, year, planting_month, weather):
    """Return seasonal climate metrics for a region and planting window.

    The growing season is the planting month plus the next three months (4-month window).
    If an exact season match is missing, fall back to nearby regional values rather than
    returning zero or NaN for the whole season. This keeps the dashboard informative and
    prevents the metrics from collapsing to zero for valid user inputs.
    """
    season_months = []
    season_year = year
    for delta in range(4):
        month = planting_month + delta
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

    target_rows = exact_rows if not exact_rows.empty else region_rows.copy()
    if target_rows.empty:
        target_rows = weather.copy()

    def safe_mean(series):
        return float(series.mean()) if not series.empty and not pd.isna(series.mean()) else 0.0

    return {
        'season_mean_temp': safe_mean(target_rows['avg_temp_c']),
        'season_rainfall_total': float(target_rows['monthly_rainfall_mm'].sum()) if not target_rows.empty else 0.0,
        'season_rainfall_mean': safe_mean(target_rows['monthly_rainfall_mm']),
        'season_extreme_heat_days': float(target_rows['extreme_heat_days'].sum()) if not target_rows.empty else 0.0,
        'season_temp_std': float(target_rows['avg_temp_c'].std(ddof=0)) if not target_rows.empty and target_rows['avg_temp_c'].notna().any() else 0.0,
    }


def yes_to_int(value):
    """Convert Yes/No strings to 1/0 integers for the model."""
    return 1 if value.lower() == 'yes' or value.lower() == 'አዎ' else 0


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
        altitude = st.slider(text(current_lang, 'altitude'), min_value=0.0, max_value=4000.0, value=2000.0, step=50.0, help='Altitude above sea level in meters')
        farm_size = st.slider(text(current_lang, 'farm_size'), min_value=0.1, max_value=10.0, value=1.0, step=0.1, help='Total farm area in hectares')
        fertilizer = st.slider(text(current_lang, 'fertilizer'), min_value=0.0, max_value=200.0, value=30.0, step=5.0, help='Fertilizer applied per hectare')
        soil_quality = st.slider(text(current_lang, 'soil_quality'), min_value=0.0, max_value=1.0, value=0.5, step=0.05, help='Soil quality index (0=poor, 1=excellent)')
        labor = st.slider(text(current_lang, 'labor'), min_value=0.0, max_value=200.0, value=40.0, step=5.0, help='Labor days per hectare')
        distance = st.slider(text(current_lang, 'distance'), min_value=0.0, max_value=100.0, value=12.0, step=1.0, help='Distance to nearest market in km')
        improved_seed = st.selectbox(text(current_lang, 'improved_seed'), [text(current_lang, 'no'), text(current_lang, 'yes')])
        pest_flag = st.selectbox(text(current_lang, 'pest_flag'), [text(current_lang, 'no'), text(current_lang, 'yes')])

    price_frame = prices.copy()
    price_frame['__region_norm'] = price_frame['region'].map(normalize_region_name)
    price_frame['__crop_norm'] = price_frame['crop_type'].map(normalize_crop_name)
    price_row = price_frame[
        (price_frame['__region_norm'] == normalize_region_name(region))
        & (price_frame['__crop_norm'] == normalize_crop_name(crop))
        & (price_frame['year'] == survey_year)
    ].copy()
    price_row = price_row.dropna(subset=['price_birr_per_quintal'])

    if price_row.empty:
        regional_crop = price_frame[
            (price_frame['__region_norm'] == normalize_region_name(region))
            & (price_frame['__crop_norm'] == normalize_crop_name(crop))
        ].copy()
        if regional_crop.empty:
            regional_crop = price_frame[
                (price_frame['__crop_norm'] == normalize_crop_name(crop))
            ].copy()
        if regional_crop.empty:
            regional_crop = price_frame.copy()
        price_value = float(regional_crop['price_birr_per_quintal'].median()) if not regional_crop.empty else 0.0
    else:
        price_value = float(price_row['price_birr_per_quintal'].mean())

    feature_df = build_feature_row(
        region, crop, survey_year, planting_month, altitude, farm_size, fertilizer,
        yes_to_int(improved_seed), yes_to_int(pest_flag), soil_quality, labor, distance, weather
    )

    with st.spinner('Computing yield forecast...'):
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
    with col1:
        st.metric(text(current_lang, 'predicted_yield'), f'{prediction:.2f} t/ha')
    with col2:
        st.metric(text(current_lang, 'estimated_revenue'), f'{revenue:,.0f} birr')
    with col3:
        st.metric(text(current_lang, 'market_price'), f'{price_value:,.0f} birr/quintal')

    st.markdown('---')

    with st.expander(text(current_lang, 'farm_assumptions'), expanded=True):
        st.markdown(f"**{text(current_lang, 'region')}: {region_label(current_lang, region)}**")
        st.markdown(f"**{text(current_lang, 'crop_type')}: {crop_label(current_lang, crop)}**")
        st.markdown(f"**{text(current_lang, 'survey_year')}: {survey_year}**")
        st.markdown(f"**{text(current_lang, 'planting_month')}: {planting_month}**")
        st.markdown(f"**{text(current_lang, 'altitude')}: {altitude:,.0f} m**")
        st.markdown(f"**{text(current_lang, 'farm_size')}: {farm_size:.1f} ha**")
        st.markdown(f"**{text(current_lang, 'fertilizer')}: {fertilizer:.1f} kg/ha**")
        st.markdown(f"**{text(current_lang, 'soil_quality')}: {soil_quality:.2f}**")
        st.markdown(f"**{text(current_lang, 'improved_seed')}: {text(current_lang, 'yes') if improved_seed == text(current_lang, 'yes') else text(current_lang, 'no')}**")
        st.markdown(f"**{text(current_lang, 'pest_flag')}: {text(current_lang, 'yes') if pest_flag == text(current_lang, 'yes') else text(current_lang, 'no')}**")

    with st.expander(text(current_lang, 'season_outlook'), expanded=True):
        seasonal_data = [
            {text(current_lang, 'mean_temp'): f'{season["season_mean_temp"]:.1f} °C', 'raw': season["season_mean_temp"]},
            {text(current_lang, 'rainfall_total'): f'{season["season_rainfall_total"]:.1f} mm', 'raw': season["season_rainfall_total"]},
            {text(current_lang, 'rainfall_mean'): f'{season["season_rainfall_mean"]:.1f} mm', 'raw': season["season_rainfall_mean"]},
            {text(current_lang, 'extreme_heat_days'): f'{season["season_extreme_heat_days"]:.0f}', 'raw': season["season_extreme_heat_days"]},
            {text(current_lang, 'temp_variability'): f'{season["season_temp_std"]:.1f} °C', 'raw': season["season_temp_std"]},
        ]
        seasonal_df = pd.DataFrame(seasonal_data).drop(columns=['raw'])
        st.dataframe(seasonal_df, hide_index=True, use_container_width=True)

    st.markdown('---')
    st.subheader(text(current_lang, 'yield_comparison'))
    comparison = pd.DataFrame({
        text(current_lang, 'yield_t_ha'): [prediction, 2.4],
    }, index=[text(current_lang, 'model_forecast'), text(current_lang, 'regional_baseline')])
    st.bar_chart(comparison, height=300, use_container_width=True)

    st.caption(text(current_lang, 'caption'))


if __name__ == '__main__':
    main()
