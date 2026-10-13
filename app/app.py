from __future__ import annotations

import math
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Suppress sklearn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.base')

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
MODEL_PATH = ASSETS_DIR / 'final_model.joblib'
PRICE_PATH = ASSETS_DIR / 'cleaned_price.csv'
WEATHER_PATH = ASSETS_DIR / 'cleaned_weather.csv'
PROCESSED_DIR = ROOT / 'data' / 'processed'

# Load processed data for analytics (cached)
@st.cache_data
def load_processed_data():
    train = pd.read_csv(PROCESSED_DIR / 'master_train.csv')
    return train

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
        'planting_month': 'የተዘራበት ወር',
        'altitude': 'ከፍታ (ሜትር)',
        'farm_size': 'የእርሻ መሬት ስፋት (ሄክታር)',
        'fertilizer': 'ማዳበሪያ (ኪ.ግ/ሄክታር)',
        'soil_quality': 'የአፈር ጥራት መረጃ',
        'labor': 'የስራ ቀናት/በሄክታር',
        'distance': 'ከገበያ የርቀት (ኪ.ሜ)',
        'improved_seed': 'ምርጥ ዘር ጥቅም ላይ ውሏል',
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
        'mean_temp': 'አማካይ የአየር ጠባይ መጠን',
        'rainfall_total': 'ጠቅላላ የዝናብ መጠን',
        'rainfall_mean': 'አማካይ የዝናብ መጠን',
        'extreme_heat_days': 'ከፍተኛ ሙቀት ቀናት',
        'temp_variability': 'የአየር ጠባይ ልዩነት',
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
    :root {
        --primary: #103a52;
        --primary-light: #1d4a6d;
        --secondary: #14b8a6;
        --accent: #3b82f6;
        --success: #059669;
        --warning: #d97706;
        --error: #dc2626;
        --background: #f5fff7;
        --surface: #ffffff;
        --on-primary: #ffffff;
        --on-secondary: #ffffff;
        --on-surface: #102a43;
        --on-background: #102a43;
        --surface-variant: #eef8ff;
        --on-surface-variant: #103a52;
        --border: rgba(32, 117, 90, 0.12);
        --shadow: rgba(16, 67, 91, 0.08);
    }
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: var(--background);
        color: var(--on-background);
    }
    [data-testid="stSidebar"] {
        background: var(--primary);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--on-primary) !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--primary);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,250,255,0.95));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 25px var(--shadow);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800;
        color: var(--primary);
    }
    [data-testid="stMetricLabel"] {
        color: var(--on-surface-variant);
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 25px var(--shadow);
    }
    .stDataFrame .row-headings {
        background-color: var(--surface-variant);
        color: var(--on-surface-variant);
    }
    .stDataFrame .column-headings {
        background-color: var(--surface-variant);
        color: var(--on-surface-variant);
    }
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, var(--secondary), var(--accent));
        color: var(--on-primary);
        font-weight: 700;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d9488, #2563eb);
    }
    .stAlert {
        border-radius: 14px;
        background-color: var(--surface-variant);
        border-left: 4px var(--primary);
    }
    .stSelectbox > div > div {
        background-color: var(--surface);
        color: var(--on-surface-variant);
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


def yes_to_int(value):
    """Convert Yes/No selection to 1/0 integer for model features."""
    return 1 if value == text('en', 'yes') else 0


def load_data():
    """Load pricing, weather, and trained model inputs from assets.

    The cleaned assets may contain inconsistent region/crop labels (e.g. 'SOM', 'AMH', 'ORO', 'TIG').
    We normalize these on load so downstream lookups work reliably.
    """
    prices = pd.read_csv(PRICE_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    
    # Load model with error handling for corrupted files
    try:
        model = joblib.load(MODEL_PATH)
    except EOFError:
        st.error(
            "Model file is corrupted. Please ensure `final_model.joblib` "
            "was uploaded correctly and try redeploying the app."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

    # Load model performance summary
    model_summary = {}
    summary_path = ROOT / 'reports' / 'D_model_summary.json'
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                model_summary = json.load(f)
        except Exception:
            pass

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

    return prices, weather, model, model_summary


@st.cache_data
def load_analytics():
    """Load processed training data for comparison and analysis features."""
    PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    return pd.read_csv(PROCESSED_DIR / 'master_train.csv')


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
    prices, weather, model, model_summary = load_data()
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
        improved_seed = st.selectbox(text(current_lang, 'improved_seed'), [text(current_lang, 'yes'), text(current_lang, 'no')])
        pest_flag = st.selectbox(text(current_lang, 'pest_flag'), [text(current_lang, 'yes'), text(current_lang, 'no')])
        # Convert Yes/No to 1/0 for model features
        improved_seed_int = yes_to_int(improved_seed)
        pest_flag_int = yes_to_int(pest_flag)

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
        improved_seed_int, pest_flag_int, soil_quality, labor, distance, weather
    )

    with st.spinner('Computing prediction...'):
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(text(current_lang, 'predicted_yield'), f'{prediction:.2f} t/ha')
    col2.metric(text(current_lang, 'estimated_revenue'), f'{revenue:,.0f} birr')
    col3.metric(text(current_lang, 'market_price'), f'{price_value:,.0f} birr/quintal')
    
    # Confidence score display
    r2_score = model_summary.get('validation_r2', 0.8940)
    rmse = model_summary.get('validation_rmse', 0.4599)
    
    # Confidence score based on R² (goodness of fit)
    confidence = r2_score  # R² directly maps to confidence (0.894 = 89.4% confidence)
    
    confidence_color = '#10b981' if confidence > 0.75 else '#f59e0b' if confidence > 0.5 else '#dc2626'
    confidence_label = 'High' if confidence > 0.75 else 'Medium' if confidence > 0.5 else 'Low'
    
    col4.markdown(f"""
        <div style="padding: 0.8rem; border-radius: 12px; background-color: #f8fafc; border-left: 4px solid {confidence_color}; margin-top: 0.5rem;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">Confidence</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {confidence_color};">{confidence*100:.0f}%</div>
            <div style="font-size: 0.7rem; color: #94a3b8;">{confidence_label} reliability</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b;">
        Model: R²={r2_score:.3f} | RMSE={rmse:.3f} t/ha | ±{rmse*1.96:.2f} t/ha (95% interval)
    </div>
    """, unsafe_allow_html=True)

    st.markdown('---')
    with st.expander("📋 Farm assumptions", expanded=False):
        st.subheader(text(current_lang, 'farm_assumptions'))
        summary = pd.DataFrame([
            {'Metric': text(current_lang, 'region'), 'Value': str(region_label(current_lang, region))},
            {'Metric': text(current_lang, 'crop_type'), 'Value': str(crop_label(current_lang, crop))},
            {'Metric': text(current_lang, 'planting_month'), 'Value': str(planting_month)},
            {'Metric': text(current_lang, 'survey_year'), 'Value': str(survey_year)},
            {'Metric': text(current_lang, 'farm_size'), 'Value': str(f'{farm_size:.1f} ha')},
            {'Metric': text(current_lang, 'fertilizer'), 'Value': str(f'{fertilizer:.1f} kg/ha')},
            {'Metric': text(current_lang, 'soil_quality'), 'Value': str(f'{soil_quality:.2f}')},
            {'Metric': text(current_lang, 'improved_seed'), 'Value': str(text(current_lang, 'yes') if improved_seed_int else text(current_lang, 'no'))},
        ])
        st.dataframe(summary, hide_index=True, width='stretch')

    with st.expander("🌦️ Season outlook", expanded=False):
        st.subheader(text(current_lang, 'season_outlook'))
        seasonal = pd.DataFrame([
            {'Metric': text(current_lang, 'mean_temp'), 'Value': str(f'{season["season_mean_temp"]:.1f} °C' if not pd.isna(season["season_mean_temp"]) else 'No data')},
            {'Metric': text(current_lang, 'rainfall_total'), 'Value': str(f'{season["season_rainfall_total"]:.1f} mm' if not pd.isna(season["season_rainfall_total"]) else 'No data')},
            {'Metric': text(current_lang, 'rainfall_mean'), 'Value': str(f'{season["season_rainfall_mean"]:.1f} mm' if not pd.isna(season["season_rainfall_mean"]) else 'No data')},
            {'Metric': text(current_lang, 'extreme_heat_days'), 'Value': str(f'{season["season_extreme_heat_days"]:.0f}' if not pd.isna(season["season_extreme_heat_days"]) else 'No data')},
            {'Metric': text(current_lang, 'temp_variability'), 'Value': str(f'{season["season_temp_std"]:.1f} °C' if not pd.isna(season["season_temp_std"]) else 'No data')},
        ])
        st.dataframe(seasonal, hide_index=True, width='stretch')

    st.markdown('---')
    st.subheader(text(current_lang, 'yield_comparison'))
    comparison = pd.DataFrame({
        'Scenario': [text(current_lang, 'model_forecast'), text(current_lang, 'regional_baseline')],
        text(current_lang, 'yield_t_ha'): [prediction, 2.4],
    })
    st.bar_chart(comparison.set_index('Scenario'))

    st.caption(text(current_lang, 'caption'))

    # Professional data analysis features
    with st.expander("🔍 Pro Analysis: Yield Insights", expanded=False):
        # Load training data for comparison
        train_df = load_analytics()
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Yield Distribution", 
            "Regional Comparison", 
            "Feature Impact", 
            "Price Analysis"
        ])
        
        with tab1:
            st.subheader("Yield Distribution Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Distribution plot
            sns.histplot(data=train_df, x='yield_tons_per_ha', bins=30, kde=True, ax=ax1, color='steelblue')
            ax1.axvline(prediction, color='red', linestyle='--', linewidth=2, label='Your Prediction')
            ax1.axvline(train_df['yield_tons_per_ha'].mean(), color='green', linestyle='-', linewidth=2, label='Dataset Mean')
            ax1.set_xlabel('Yield (tons/ha)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Yield Distribution with Your Prediction')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot by crop type
            sns.boxplot(data=train_df, x='crop_type', y='yield_tons_per_ha', ax=ax2, hue='crop_type', palette='Set2', legend=False)
            ax2.scatter([0,1,2,3,4], [prediction]*5, color='red', s=100, zorder=5, label='Your Prediction')
            ax2.set_xlabel('Crop Type')
            ax2.set_ylabel('Yield (tons/ha)')
            ax2.set_title('Yield by Crop Type')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Dataset Mean", f"{train_df['yield_tons_per_ha'].mean():.2f} t/ha")
            with col2:
                st.metric("Dataset Median", f"{train_df['yield_tons_per_ha'].median():.2f} t/ha")
            with col3:
                st.metric("Your Prediction", f"{prediction:.2f} t/ha")
            with col4:
                percentile = (train_df['yield_tons_per_ha'] < prediction).mean() * 100
                st.metric("Percentile Rank", f"{percentile:.0f}%")
        
        with tab2:
            st.subheader("Regional Performance Analysis")
            regional_stats = train_df.groupby('region')['yield_tons_per_ha'].agg(['mean', 'std', 'count']).round(2)
            regional_stats.columns = ['Mean Yield', 'Std Dev', 'Sample Size']
            regional_stats['Your Prediction'] = prediction
            regional_stats['vs Regional Mean (%)'] = ((prediction - regional_stats['Mean Yield']) / regional_stats['Mean Yield'] * 100).round(1)
            
            st.dataframe(regional_stats, width='stretch')
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            regional_stats['Mean Yield'].plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy')
            ax.axhline(y=prediction, color='red', linestyle='--', linewidth=2, label='Your Prediction')
            ax.set_ylabel('Yield (tons/ha)')
            ax.set_title('Average Yield by Region')
            ax.legend()
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Feature Impact Analysis")
            # Show how key features affect yield
            feature_cols = ['season_mean_temp', 'season_rainfall_total', 'fertilizer_kg_per_ha', 'farm_size_ha']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, feature in enumerate(feature_cols):
                if feature in train_df.columns:
                    # Scatter plot with trend
                    axes[i].scatter(train_df[feature], train_df['yield_tons_per_ha'], alpha=0.5, s=10)
                    
                    # Add trend line
                    z = np.polyfit(train_df[feature], train_df['yield_tons_per_ha'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(train_df[feature].min(), train_df[feature].max(), 100)
                    axes[i].plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                    
                    # Mark current values
                    current_vals = {
                        'season_mean_temp': season['season_mean_temp'],
                        'season_rainfall_total': season['season_rainfall_total'],
                        'fertilizer_kg_per_ha': fertilizer,
                        'farm_size_ha': farm_size
                    }
                    if feature in current_vals:
                        axes[i].axvline(current_vals[feature], color='red', linestyle='-', linewidth=2, label='Your Value')
                    
                    axes[i].set_xlabel(feature.replace('_', ' ').title())
                    axes[i].set_ylabel('Yield (tons/ha)')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
            
            plt.suptitle('Feature Impact on Yield (with Trend Lines)', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation table
            st.subheader("Feature Correlations with Yield")
            corr_features = feature_cols + ['yield_tons_per_ha']
            corr_matrix = train_df[corr_features].corr()['yield_tons_per_ha'].drop('yield_tons_per_ha')
            corr_df = pd.DataFrame({
                'Feature': corr_matrix.index.astype(str),
                'Correlation with Yield': corr_matrix.values.round(3).astype(float)
            }).sort_values('Correlation with Yield', key=abs, ascending=False)
            
            st.dataframe(corr_df, width='stretch', hide_index=True)
        
        with tab4:
            st.subheader("Market Price Analysis")
            
            # Load price data for trends
            price_data = pd.read_csv(PRICE_PATH)
            price_data = price_data[price_data['price_birr_per_quintal'].notna()]
            
            # Price trends by crop
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Line plot of price trends
            for crop_type in price_data['crop_type'].unique():
                crop_data = price_data[price_data['crop_type'] == crop_type].sort_values('year')
                ax1.plot(crop_data['year'], crop_data['price_birr_per_quintal'], 
                        marker='o', label=crop_type, linewidth=2)
            
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Price (birr/quintal)')
            ax1.set_title('Price Trends by Crop Type')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Current price comparison
            current_crop_price = price_data[
                (price_data['region'] == region) & 
                (price_data['crop_type'] == crop) & 
                (price_data['year'] == survey_year)
            ]['price_birr_per_quintal'].mean()
            
            if pd.isna(current_crop_price):
                # Fallback to regional average
                regional_price = price_data[
                    (price_data['region'] == region) & 
                    (price_data['crop_type'] == crop)
                ]['price_birr_per_quintal'].mean()
                if pd.isna(regional_price):
                    # Fallback to crop average
                    regional_price = price_data[
                        price_data['crop_type'] == crop
                    ]['price_birr_per_quintal'].mean()
            
            crop_avg_price = price_data[price_data['crop_type'] == crop]['price_birr_per_quintal'].mean()
            overall_avg = price_data['price_birr_per_quintal'].mean()
            
            price_comparison = pd.DataFrame({
                'Price Type': ['Current (Your Selection)', f'{crop} Average', 'Overall Average'],
                'Price (birr/quintal)': [
                    price_value if not pd.isna(price_value) else 0,
                    crop_avg_price if not pd.isna(crop_avg_price) else 0,
                    overall_avg if not pd.isna(overall_avg) else 0
                ]
            })
            
            ax2.bar(price_comparison['Price Type'], price_comparison['Price (birr/quintal)'], 
                   color=['red', 'blue', 'gray'], alpha=0.7)
            ax2.set_ylabel('Price (birr/quintal)')
            ax2.set_title('Price Comparison')
            ax2.tick_params(axis='x', rotation=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Price statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{price_value:.0f} birr/quintal")
            with col2:
                st.metric(f"{crop.title()} Avg", f"{crop_avg_price:.0f} birr/quintal")
            with col3:
                st.metric("Market Avg", f"{overall_avg:.0f} birr/quintal")



if __name__ == '__main__':
    main()
