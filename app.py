import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import base64

# 1. Page Configuration
st.set_page_config(page_title="Vehicle Carbon Footprint Predictor", page_icon="🚗", layout="wide")

# Base64 background image helper
def get_base64_bg():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(base_dir, 'bg_minimal.png')
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception:
            return ""
    return ""

bg_base64 = get_base64_bg()

# Inject CSS styles
if bg_base64:
    bg_style = f"""
    body {{
        background-color: #0b0c0e !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg_base64}") !important;
        background-size: 115% 115% !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        animation: windBg 40s ease-in-out infinite alternate !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    """
else:
    bg_style = """
    body {
        background-color: #0b0c0e !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

{bg_style}

@keyframes windBg {{
    0% {{
        background-position: 0% 50%;
    }}
    50% {{
        background-position: 100% 50%;
    }}
    100% {{
        background-position: 50% 100%;
    }}
}}

/* Hide Streamlit footer and main menu */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stHeader"] {{background: transparent !important;}}

/* Centered container styling */
[data-testid="stMainBlockContainer"] {{
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
}}

/* Header style elements */
.badge-container {{
    text-align: center;
    margin-bottom: 12px;
}}
.enterprise-badge {{
    background-color: rgba(26, 26, 30, 0.6) !important;
    color: #8b9ffb;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    display: inline-block;
    text-transform: uppercase;
    backdrop-filter: blur(8px) !important;
}}
.main-title {{
    text-align: center;
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
    letter-spacing: -0.03em;
}}
.main-subtitle {{
    text-align: center;
    color: #8a8d98;
    font-size: 0.95rem;
    max-width: 650px;
    margin: 0 auto 35px auto;
    line-height: 1.6;
}}
.main-subtitle strong {{
    color: #8b9ffb;
    font-weight: 600;
}}

/* Card titles */
.card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 24px;
}}
.card-header svg {{
    width: 20px;
    height: 20px;
    fill: #8b9ffb;
}}

/* Glassmorphism card container */
[data-testid="stForm"] {{
    background-color: rgba(20, 20, 24, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    padding: 28px !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(20px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(140%) !important;
}}

/* Remove default form border and padding */
[data-testid="stForm"] > div {{
    padding: 0 !important;
}}

/* Input labels */
.stSelectbox label, .stNumberInput label, .stSlider label {{
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #8A8D98 !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}}

/* Selectbox glass styling */
div[data-baseweb="select"] > div {{
    background-color: rgba(10, 10, 12, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 14px !important;
    height: 42px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.25s ease !important;
}}
div[data-baseweb="select"] {{
    border-radius: 8px !important;
}}
div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {{
    border-color: rgba(139, 159, 251, 0.5) !important;
    box-shadow: 0 0 10px rgba(139, 159, 251, 0.3) !important;
}}

/* Dropdown list styling */
ul[role="listbox"] {{
    background-color: #121212 !important;
    border: 1px solid #2d2d2d !important;
}}
li[role="option"] {{
    color: #ffffff !important;
}}
li[role="option"]:hover {{
    background-color: #262626 !important;
}}

/* Slider styling - stay grey track & glowing handle */
div[data-testid="stSlider"] [data-testid="stSliderTrack"] {{
    background: #2d2d2d !important;
}}
div[data-testid="stSlider"] [data-testid="stSliderTrack"] > div {{
    background: #2d2d2d !important;
}}
div[data-testid="stSlider"] [data-testid="stSliderTrack"] > div > div {{
    background: #2d2d2d !important;
}}
div[data-testid="stSlider"] [class*="StyledTrack"] {{
    background: #2d2d2d !important;
}}
div[data-testid="stSlider"] [class*="StyledTrack"] > div {{
    background: #2d2d2d !important;
}}
div[data-testid="stSlider"] [role="slider"] {{
    background-color: #8b9ffb !important;
    border: 2px solid #8b9ffb !important;
    width: 14px !important;
    height: 14px !important;
    box-shadow: 0 0 10px rgba(139, 159, 251, 0.8) !important;
    transition: box-shadow 0.25s ease, transform 0.2s !important;
}}
div[data-testid="stSlider"] [role="slider"]:hover, div[data-testid="stSlider"] [role="slider"]:active {{
    transform: scale(1.1) !important;
    box-shadow: 0 0 15px rgba(139, 159, 251, 1.0) !important;
}}

/* Number input custom layout: [- value +] */
div[data-testid="stNumberInputContainer"] {{
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    background-color: rgba(10, 10, 12, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    height: 42px !important;
    overflow: hidden !important;
    box-shadow: none !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.25s ease !important;
}}
div[data-testid="stNumberInputContainer"]:hover, div[data-testid="stNumberInputContainer"]:focus-within {{
    border-color: rgba(139, 159, 251, 0.5) !important;
    box-shadow: 0 0 10px rgba(139, 159, 251, 0.3) !important;
}}
div[data-testid="stNumberInputContainer"] div[data-baseweb="input"] {{
    border: none !important;
    background-color: transparent !important;
    width: 100% !important;
    height: 100% !important;
}}
div[data-testid="stNumberInputContainer"] div[data-baseweb="base-input"] {{
    border: none !important;
    background-color: transparent !important;
    width: 100% !important;
    height: 100% !important;
}}
div[data-testid="stNumberInputContainer"] input[data-testid="stNumberInputField"] {{
    text-align: center !important;
    background-color: transparent !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 14px !important;
    height: 100% !important;
    padding-left: 45px !important;
    padding-right: 45px !important;
    width: 100% !important;
}}
/* Absolute buttons container */
div[data-testid="stNumberInputContainer"] > div:last-child {{
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    pointer-events: none !important;
    display: block !important;
    background: transparent !important;
    border: none !important;
}}
/* Style individual buttons */
button[data-testid="stNumberInputStepDown"] {{
    pointer-events: auto !important;
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    bottom: 0 !important;
    width: 40px !important;
    height: 100% !important;
    background-color: transparent !important;
    border: none !important;
    color: #ffffff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}}
button[data-testid="stNumberInputStepUp"] {{
    pointer-events: auto !important;
    position: absolute !important;
    right: 0 !important;
    top: 0 !important;
    bottom: 0 !important;
    width: 40px !important;
    height: 100% !important;
    background-color: transparent !important;
    border: none !important;
    color: #ffffff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}}
button[data-testid="stNumberInputStepDown"]:hover, button[data-testid="stNumberInputStepUp"]:hover {{
    background-color: transparent !important;
}}
button[data-testid="stNumberInputStepDown"]:hover svg, button[data-testid="stNumberInputStepUp"]:hover svg {{
    fill: #8b9ffb !important;
    filter: drop-shadow(0 0 4px rgba(139, 159, 251, 0.8)) !important;
    transform: scale(1.15) !important;
}}
button[data-testid="stNumberInputStepDown"] svg, button[data-testid="stNumberInputStepUp"] svg {{
    fill: #ffffff !important;
    width: 10px !important;
    height: 10px !important;
    transition: all 0.2s ease !important;
}}

/* Submit Button styling - Glowing lavender-blue */
div[data-testid="stFormSubmitButton"] button {{
    background-color: #8b9ffb !important;
    color: #0e0e0e !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    padding: 12px 24px !important;
    margin-top: 15px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    height: 46px !important;
    animation: pulseGlow 3s infinite ease-in-out !important;
}}
div[data-testid="stFormSubmitButton"] button:hover {{
    background-color: #a3b4fc !important;
    transform: translateY(-2px) !important;
    animation: none !important;
    box-shadow: 0 0 25px rgba(139, 159, 251, 1.0) !important;
    color: #0e0e0e !important;
}}
div[data-testid="stFormSubmitButton"] button:active {{
    transform: translateY(0) !important;
}}
div[data-testid="stFormSubmitButton"] button:before {{
    content: "" !important;
    display: inline-block !important;
    width: 16px !important;
    height: 16px !important;
    margin-right: 6px !important;
    background-color: #0e0e0e !important;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4.5 16.5c-1.5 1.25-2.5 3.5-2.5 3.5s2.25-1 3.5-2.5M19.5 7.5c1.5-1.25 2.5-3.5 2.5-3.5s-2.25 1-3.5 2.5M14 2s2 2 3.5 4.5M2 10s2 2 4.5 3.5M12 12l9-9-3 12-4 1-2-2-2-2 1-4-9-3z'/%3E%3Cpath d='M9 15l-3 3M15 9l3-3'/%3E%3C/svg%3E") no-repeat center !important;
    -webkit-mask-size: contain !important;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4.5 16.5c-1.5 1.25-2.5 3.5-2.5 3.5s2.25-1 3.5-2.5M19.5 7.5c1.5-1.25 2.5-3.5 2.5-3.5s-2.25 1-3.5 2.5M14 2s2 2 3.5 4.5M2 10s2 2 4.5 3.5M12 12l9-9-3 12-4 1-2-2-2-2 1-4-9-3z'/%3E%3Cpath d='M9 15l-3 3M15 9l3-3'/%3E%3C/svg%3E") no-repeat center !important;
    mask-size: contain !important;
}}

@keyframes pulseGlow {{
    0% {{
        box-shadow: 0 0 12px rgba(139, 159, 251, 0.4);
    }}
    50% {{
        box-shadow: 0 0 20px rgba(139, 159, 251, 0.7);
    }}
    100% {{
        box-shadow: 0 0 12px rgba(139, 159, 251, 0.4);
    }}
}}

/* Hide Streamlit slider floating red number value */
div[data-testid="stSliderThumbValue"] {{
    display: none !important;
}}

/* Glassmorphism Result Card styling */
.result-card {{
    background-color: rgba(20, 20, 24, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    padding: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    justify-content: center;
    min-height: 440px;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(20px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(140%) !important;
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

/* Circular Progress Gauge */
.gauge-container {{
    position: relative;
    width: 180px;
    height: 180px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.gauge {{
    width: 100%;
    height: 100%;
    transform: rotate(-90deg);
}}
.gauge-bg {{
    fill: none;
    stroke: rgba(255, 255, 255, 0.04);
    stroke-width: 6;
}}
.gauge-progress {{
    fill: none;
    stroke-width: 6;
    stroke-linecap: round;
    transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    animation: progressAnim 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}}
.gauge-text {{
    position: absolute;
    text-align: center;
}}
.gauge-value {{
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}}
.gauge-unit {{
    font-size: 0.8rem;
    color: #8a8d98;
    margin-top: 4px;
    font-weight: 500;
}}

/* Footprint Text styling */
.footprint-category {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 24px;
    margin-bottom: 8px;
}}
.footprint-desc {{
    font-size: 0.85rem;
    color: #8a8d98;
    line-height: 1.5;
    max-width: 280px;
    margin-bottom: 24px;
}}

/* Climate Pill styling */
.climate-pill {{
    background-color: rgba(10, 10, 12, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    max-width: 320px;
}}
.climate-left {{
    display: flex;
    align-items: center;
    gap: 12px;
    text-align: left;
}}
.climate-icon-container {{
    display: flex;
    align-items: center;
    justify-content: center;
}}
.climate-icon {{
    width: 16px;
    height: 16px;
}}
.climate-label {{
    font-size: 8px;
    font-weight: 700;
    color: #8a8d98;
    letter-spacing: 0.08em;
}}
.climate-value {{
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
}}
.climate-right {{
    color: #8a8d98;
    display: flex;
    align-items: center;
}}
.share-icon {{
    width: 16px;
    height: 16px;
}}

/* Glassmorphism SDG Card styling */
.sdg-card {{
    background-color: rgba(20, 20, 24, 0.6) !important;
    border: 1px solid rgba(16, 185, 129, 0.15) !important;
    border-radius: 16px;
    padding: 16px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-top: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(15px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(15px) saturate(120%) !important;
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.15s forwards;
    opacity: 0;
}}
.sdg-icon-container {{
    background-color: rgba(17, 37, 25, 0.6) !important;
    color: #10b981;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.sdg-icon {{
    width: 20px;
    height: 20px;
}}
.sdg-content {{
    text-align: left;
}}
.sdg-title {{
    font-size: 13px;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 3px;
}}
.sdg-desc {{
    font-size: 11px;
    color: #8a8d98;
    line-height: 1.5;
}}

/* Animations */
@keyframes progressAnim {{
    from {{
        stroke-dashoffset: 314;
    }}
}}
@keyframes fadeIn {{
    from {{
        opacity: 0;
        transform: translateY(12px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}
</style>
""", unsafe_allow_html=True)

# 2. Load Model & Encoders
@st.cache_resource
def load_assets():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'co2_xgb_model.pkl')
        encoders_path = os.path.join(base_dir, 'label_encoders.pkl')
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        return model, encoders
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan! Pastikan file model dan encoder ada di satu folder.")
        st.stop()

model, encoders = load_assets()

# 3. Mappings & Helper Data
make_options = encoders['Make'].classes_
vehicle_options = encoders['Vehicle Class'].classes_
trans_options = encoders['Transmission'].classes_

fuel_map = {
    "Regular Gasoline (X)": "X",
    "Premium Gasoline (Z)": "Z",
    "Diesel (D)": "D",
    "Ethanol (E)": "E",
    "Natural Gas (N)": "N"
}
fuel_display_map = {v: k for k, v in fuel_map.items()}

trans_map = {
    "A10": "A10 (Automatic 10-speed)",
    "A4": "A4 (Automatic 4-speed)",
    "A5": "A5 (Automatic 5-speed)",
    "A6": "A6 (Automatic 6-speed)",
    "A7": "A7 (Automatic 7-speed)",
    "A8": "A8 (Automatic 8-speed)",
    "A9": "A9 (Automatic 9-speed)",
    "AM5": "AM5 (Automated Manual 5-speed)",
    "AM6": "AM6 (Automated Manual 6-speed)",
    "AM7": "AM7 (Automated Manual 7-speed)",
    "AM8": "AM8 (Automated Manual 8-speed)",
    "AM9": "AM9 (Automated Manual 9-speed)",
    "AS4": "AS4 (Automatic with Select Shift 4-speed)",
    "AS5": "AS5 (Automatic with Select Shift 5-speed)",
    "AS6": "AS6 (Automatic with Select Shift 6-speed)",
    "AS7": "AS7 (Automatic with Select Shift 7-speed)",
    "AS8": "AS8 (Automatic with Select Shift 8-speed)",
    "AS9": "AS9 (Automatic with Select Shift 9-speed)",
    "AS10": "AS10 (Automatic with Select Shift 10-speed)",
    "AV": "AV (Continuously Variable)",
    "AV6": "AV6 (Continuously Variable 6-speed)",
    "AV7": "AV7 (Continuously Variable 7-speed)",
    "AV8": "AV8 (Continuously Variable 8-speed)",
    "AV10": "AV10 (Continuously Variable 10-speed)",
    "M5": "M5 (Manual 5-speed)",
    "M6": "M6 (Manual 6-speed)",
    "M7": "M7 (Manual 7-speed)"
}

def get_index(options, target):
    try:
        return list(options).index(target)
    except ValueError:
        return 0

def predict_co2(make, vehicle_class, transmission, fuel_type_code, engine_size, cylinders, fuel_comb):
    make_enc = encoders['Make'].transform([make])[0]
    vehicle_enc = encoders['Vehicle Class'].transform([vehicle_class])[0]
    trans_enc = encoders['Transmission'].transform([transmission])[0]
    fuel_enc = encoders['Fuel Type'].transform([fuel_type_code])[0]
    
    input_data = [[make_enc, vehicle_enc, trans_enc, fuel_enc, engine_size, cylinders, fuel_comb]]
    prediction = model.predict(input_data)[0]
    return prediction

def get_footprint_details(prediction):
    if prediction < 170:
        return {
            "category": "Low Footprint",
            "rating": "A+ Efficiency Rating",
            "color": "#10B981", # Green
            "desc": f"Your vehicle produces approximately {prediction:.0f} grams of CO2 per kilometer. This falls within the top 15% of fuel-efficient vehicles."
        }
    elif prediction < 240:
        return {
            "category": "Moderate Footprint",
            "rating": "B- Efficiency Rating",
            "color": "#79C1FC", # Cyan/Blue
            "desc": f"Your vehicle produces approximately {prediction:.0f} grams of CO2 per kilometer. This falls within the top 40% of fuel-efficient vehicles in its class."
        }
    elif prediction < 300:
        return {
            "category": "High Footprint",
            "rating": "C Efficiency Rating",
            "color": "#F59E0B", # Orange
            "desc": f"Your vehicle produces approximately {prediction:.0f} grams of CO2 per kilometer. This vehicle is moderately polluting."
        }
    else:
        return {
            "category": "Severe Footprint",
            "rating": "F Efficiency Rating",
            "color": "#EF4444", # Red
            "desc": f"Your vehicle produces approximately {prediction:.0f} grams of CO2 per kilometer. This vehicle has very high emissions."
        }

# 4. Header Section
st.markdown('<div class="badge-container"><span class="enterprise-badge">Enterprise Grade</span></div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Vehicle Carbon Footprint Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Harnessing advanced <strong>XGBoost Regression Analysis</strong> to deliver precision metrics for SDG #13 Climate Action compliance and environmental transparency.</p>', unsafe_allow_html=True)

# 5. Main Layout
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    with st.form("prediction_form", clear_on_submit=False):
        # Header inside specification card
        st.markdown('''
        <div class="card-header">
            <svg viewBox="0 0 24 24">
                <path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/>
            </svg>
            Vehicle Specifications
        </div>
        ''', unsafe_allow_html=True)
        
        # Row 1: Car Brand & Vehicle Class
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            make = st.selectbox("Car Brand", options=make_options, index=get_index(make_options, "ACURA"))
        with r1_c2:
            vehicle_class = st.selectbox("Vehicle Class", options=vehicle_options, index=get_index(vehicle_options, "COMPACT"))
            
        # Row 2: Transmission Type & Fuel Type
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            transmission = st.selectbox("Transmission Type", options=trans_options, format_func=lambda x: trans_map.get(x, x), index=get_index(trans_options, "A10"))
        with r2_c2:
            fuel_type_code = st.selectbox("Fuel Type", options=encoders['Fuel Type'].classes_, format_func=lambda x: fuel_display_map.get(x, x), index=get_index(encoders['Fuel Type'].classes_, "X"))
            
        # Row 3: Engine Size & Cylinders
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.00, step=0.1, format="%.2f")
        with r3_c2:
            cylinders_val = st.session_state.get("cylinders_val", 4)
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8A8D98; font-weight: 700; margin-bottom: 0px;">Cylinders</span>
                <span style="font-size: 0.85rem; color: #8b9ffb; font-weight: 700;">{cylinders_val}</span>
            </div>
            ''', unsafe_allow_html=True)
            cylinders = st.slider("Cylinders", min_value=3, max_value=16, value=4, key="cylinders_val", label_visibility="collapsed")
            
        # Row 4: Fuel Consumption
        fuel_comb = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=50.0, value=8.50, step=0.1, format="%.2f")
        
        submitted = st.form_submit_button("Calculate Emissions")

# 6. Make Prediction
prediction = predict_co2(
    make,
    vehicle_class,
    transmission,
    fuel_type_code,
    engine_size,
    cylinders,
    fuel_comb
)

details = get_footprint_details(prediction)
color = details["color"]
category = details["category"]
desc = details["desc"]
rating = details["rating"]

# Calculate circular gauge offset: circumference is 314
# Max CO2 scale is 500 for calculation
percentage = min(prediction / 500.0, 1.0)
dashoffset = 314.0 * (1.0 - percentage)

with col2:
    # Result Card (dynamic prediction)
    st.markdown(f"""
    <div class="result-card">
      <div class="gauge-container">
        <svg class="gauge" viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="50" />
          <circle class="gauge-progress" cx="60" cy="60" r="50" style="stroke-dasharray: 314; stroke-dashoffset: {dashoffset}; stroke: {color};" />
        </svg>
        <div class="gauge-text">
          <div class="gauge-value">{prediction:.0f}</div>
          <div class="gauge-unit">g/km CO2</div>
        </div>
      </div>
      
      <div class="footprint-category">{category}</div>
      <div class="footprint-desc">{desc}</div>
      
      <div class="climate-pill">
        <div class="climate-left">
          <div class="climate-icon-container" style="color: {color};">
            <svg class="climate-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill="currentColor"/>
            </svg>
          </div>
          <div>
            <div class="climate-label">CLIMATE IMPACT</div>
            <div class="climate-value">{rating}</div>
          </div>
        </div>
        <div class="climate-right">
          <svg class="share-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="18" cy="5" r="3"/>
            <circle cx="6" cy="12" r="3"/>
            <circle cx="18" cy="19" r="3"/>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
          </svg>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # SDG Card
    st.markdown("""
    <div class="sdg-card">
      <div class="sdg-icon-container">
        <svg class="sdg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
          <path d="M2 12h20"/>
        </svg>
      </div>
      <div class="sdg-content">
        <div class="sdg-title">SDG #13: Climate Action</div>
        <div class="sdg-desc">Every data point collected helps refine our global intelligence engine, enabling more accurate regional carbon mapping and sustainable policy recommendations.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)