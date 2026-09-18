import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="DELIVR — AI Delivery Time Prediction",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 2. INJECT CUSTOM OLED STYLING & ASSETS
# ==============================================================================
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")

def load_custom_css():
    """Load and inject custom OLED dark glassmorphic CSS."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
                body { background-color: #050505; color: #F5F5F5; font-family: sans-serif; }
            </style>
            """,
            unsafe_allow_html=True,
        )

load_custom_css()

# ==============================================================================
# 3. MODEL LOADING & CACHING
# ==============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_delivery_model.pkl")

@st.cache_resource(show_spinner=False)
def load_prediction_model():
    """Load the pre-trained scikit-learn Random Forest pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None, "Prediction model unavailable. Please check that food_delivery_model.pkl exists."
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model, None
    except Exception as e:
        return None, "Unable to load prediction model. Please ensure dependencies and model file are valid."

model, model_error = load_prediction_model()

# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
DEFAULT_STATE = {
    "distance": 8.5,
    "prep_time": 15,
    "courier_exp": 2.0,
    "weather": "Clear",
    "traffic": "Medium",
    "time_of_day": "Evening",
    "vehicle_type": "Bike",
    "prediction_result": None,
    "last_params": None,
    "validation_error": None,
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_parameters():
    """Reset input parameters and prediction results."""
    st.session_state.distance = 8.5
    st.session_state.prep_time = 15
    st.session_state.courier_exp = 2.0
    st.session_state.weather = "Clear"
    st.session_state.traffic = "Medium"
    st.session_state.time_of_day = "Evening"
    st.session_state.vehicle_type = "Bike"
    st.session_state.prediction_result = None
    st.session_state.last_params = None
    st.session_state.validation_error = None

# ==============================================================================
# 5. HEADER COMPONENT (ZERO INDENTATION TO PREVENT CODE BLOCK PARSING)
# ==============================================================================
badges_markup = (
    '<div class="status-badge"><span class="status-dot"></span><span>MODEL ONLINE</span></div>'
    '<div class="model-badge">Random Forest</div>'
    if model is not None else
    '<div class="status-badge" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.3);color:#F87171;">'
    '<span class="status-dot" style="background-color:#EF4444;box-shadow:0 0 8px #EF4444;"></span><span>OFFLINE</span></div>'
)

header_html = (
    '<div class="delivr-header">'
    '<div class="delivr-brand">'
    '<div class="brand-icon-box">'
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>'
    '</svg>'
    '</div>'
    '<div class="brand-titles">'
    '<span class="brand-name">DELIVR</span>'
    '<span class="brand-subtitle">AI Delivery Time Prediction</span>'
    '</div>'
    '</div>'
    f'<div class="header-badges">{badges_markup}</div>'
    '</div>'
)

st.markdown(header_html, unsafe_allow_html=True)

# ==============================================================================
# 6. HERO SECTION
# ==============================================================================
hero_html = (
    '<div class="delivr-hero">'
    '<h1 class="hero-title">Predict your delivery time.</h1>'
    '<p class="hero-subtitle">Estimate delivery time using distance, traffic, weather, preparation time and courier experience.</p>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

# Show model load error if any
if model_error:
    st.markdown(
        f'<div class="delivr-alert delivr-alert-error">'
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
        f'<span>{model_error}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

if st.session_state.validation_error:
    st.markdown(
        f'<div class="delivr-alert delivr-alert-warning">'
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        f'<span>{st.session_state.validation_error}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# 7. MAIN PREDICTION WORKSPACE (UNIFIED GLASS CARD CONTAINER)
# ==============================================================================
with st.container(border=True):
    st.markdown(
        '<div class="card-header-block">'
        '<div class="card-title">Delivery Parameters</div>'
        '<div class="card-desc">Enter the details of your order to generate an estimated delivery time.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        distance_val = st.number_input(
            "Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.distance),
            step=0.5,
            format="%.1f",
            help="Total travel distance in kilometers between store and destination.",
            key="input_distance",
        )

        weather_options = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
        weather_idx = weather_options.index(st.session_state.weather) if st.session_state.weather in weather_options else 0
        weather_val = st.selectbox(
            "Weather Condition",
            options=weather_options,
            index=weather_idx,
            help="Current atmospheric condition.",
            key="input_weather",
        )

        traffic_options = ["Low", "Medium", "High"]
        traffic_idx = traffic_options.index(st.session_state.traffic) if st.session_state.traffic in traffic_options else 1
        traffic_val = st.selectbox(
            "Traffic Level",
            options=traffic_options,
            index=traffic_idx,
            help="Real-time road congestion level.",
            key="input_traffic",
        )

        tod_options = ["Morning", "Afternoon", "Evening", "Night"]
        tod_idx = tod_options.index(st.session_state.time_of_day) if st.session_state.time_of_day in tod_options else 2
        tod_val = st.selectbox(
            "Time of Day",
            options=tod_options,
            index=tod_idx,
            help="Delivery dispatch window.",
            key="input_tod",
        )

    with col2:
        prep_val = st.number_input(
            "Preparation Time (minutes)",
            min_value=1,
            max_value=120,
            value=int(st.session_state.prep_time),
            step=1,
            help="Kitchen food preparation and packaging duration.",
            key="input_prep",
        )

        exp_val = st.number_input(
            "Courier Experience (years)",
            min_value=0.0,
            max_value=30.0,
            value=float(st.session_state.courier_exp),
            step=0.5,
            format="%.1f",
            help="Courier's professional delivery experience in years.",
            key="input_exp",
        )

        vehicle_options = ["Bike", "Car", "Scooter"]
        vehicle_idx = vehicle_options.index(st.session_state.vehicle_type) if st.session_state.vehicle_type in vehicle_options else 0
        vehicle_val = st.selectbox(
            "Vehicle Type",
            options=vehicle_options,
            index=vehicle_idx,
            help="Vehicle utilized for order transit.",
            key="input_vehicle",
        )

# ==============================================================================
# 8. ACTION BUTTONS & PREDICTION LOGIC
# ==============================================================================
btn_col1, btn_col2 = st.columns([3, 1], gap="small")

with btn_col1:
    predict_clicked = st.button("Predict Delivery Time", type="primary", use_container_width=True)

with btn_col2:
    reset_clicked = st.button("Reset", type="secondary", use_container_width=True)

if reset_clicked:
    reset_parameters()
    st.rerun()

if predict_clicked:
    if model is None:
        st.session_state.validation_error = "Prediction model unavailable. Please check that food_delivery_model.pkl exists."
    elif distance_val < 0:
        st.session_state.validation_error = "Distance cannot be negative."
    elif prep_val < 0:
        st.session_state.validation_error = "Preparation time cannot be negative."
    elif exp_val < 0:
        st.session_state.validation_error = "Courier experience cannot be negative."
    else:
        st.session_state.validation_error = None
        
        # Construct raw pandas DataFrame with exact 7 feature columns
        input_data = pd.DataFrame({
            "Distance_km": [float(distance_val)],
            "Weather": [str(weather_val)],
            "Traffic_Level": [str(traffic_val)],
            "Time_of_Day": [str(tod_val)],
            "Vehicle_Type": [str(vehicle_val)],
            "Preparation_Time_min": [float(prep_val)],
            "Courier_Experience_yrs": [float(exp_val)],
        })

        try:
            raw_prediction = model.predict(input_data)[0]
            prediction_val = max(1.0, round(float(raw_prediction), 1))
            st.session_state.prediction_result = prediction_val
            st.session_state.last_params = {
                "distance": distance_val,
                "weather": weather_val,
                "traffic": traffic_val,
                "time_of_day": tod_val,
                "vehicle": vehicle_val,
                "prep": prep_val,
                "exp": exp_val,
            }
        except Exception as e:
            st.session_state.validation_error = "An error occurred during prediction. Please verify input values."

# ==============================================================================
# 9. RESULT PANEL (INACTIVE vs ACTIVE STATE)
# ==============================================================================
if st.session_state.prediction_result is not None:
    pred = st.session_state.prediction_result
    params = st.session_state.last_params or {}
    
    st.markdown(
        f'<div class="result-container result-active">'
        f'<span class="result-active-tag">Estimated Delivery Time</span>'
        f'<div class="result-metric-display">'
        f'<span class="result-number">{int(round(pred))}</span>'
        f'<span class="result-unit">min</span>'
        f'</div>'
        f'<p class="result-summary-text">'
        f'Your order is estimated to arrive in approximately <strong>{int(round(pred))} minutes</strong> ({pred:.1f} min exact).'
        f'</p>'
        f'<div class="result-breakdown-row">'
        f'<div class="breakdown-chip"><span>Distance:</span><span class="breakdown-chip-val">{params.get("distance", distance_val)} km</span></div>'
        f'<div class="breakdown-chip"><span>Weather:</span><span class="breakdown-chip-val">{params.get("weather", weather_val)}</span></div>'
        f'<div class="breakdown-chip"><span>Traffic:</span><span class="breakdown-chip-val">{params.get("traffic", traffic_val)}</span></div>'
        f'<div class="breakdown-chip"><span>Vehicle:</span><span class="breakdown-chip-val">{params.get("vehicle", vehicle_val)}</span></div>'
        f'<div class="breakdown-chip"><span>Prep:</span><span class="breakdown-chip-val">{params.get("prep", prep_val)} min</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="result-container result-inactive">'
        '<div class="result-placeholder-icon">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polyline points="12 6 12 12 16 14"></polyline>'
        '</svg>'
        '</div>'
        '<div class="result-inactive-title">Your estimated delivery time</div>'
        '<div class="result-inactive-desc">Enter your order details above to calculate.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# 10. INSIGHTS SECTION
# ==============================================================================
st.markdown(
    '<div class="insights-section">'
    '<div class="section-header-title">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<line x1="12" y1="16" x2="12" y2="12"></line>'
    '<line x1="12" y1="8" x2="12.01" y2="8"></line>'
    '</svg>'
    '<span>What influences delivery time?</span>'
    '</div>'
    '<div class="insights-grid">'
    '<div class="insight-card">'
    '<div class="insight-icon">'
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>'
    '</div>'
    '<div class="insight-card-title">Distance</div>'
    '<div class="insight-card-desc">Longer travel distances can increase delivery duration.</div>'
    '</div>'
    '<div class="insight-card">'
    '<div class="insight-icon">'
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>'
    '</div>'
    '<div class="insight-card-title">Traffic</div>'
    '<div class="insight-card-desc">Traffic conditions are considered by the model when estimating delivery time.</div>'
    '</div>'
    '<div class="insight-card">'
    '<div class="insight-icon">'
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>'
    '</div>'
    '<div class="insight-card-title">Preparation</div>'
    '<div class="insight-card-desc">Preparation time contributes directly to the overall delivery duration.</div>'
    '</div>'
    '<div class="insight-card">'
    '<div class="insight-icon">'
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M22 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
    '</div>'
    '<div class="insight-card-title">Courier Experience</div>'
    '<div class="insight-card-desc">Courier experience is one of the model\'s input features.</div>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# 11. MODEL INFORMATION (EXPANDER)
# ==============================================================================
with st.expander("Model Details"):
    st.markdown(
        '<div class="model-spec-grid">'
        '<div class="spec-item"><div class="spec-label">Model Architecture</div><div class="spec-value">Random Forest Regressor</div></div>'
        '<div class="spec-item"><div class="spec-label">Task Type</div><div class="spec-value">Regression</div></div>'
        '<div class="spec-item"><div class="spec-label">Preprocessing</div><div class="spec-value">OneHotEncoder</div></div>'
        '<div class="spec-item"><div class="spec-label">Target Variable</div><div class="spec-value">Delivery_Time_min</div></div>'
        '<div class="spec-item"><div class="spec-label">Estimators</div><div class="spec-value">100 Trees</div></div>'
        '<div class="spec-item"><div class="spec-label">Test Performance</div><div class="spec-value">R² 0.77 · MAE 6.78m</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# 12. FOOTER
# ==============================================================================
st.markdown(
    '<div class="delivr-footer">DELIVR AI · High-Precision Food Delivery Duration Estimation Engine</div>',
    unsafe_allow_html=True,
)
