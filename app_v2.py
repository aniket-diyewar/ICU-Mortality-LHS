' py -m streamlit run app_v2.py '

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# 1. Page Config & Custom CSS for a sleek look
st.set_page_config(page_title="LHS Predictor v2.0", page_icon="🏥", layout="centered")

st.markdown("""
    <style>
    /* Styling for the calculate button */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    with open(r"models/mortality_model_v2.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model not found! Please run 02_train_model_v2.py first.")
    st.stop()

# 3. Header
st.markdown("<h1 style='text-align: center;'>🏥 ICU Mortality Risk Predictor 2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Learning Health System (LHS) Prototype • Powered by XGBoost</p>", unsafe_allow_html=True)
st.divider()

# 4. Input Section (Matched to your layout)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("📄 Patient Age", 1, 100, 18)
    los = st.slider("🛏️ Length of Stay (Days)", 0, 50, 0)
    gender = st.radio("🚻 Gender", ["Male", "Female"], horizontal=True)
    admission = st.radio("🏥 Admission Type", ["Emergency", "Elective/Other"],horizontal=True)
    
    

with col2:
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 40, 200, 72)
    spo2 = st.slider("💨 SpO2 (Oxygen %)", 50, 100, 100)
    sys_bp = st.slider("🩸 Systolic BP (mmHg)", 60, 250, 120)
    
    st.write("") # Spacer
    st.write("") # Spacer
    calculate_btn = st.button("Calculate Risk & Explain", type="primary")

# 5. Prediction & Output Section
if calculate_btn:
    # Prepare Data
    gender_encoded = 1 if gender == "Male" else 0
    is_emergency = 1 if admission == "Emergency" else 0
    feature_names = ['age', 'los', 'gender_encoded', 'is_emergency', 'heart_rate', 'sys_bp', 'spo2']
    input_data = pd.DataFrame([[age, los, gender_encoded, is_emergency, heart_rate, sys_bp, spo2]], columns=feature_names)
    
    # Predict
    risk_prob = model.predict_proba(input_data)[0][1]
    
    # Determine Status & Colors
    if risk_prob < 0.3:
        status_text, status_color = "Low Risk", "#00FF7F" # Spring Green
    elif risk_prob < 0.6:
        status_text, status_color = "Moderate Risk", "#FFD700" # Gold
    else:
        status_text, status_color = "High Risk", "#FF4500" # Orange Red

    st.divider()
    
    # Custom HTML Metric Display (Matches your screenshot)
    st.markdown(f"""
        <div style="display: flex; gap: 50px; padding: 10px 0px;">
            <div>
                <p style="color: #A0AEC0; font-size: 13px; margin-bottom: 5px; font-weight: bold; letter-spacing: 1px;">PREDICTED RISK</p>
                <h2 style="margin: 0px; font-size: 32px;">{risk_prob:.1%}</h2>
            </div>
            <div>
                <p style="color: #A0AEC0; font-size: 13px; margin-bottom: 5px; font-weight: bold; letter-spacing: 1px;">STATUS</p>
                <h2 style="margin: 0px; font-size: 32px; color: {status_color};">{status_text}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")

    # Custom Clean Explainable AI Chart
    st.markdown("<p style='color: #A0AEC0; font-size: 13px; font-weight: bold; letter-spacing: 1px;'>AI DECISION DRIVERS (SHAP)</p>", unsafe_allow_html=True)
    
    with st.spinner("Generating AI explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # Extract values for custom plotting
        vals = shap_values.values[0]
        
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Colors: Red for increasing risk, Blue for decreasing
        colors = ['#FF4500' if x > 0 else '#1E90FF' for x in vals]
        
        # Plot horizontal bar chart
        bars = ax.barh(feature_names, vals, color=colors, height=0.6)
        
        # Formatting the chart
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Impact on Mortality Risk", color='lightgray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        
        # Add values inside bars
        for i, (bar, val) in enumerate(zip(bars, vals)):
            text_x = bar.get_width() - 0.05 if val > 0 else bar.get_width() + 0.05
            ha = 'right' if val > 0 else 'left'
            ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{val:+.2f}", 
                    va='center', ha=ha, color='white', fontweight='bold')
            
        st.pyplot(fig)