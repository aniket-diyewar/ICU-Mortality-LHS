' py -m streamlit run app_v4.py '

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import json
import os

# 1. Page Config & Custom CSS
st.set_page_config(page_title="LHS Predictor v4.0", page_icon="🏥", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. LHS LIVE SYSTEM STATUS SIDEBAR (Checkpoint 3)
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>⚙️ LHS System Status</h3>", unsafe_allow_html=True)
    
    try:
        with open(r"models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            
        st.metric(label="Total Patients Learned", value=metadata["total_patients"])
        st.metric(label="Current Model AUC", value=f"{metadata['current_auc']:.4f}")
        st.caption(f"🕒 Last Retrained: {metadata['last_trained']}")
        
        st.divider()
        st.success("🟢 System is Live and Learning")
        st.info("As the EMR generates new patient data, the XGBoost engine automatically absorbs it to refine its clinical predictions.")
        
    except FileNotFoundError:
        st.warning("Live tracking disabled. Run 04_auto_train.py to activate.")

# 3. Load Model
@st.cache_resource
def load_model():
    with open(r"models/mortality_model_v2.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model not found! Please run 04_auto_train.py first.")
    st.stop()

# 4. Header (Modern Badge Style)
st.markdown("<h1 style='text-align: center;'>🏥 ICU Mortality Risk Predictor 4.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Learning Health System (LHS) Prototype • Powered by XGBoost</p>", unsafe_allow_html=True)
st.divider()


# 5. Input Section
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

# 6. Prediction & Output Section
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
        status_text, status_color = "Low Risk", "#00FF7F" 
    elif risk_prob < 0.6:
        status_text, status_color = "Moderate Risk", "#FFD700" 
    else:
        status_text, status_color = "High Risk", "#FF4500" 

    st.divider()
    
    # Custom HTML Metric Display
    st.markdown(f"""
        <div style="display: flex; justify-content: center; gap: 80px; padding: 10px 0px; text-align: center;">
            <div>
                <p style="color: #A0AEC0; font-size: 13px; margin-bottom: 5px; font-weight: bold; letter-spacing: 1px;">PREDICTED RISK</p>
                <h2 style="margin: 0px; font-size: 38px;">{risk_prob:.1%}</h2>
            </div>
            <div>
                <p style="color: #A0AEC0; font-size: 13px; margin-bottom: 5px; font-weight: bold; letter-spacing: 1px;">STATUS</p>
                <h2 style="margin: 0px; font-size: 38px; color: {status_color};">{status_text}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")

    # Custom Clean Explainable AI Chart
    st.markdown("<p style='color: #A0AEC0; font-size: 13px; font-weight: bold; letter-spacing: 1px;'>AI DECISION DRIVERS (SHAP)</p>", unsafe_allow_html=True)
    
    with st.spinner("Generating AI explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        vals = shap_values.values[0]
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        
        colors = ['#FF4500' if x > 0 else '#1E90FF' for x in vals]
        bars = ax.barh(feature_names, vals, color=colors, height=0.6)
        
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Impact on Mortality Risk", color='lightgray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        
        for i, (bar, val) in enumerate(zip(bars, vals)):
            text_x = bar.get_width() - 0.05 if val > 0 else bar.get_width() + 0.05
            ha = 'right' if val > 0 else 'left'
            ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{val:+.2f}", 
                    va='center', ha=ha, color='white', fontweight='bold')
            
        st.pyplot(fig)