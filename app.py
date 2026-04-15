' py -m streamlit run app.py '

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="LHS Mortality Predictor", page_icon="🏥", layout="centered")

# Load Model
MODEL_PATH = r"models/mortality_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run 02_train_model.py first.")
    st.stop()

# Header
st.title("🏥 ICU Mortality Risk Predictor")
st.markdown("""
**Learning Health System (LHS) Prototype** This tool predicts patient mortality risk by analyzing integrated data from **MIMIC-III** and **MIMIC-IV** datasets.
""")

st.divider()

# Input Form
st.subheader("Patient Clinical Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", 1, 100, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    los = st.number_input("Current Length of Stay (Days)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    admission = st.selectbox("Admission Type", ["Emergency", "Elective/Other"])

# Mapping inputs to model features
gender_encoded = 1 if gender == "Male" else 0
is_emergency = 1 if admission == "Emergency" else 0

# --- Predict Button ---
st.divider()

if st.button("Calculate Mortality Risk", type="primary", use_container_width=True):
    with st.spinner('Analyzing patient data...'): # Adds a loading animation
        # Prepare input data
        input_features = np.array([[age, los, gender_encoded, is_emergency]])
        risk_prob = model.predict_proba(input_features)[0][1]
        
        st.subheader("Risk Assessment Results")
        
        # Use columns for a nicer layout
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Big metric display
            st.metric(label="Calculated Risk", value=f"{risk_prob:.1%}")
            
        with res_col2:
            if risk_prob < 0.3:
                st.success("🟢 **Low Risk**")
                st.balloons() # Fun animation for good news
            elif risk_prob < 0.6:
                st.warning("🟡 **Moderate Risk** - Monitor closely.")
            else:
                st.error("🔴 **High Risk** - Immediate attention recommended.")
                
        st.progress(float(risk_prob))
                    
        st.info("💡 **LHS Note:** This prediction is based on the current model weights. As more data is ingested, the system can be scheduled to retrain and adapt.")

# Sidebar info
st.sidebar.header("About the Project")
st.sidebar.write("**Model:** XGBoost Classifier")
st.sidebar.write("**Dataset:** MIMIC-III & IV Integrated")
st.sidebar.write("**Target:** Hospital Mortality")