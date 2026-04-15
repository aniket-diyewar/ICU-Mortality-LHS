# 🏥 Continuous Learning Health System (LHS) for ICU Mortality Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
This project is an end-to-end **Learning Health System (LHS)** prototype designed to predict Intensive Care Unit (ICU) mortality risk. Unlike static machine learning models, this system features a **continuous-learning pipeline** that simulates real-world Electronic Medical Record (EMR) data ingestion, automatically retraining itself as new patients are admitted and discharged. 

It integrates historical demographic data, real-time physiological vitals, and **Explainable AI (XAI)** to provide transparent, adaptive, and highly accurate risk stratification for clinical decision support.

## 🚀 Key Features
* **Automated Continuous Learning:** Simulates a live hospital EMR system, automatically ingesting new patient data and retraining the XGBoost model to absorb shifting clinical patterns.
* **Explainable AI (SHAP):** Features an interactive, dynamic force plot that visually breaks down the mathematical log-odds of every prediction, telling clinicians exactly *why* a risk score was assigned.
* **Cross-Generational Data Integration:** Built on a unified cohort bridging **MIMIC-III** and **MIMIC-IV** clinical databases.
* **Clinical-Grade Interface:** A sleek, dark-mode Streamlit dashboard that tracks live system metadata (Total Patients Learned, Current AUC) and provides real-time risk assessments.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Data Processing:** Pandas, NumPy
* **Frontend UI:** Streamlit, Matplotlib

## 📂 Project Architecture

```text
ICU-Mortality-LHS/
│
├── models/
│   ├── mortality_model_v2.pkl     # The live, continuously updated XGBoost model
│   └── model_metadata.json        # Tracks system status (AUC, last trained, patient count)
│
├── outputs/
│   ├── combined_mimic.csv         # Baseline historical data (MIMIC III + IV)
│   └── live_clinic_data.csv       # Active database that grows with new simulated patients
│
├── app.py                         # Streamlit Dashboard (Frontend)
├── 03_data_ingestion.py           # EMR Simulator (Generates new patient records)
└── 04_auto_train.py               # Auto-Retrainer (Retrains model on new EMR data)
