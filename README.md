# 🏥 Learning Health System for ICU Mortality Prediction
### MIMIC-III + MIMIC-IV | XGBoost + SHAP | Streamlit | Auto-Retraining

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This project implements a **Learning Health System (LHS)** — a clinical AI pipeline where the model **continuously learns and improves** as new patient data flows in. It predicts **ICU mortality risk** for patients using real physiological vitals and clinical admission data.

> **"A Learning Health System is a healthcare ecosystem where data from patient care is routinely captured and used to generate new knowledge, which in turn improves future care."**  
> — Institute of Medicine, 2007

### 🎯 What This System Does

| Stage | Description |
|-------|-------------|
| 🔗 **Data Integration** | Merges MIMIC-III and MIMIC-IV ICU datasets into a unified pipeline |
| 🤖 **ML Prediction** | XGBoost model predicts mortality risk from 7 clinical features |
| 🧠 **Explainable AI** | SHAP force plots explain every single prediction in real-time |
| 🔄 **Auto-Retraining** | System automatically retrains when new patient data arrives |
| 📊 **Live Dashboard** | Streamlit app shows live model status, AUC score, and patient count |

---

## 🗂️ Project Architecture

```
ICU_Mortality_LHS/
│
├── 📊 Data Pipeline
│   ├── 01_data_prep.py          → Merges MIMIC-III + MIMIC-IV, feature engineering
│   ├── 03_data_ingestion.py     → EMR Simulator: generates live synthetic patient records
│   └── 04_auto_train.py         → Auto-Retrainer: ingests new data & retrains XGBoost
│
├── 📁 outputs/
│   ├── combined_mimic.csv       → Integrated historical dataset
│   └── live_clinic_data.csv     → Growing live database (EMR feed)
│
├── 📁 models/
│   ├── mortality_model_v2.pkl   → Trained XGBoost model (auto-updated)
│   └── model_metadata.json      → AUC score, patient count, last trained timestamp
│
├── 🌐 app.py                    → Streamlit web application (dark-mode UI)
├── 02_train_model.py            → Baseline model training script
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset: MIMIC-III + MIMIC-IV

| Property | MIMIC-III | MIMIC-IV |
|---|---|---|
| ICU Stays | ~61,000 | ~76,000 |
| Years Covered | 2001–2012 | 2008–2019 |
| Age Handling | Privacy-shifted (year ~3000) — custom fix applied | `anchor_age` field |
| Merged Via | `subject_id`, `hadm_id` | `subject_id`, `hadm_id` |

### Features Used (7 Clinical Variables)

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age at ICU admission |
| `los` | Numerical | Length of ICU Stay (days) |
| `gender_encoded` | Binary | 1 = Male, 0 = Female |
| `is_emergency` | Binary | 1 = Emergency admission |
| `heart_rate` | Numerical | Mean heart rate (bpm) |
| `sys_bp` | Numerical | Systolic Blood Pressure (mmHg) |
| `spo2` | Numerical | Oxygen Saturation (%) |

**Target Variable:** `hospital_expire_flag` → `1` = Died, `0` = Survived

---

## 🤖 Model Performance

| Model | AUC-ROC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | ~0.72 | ~0.71 |
| Random Forest | ~0.78 | ~0.77 |
| **XGBoost ✅ (Best)** | **~0.82** | **~0.81** |

> Model continuously improves as `04_auto_train.py` ingests new patient data.

---

## 🖥️ Web Application

The Streamlit app (`app.py`) features:

- **🎨 Dark-mode clinical UI** — professional hospital-grade design
- **7 Clinical Input Sliders** — Age, LOS, Gender, Admission Type, Heart Rate, BP, SpO2
- **Risk Gauge** — Low / Moderate / High risk classification with percentage score
- **SHAP Force Plot** — Real-time explainability: shows exactly which features pushed the prediction up or down
- **Live System Sidebar** — Displays `Total Patients Learned`, `Current AUC`, and `Last Trained` timestamp from `model_metadata.json`

### Run the App

```bash
streamlit run app.py
```

---

## 🔄 The Learning Health System Workflow

```
New Patient Admitted
        ↓
EMR Simulator (03_data_ingestion.py)
        ↓
live_clinic_data.csv grows
        ↓
Auto-Retrainer (04_auto_train.py) triggers
        ↓
XGBoost retrains on ALL data (historical + new)
        ↓
model_metadata.json updated (new AUC, patient count)
        ↓
Streamlit sidebar reflects new model stats LIVE
        ↓
Better predictions for next patient 🔁
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ICU_Mortality_LHS.git
cd ICU_Mortality_LHS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add MIMIC Data
Download from:
> MIMIC-III --> https://www.kaggle.com/datasets/asjad99/mimiciii  
> MIMIC-IV ---> https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2

Place your data at:
```
E:\Data\mimic-iii-clinical-database-demo-1.4\
E:\Data\mimic-iv-clinical-database-demo-2.2\
```

### 4. Run the Pipeline
```bash
# Step 1: Prepare data
python 01_data_prep.py

# Step 2: Train model
python 02_train_model.py

# Step 3: Simulate new patient data
python 03_data_ingestion.py

# Step 4: Auto-retrain
python 04_auto_train.py

# Step 5: Launch app
streamlit run app.py
```

## 📱Website Preview 


<img src="./Webpage_Preview/Screenshot 2026-04-15 172502.png" width="350" alt="Dashboard Preview"/>

