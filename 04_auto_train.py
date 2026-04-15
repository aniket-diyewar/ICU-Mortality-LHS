import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import json
from datetime import datetime

LIVE_DATA_PATH = r"outputs/live_clinic_data.csv"
MODEL_PATH = r"models/mortality_model_v2.pkl"
META_PATH = r"models/model_metadata.json"

print("🧠 LHS Auto-Retraining Initiated...")

# 1. Load the continuously growing live data
df = pd.read_csv(LIVE_DATA_PATH)
num_records = len(df)
print(f"📊 Found {num_records} total patient records in the EMR database.")

# 2. Prepare features
features = ['age', 'los', 'gender_encoded', 'is_emergency', 'heart_rate', 'sys_bp', 'spo2']
X = df[features]
y = df['hospital_expire_flag']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Retrain the model on the new dataset
print("⚙️ Retraining XGBoost model to absorb new patterns...")
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate performance
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"📈 Updated Model AUC-ROC: {auc:.4f}")

# 5. Save the updated model (overwriting the old one)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# 6. Save Metadata for the Streamlit Dashboard
metadata = {
    "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_patients": num_records,
    "current_auc": round(auc, 4)
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f)

print(f"✅ LHS successfully updated! Metadata saved to {META_PATH}")