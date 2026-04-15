import pandas as pd
import numpy as np
import os
import time

LIVE_DATA_PATH = r"outputs/live_clinic_data.csv"
BASE_DATA_PATH = r"outputs/combined_mimic.csv"

def generate_new_patients(num_patients):
    print(f"🏥 Hospital EMR: Generating {num_patients} new patient records...")
    
    # 1. Generate outcomes (Did they survive?) - roughly 20% mortality for simulation
    outcomes = np.random.choice([0, 1], size=num_patients, p=[0.8, 0.2])
    
    # 2. Generate demographics
    ages = np.random.randint(18, 95, size=num_patients)
    los = np.round(np.random.uniform(0.5, 30.0, size=num_patients), 1)
    genders = np.random.choice([0, 1], size=num_patients)
    emergencies = np.random.choice([0, 1], size=num_patients, p=[0.3, 0.7])
    
    # 3. Generate vitals (correlated with outcome so the model can learn)
    heart_rates = np.where(outcomes == 1, np.random.normal(105, 15, num_patients), np.random.normal(80, 10, num_patients))
    sys_bps = np.where(outcomes == 1, np.random.normal(90, 20, num_patients), np.random.normal(120, 15, num_patients))
    spo2s = np.where(outcomes == 1, np.random.normal(90, 4, num_patients), np.random.normal(98, 2, num_patients))
    spo2s = np.clip(spo2s, 0, 100) # Cap oxygen at 100%
    
    # Build DataFrame
    new_df = pd.DataFrame({
        'age': ages,
        'los': los,
        'gender_encoded': genders,
        'is_emergency': emergencies,
        'heart_rate': heart_rates,
        'sys_bp': sys_bps,
        'spo2': spo2s,
        'hospital_expire_flag': outcomes
    })
    return new_df

# --- Main Execution ---
print("🔄 Starting Data Ingestion Pipeline...")
time.sleep(1)

# If the live database doesn't exist, initialize it using our Checkpoint 2 logic
if not os.path.exists(LIVE_DATA_PATH):
    print("📁 Initializing live database with historical MIMIC records...")
    base_df = pd.read_csv(BASE_DATA_PATH)
    
    np.random.seed(42)
    base_df['heart_rate'] = np.where(base_df['hospital_expire_flag'] == 1, np.random.normal(105, 15, len(base_df)), np.random.normal(80, 10, len(base_df)))
    base_df['sys_bp'] = np.where(base_df['hospital_expire_flag'] == 1, np.random.normal(90, 20, len(base_df)), np.random.normal(120, 15, len(base_df)))
    base_df['spo2'] = np.where(base_df['hospital_expire_flag'] == 1, np.random.normal(90, 4, len(base_df)), np.random.normal(98, 2, len(base_df)))
    base_df['spo2'] = base_df['spo2'].clip(upper=100)
    
    features_to_keep = ['age', 'los', 'gender_encoded', 'is_emergency', 'heart_rate', 'sys_bp', 'spo2', 'hospital_expire_flag']
    base_df = base_df[features_to_keep]
    base_df.to_csv(LIVE_DATA_PATH, index=False)
    print(f"✅ Initialized with {len(base_df)} records.")

# Generate today's new patients
new_data = generate_new_patients(num_patients=15)

# Read the current live database, append the new records, and save it back
current_live_data = pd.read_csv(LIVE_DATA_PATH)
updated_live_data = pd.concat([current_live_data, new_data], ignore_index=True)
updated_live_data.to_csv(LIVE_DATA_PATH, index=False)

print(f"✅ Ingestion Complete. The live database now has {len(updated_live_data)} total records.")