# 01_data_prep.py
import pandas as pd
import os

# ============================================================
# PATHS — adjust only if your folder names differ
# ============================================================
MIMIC3 = r"E:\Data\mimic-iii-clinical-database-demo-1.4"
MIMIC4 = r"E:\Data\mimic-iv-clinical-database-demo-2.2"
OUTPUT = r"E:\ICU_Mortality_LHS\outputs"

# ============================================================
# HELPER: smart CSV reader (handles .csv and .csv.gz both)
# ============================================================
def read_csv_smart(folder, filename_no_ext):
    for ext in ['.csv.gz', '.csv']:
        path = os.path.join(folder, filename_no_ext + ext)
        if os.path.exists(path):
            print(f"  ✅ Reading: {path}")
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"❌ Could not find {filename_no_ext} in {folder}")

# ============================================================
# MIMIC-IV — Load
# ============================================================
print("\n📂 Loading MIMIC-IV...")

hosp_folder = os.path.join(MIMIC4, 'hosp')
icu_folder  = os.path.join(MIMIC4, 'icu')

adm4 = read_csv_smart(hosp_folder, 'admissions')
pat4 = read_csv_smart(hosp_folder, 'patients')
icu4 = read_csv_smart(icu_folder,  'icustays')

# Lowercase all columns
adm4.columns = adm4.columns.str.lower()
pat4.columns = pat4.columns.str.lower()
icu4.columns = icu4.columns.str.lower()

print(f"  admissions: {adm4.shape}, patients: {pat4.shape}, icustays: {icu4.shape}")

# Merge MIMIC-IV
df4 = icu4.merge(
    adm4[['hadm_id', 'hospital_expire_flag', 'admission_type']],
    on='hadm_id', how='left'
)
df4 = df4.merge(
    pat4[['subject_id', 'anchor_age', 'gender']],
    on='subject_id', how='left'
)
df4['source'] = 'mimic4'
df4.rename(columns={'anchor_age': 'age'}, inplace=True)

print(f"  ✅ MIMIC-IV merged: {df4.shape}")

# ============================================================
# MIMIC-III — Load
# ============================================================
print("\n📂 Loading MIMIC-III...")

adm3 = read_csv_smart(MIMIC3, 'ADMISSIONS')
pat3 = read_csv_smart(MIMIC3, 'PATIENTS')
icu3 = read_csv_smart(MIMIC3, 'ICUSTAYS')

# Lowercase all columns
adm3.columns = adm3.columns.str.lower()
pat3.columns = pat3.columns.str.lower()
icu3.columns = icu3.columns.str.lower()

print(f"  admissions: {adm3.shape}, patients: {pat3.shape}, icustays: {icu3.shape}")

# ---  MIMIC-III Age block  ---
print("  Calculating ages for MIMIC-III (handling shifted dates)...")
adm3['admittime'] = pd.to_datetime(adm3['admittime'], errors='coerce')
pat3['dob'] = pd.to_datetime(pat3['dob'], errors='coerce')

adm3_pat3 = adm3.merge(pat3[['subject_id', 'dob', 'gender']], on='subject_id', how='left')

# Robust age calculation: If DOB is before 1900 or results in huge ages, it's a shifted date
def calculate_mimic3_age(row):
    try:
        age = (row['admittime'].year - row['dob'].year)
        if age > 100 or age < 0:
            return 90.0 # Standard MIMIC practice for de-identified elderly
        return float(age)
    except:
        return 50.0 # Default median age if data is missing

adm3_pat3['age'] = adm3_pat3.apply(calculate_mimic3_age, axis=1)
# -----------------------------------------------------

# MIMIC-III masks age >89 as ~300 — cap at 90
adm3_pat3['age'] = adm3_pat3['age'].clip(upper=90)

# Merge MIMIC-III
df3 = icu3.merge(
    adm3_pat3[['hadm_id', 'hospital_expire_flag', 'age', 'gender', 'admission_type']],
    on='hadm_id', how='left'
)
df3['source'] = 'mimic3'

print(f"  ✅ MIMIC-III merged: {df3.shape}")

# ============================================================
# COMBINE MIMIC-III + MIMIC-IV
# ============================================================
print("\n🔗 Combining datasets...")

COMMON_COLS = ['subject_id', 'hadm_id', 'los', 'age', 'gender',
               'hospital_expire_flag', 'admission_type', 'source']

# Keep only columns that exist
df4_clean = df4[[c for c in COMMON_COLS if c in df4.columns]]
df3_clean = df3[[c for c in COMMON_COLS if c in df3.columns]]

combined = pd.concat([df4_clean, df3_clean], ignore_index=True)

print(f"  Total rows before cleaning: {combined.shape[0]}")

# ============================================================
# CLEANING
# ============================================================
# Drop rows with missing target or key features
combined.dropna(subset=['hospital_expire_flag', 'age', 'los', 'gender'], inplace=True)

# Encode gender
combined['gender_encoded'] = (combined['gender'] == 'M').astype(int)

# Encode admission type
combined['admission_type'] = combined['admission_type'].fillna('UNKNOWN').str.upper()
combined['is_emergency'] = combined['admission_type'].str.contains('EMERG').astype(int)

# Remove invalid ages/LOS
combined = combined[(combined['age'] > 0) & (combined['los'] >= 0)]

print(f"  Total rows after cleaning: {combined.shape[0]}")
print(f"\n📊 Target Distribution:")
print(combined['hospital_expire_flag'].value_counts())
print(f"\n  Mortality Rate: {combined['hospital_expire_flag'].mean():.1%}")
print(f"\n  Source breakdown:")
print(combined['source'].value_counts())

# Save
os.makedirs(OUTPUT, exist_ok=True)
save_path = os.path.join(OUTPUT, 'combined_mimic.csv')
combined.to_csv(save_path, index=False)
print(f"\n✅ Saved to: {save_path}")