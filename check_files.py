# check_files.py
import os

MIMIC3 = r"E:\Data\mimic-iii-clinical-database-demo-1.4"
MIMIC4 = r"E:\Data\mimic-iv-clinical-database-demo-2.2"

print("===== MIMIC-III FILES =====")
for f in os.listdir(MIMIC3):
    print(f)

print("\n===== MIMIC-IV ROOT =====")
for f in os.listdir(MIMIC4):
    print(f)

# Check subfolders
for sub in ['hosp', 'icu']:
    path = os.path.join(MIMIC4, sub)
    if os.path.exists(path):
        print(f"\n===== MIMIC-IV/{sub} =====")
        for f in os.listdir(path):
            print(f)