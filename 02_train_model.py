# 02_train_model.py
import pandas as pd
import numpy as np
import pickle, os, json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

OUTPUT = r"E:\ICU_Mortality_LHS\outputs"
MODELS = r"E:\ICU_Mortality_LHS\models"
os.makedirs(MODELS, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(os.path.join(OUTPUT, 'combined_mimic.csv'))
print(f"✅ Loaded: {df.shape}")

# ============================================================
# FEATURES & TARGET
# ============================================================
FEATURES = ['age', 'los', 'gender_encoded', 'is_emergency']
TARGET   = 'hospital_expire_flag'

X = df[FEATURES]
y = df[TARGET]

print(f"\n📊 Features: {FEATURES}")
print(f"   Samples: {len(X)} | Mortality rate: {y.mean():.1%}")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# TRAIN 3 MODELS — pick best
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, max_depth=5,
                                          use_label_encoder=False,
                                          eval_metric='logloss', random_state=42)
}

results = {}
print("\n🏋️ Training models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    cv     = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

    results[name] = {'model': model, 'auc': auc, 'cv_auc': cv,
                     'y_pred': y_pred, 'y_prob': y_prob}

    print(f"  {'='*40}")
    print(f"  Model  : {name}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  CV AUC : {cv:.4f}  (5-fold)")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

# ============================================================
# PICK BEST MODEL
# ============================================================
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
best_model = best['model']

print(f"\n🏆 Best Model: {best_name} (AUC = {best['auc']:.4f})")

# ============================================================
# SAVE MODEL + METADATA
# ============================================================
model_path = os.path.join(MODELS, 'mortality_model.pkl')
pickle.dump(best_model, open(model_path, 'wb'))

meta = {
    "model_name":   best_name,
    "features":     FEATURES,
    "auc_roc":      round(best['auc'], 4),
    "cv_auc":       round(best['cv_auc'], 4),
    "train_size":   int(X_train.shape[0]),
    "test_size":    int(X_test.shape[0]),
    "mortality_rate": round(float(y.mean()), 4),
    "dataset":      "MIMIC-III + MIMIC-IV Demo"
}
json.dump(meta, open(os.path.join(MODELS, 'model_meta.json'), 'w'), indent=2)
print(f"✅ Model saved to: {model_path}")

# ============================================================
# PLOTS — ROC Curve + Confusion Matrix
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, best['y_prob'])
axes[0].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'{best_name} (AUC = {best["auc"]:.4f})')
axes[0].plot([0,1],[0,1],'k--', lw=1, label='Random Baseline')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — ICU Mortality Prediction')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=['Survived', 'Died'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plot_path = os.path.join(OUTPUT, 'model_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Plot saved to: {plot_path}")

# ============================================================
# FEATURE IMPORTANCE (if XGBoost or RF)
# ============================================================
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fi.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Importance Score')
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Feature importance saved to: {fi_path}")

print("\n🎉 Phase 2 Complete! Check your outputs/ and models/ folders.")
print(f"   Model metadata: {json.dumps(meta, indent=2)}")