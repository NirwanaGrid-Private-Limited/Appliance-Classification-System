"""
Phase 2 Part 1: Baseline Audit Script
Retrains LightGBM_v1, freezes it, and runs the complete engineering audit.
"""
import pandas as pd
import numpy as np
import os
import json
import time
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r"c:\Users\saury\Desktop\nirvana_grid\data\processed\cleaned_dataset.csv"
MODEL_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\models"
PLOTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\plots\audit"
REPORTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\reports"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
logging.info("Loading data...")
df = pd.read_csv(DATA_PATH)
df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)

y = df['label']
groups = df['nodeid']

# Drop leakage + non-feature columns (same as Stage A)
LEAKAGE_COLS = ['label', 'nodeid', 'source_file', 'timestamp_ist', 'timestamp_ist_dt', 'energy']
X = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

FEATURE_NAMES = X.columns.tolist()
CLASSES = sorted(y.unique().tolist())
logging.info(f"Features: {FEATURE_NAMES}")
logging.info(f"Classes: {CLASSES}")
logging.info(f"Shape: {X.shape}")

# Stratified 80/20 split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(sss.split(X, y))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ============================================================
# 2. RETRAIN AND FREEZE LightGBM_v1
# ============================================================
logging.info("Training LightGBM_v1 (frozen baseline)...")
t0 = time.time()
model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)
train_time = time.time() - t0

t0 = time.time()
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)
inference_time = time.time() - t0

# Metrics
macro_f1 = f1_score(y_val, y_pred, average='macro')
macro_prec = precision_score(y_val, y_pred, average='macro')
macro_rec = recall_score(y_val, y_pred, average='macro')
per_class_rec = recall_score(y_val, y_pred, average=None, labels=CLASSES)

logging.info(f"LightGBM_v1 Macro F1: {macro_f1:.4f}")
logging.info(f"Per-class recall: {dict(zip(CLASSES, per_class_rec))}")

# Save frozen model
model_path = os.path.join(MODEL_DIR, "LightGBM_v1.joblib")
joblib.dump(model, model_path)

# Save metadata
metadata = {
    "version": "LightGBM_v1",
    "training_date": "2026-07-13",
    "features": FEATURE_NAMES,
    "hyperparameters": model.get_params(),
    "validation_strategy": "Stratified 80/20",
    "macro_f1": round(macro_f1, 4),
    "macro_precision": round(macro_prec, 4),
    "macro_recall": round(macro_rec, 4),
    "per_class_recall": {c: round(r, 4) for c, r in zip(CLASSES, per_class_rec)},
    "train_time_sec": round(train_time, 4),
    "inference_time_sec": round(inference_time, 4),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "status": "FROZEN_BASELINE"
}
with open(os.path.join(MODEL_DIR, "LightGBM_v1_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4, default=str)

logging.info(f"Model saved to {model_path}")

# ============================================================
# 3. SHAP ANALYSIS
# ============================================================
logging.info("Running SHAP analysis...")
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # shap_values is a list of arrays [n_samples x n_features] per class
    # or a single array for binary. For multi-class LightGBM, it's a list.
    
    shap_dir = os.path.join(PLOTS_DIR, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    # Global summary - bar plot (mean |SHAP|)
    # Compute mean absolute SHAP per feature across all classes
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    shap_ranking = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: -x[1])
    
    # Plot global feature ranking
    plt.figure(figsize=(10, 6))
    features_sorted = [x[0] for x in shap_ranking]
    values_sorted = [x[1] for x in shap_ranking]
    plt.barh(features_sorted[::-1], values_sorted[::-1], color='#2196F3')
    plt.title('Global SHAP Feature Importance (Mean |SHAP|)')
    plt.xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "global_shap_bar.png"), dpi=150)
    plt.close()

    # Per-class SHAP bar plots
    if isinstance(shap_values, list):
        for i, cls in enumerate(model.classes_):
            cls_shap = np.abs(shap_values[i]).mean(axis=0)
            cls_ranking = sorted(zip(FEATURE_NAMES, cls_shap), key=lambda x: -x[1])
            
            plt.figure(figsize=(10, 6))
            f_sorted = [x[0] for x in cls_ranking]
            v_sorted = [x[1] for x in cls_ranking]
            plt.barh(f_sorted[::-1], v_sorted[::-1], color='#4CAF50')
            plt.title(f'SHAP Feature Importance for Class: {cls}')
            plt.xlabel('Mean |SHAP Value|')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"shap_class_{cls}.png"), dpi=150)
            plt.close()

    # Local explanations: pick 1 correct and 1 misclassified per class
    local_explanations = []
    for cls in CLASSES:
        cls_mask = y_val == cls
        cls_preds = y_pred[cls_mask.values]
        cls_indices = np.where(cls_mask.values)[0]
        
        # Correct prediction
        correct_mask = cls_preds == cls
        if correct_mask.any():
            correct_idx = cls_indices[correct_mask][0]
            conf = y_proba[correct_idx].max()
            local_explanations.append({
                "sample_idx": int(correct_idx),
                "true_class": cls,
                "predicted_class": cls,
                "confidence": round(float(conf), 4),
                "type": "correct",
                "top_features": {}
            })
            if isinstance(shap_values, list):
                cls_class_idx = list(model.classes_).index(cls)
                sample_shap = shap_values[cls_class_idx][correct_idx]
                top_feat_idx = np.argsort(np.abs(sample_shap))[::-1][:3]
                local_explanations[-1]["top_features"] = {
                    FEATURE_NAMES[j]: round(float(sample_shap[j]), 4) for j in top_feat_idx
                }
        
        # Misclassified
        wrong_mask = cls_preds != cls
        if wrong_mask.any():
            wrong_idx = cls_indices[wrong_mask][0]
            pred_cls = y_pred[wrong_idx]
            conf = y_proba[wrong_idx].max()
            local_explanations.append({
                "sample_idx": int(wrong_idx),
                "true_class": cls,
                "predicted_class": pred_cls,
                "confidence": round(float(conf), 4),
                "type": "misclassified",
                "top_features": {}
            })
            if isinstance(shap_values, list):
                pred_class_idx = list(model.classes_).index(pred_cls)
                sample_shap = shap_values[pred_class_idx][wrong_idx]
                top_feat_idx = np.argsort(np.abs(sample_shap))[::-1][:3]
                local_explanations[-1]["top_features"] = {
                    FEATURE_NAMES[j]: round(float(sample_shap[j]), 4) for j in top_feat_idx
                }
    
    shap_results = {
        "global_ranking": shap_ranking,
        "local_explanations": local_explanations
    }
    logging.info("SHAP analysis complete.")

except Exception as e:
    logging.error(f"SHAP failed: {e}. Falling back to LightGBM gain importance.")
    shap_results = None

# ============================================================
# 4. FEATURE DOMINANCE ANALYSIS
# ============================================================
logging.info("Feature Dominance Analysis...")
lgbm_importance = model.feature_importances_
lgbm_ranking = sorted(zip(FEATURE_NAMES, lgbm_importance), key=lambda x: -x[1])

total_importance = sum(lgbm_importance)
dominance = {}
for feat, imp in lgbm_ranking:
    pct = (imp / total_importance) * 100
    dominance[feat] = round(pct, 2)

# Check if any feature > 50%
dominant_feature = lgbm_ranking[0]
is_dominant = (dominant_feature[1] / total_importance) > 0.50

logging.info(f"Feature dominance: {dominance}")
logging.info(f"Dominant feature: {dominant_feature[0]} = {dominance[dominant_feature[0]]}%")

# ============================================================
# 5. PERMUTATION IMPORTANCE
# ============================================================
logging.info("Computing Permutation Importance (10 repeats)...")
perm_result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring='f1_macro',
    n_jobs=-1
)

perm_ranking = sorted(
    zip(FEATURE_NAMES, perm_result.importances_mean, perm_result.importances_std),
    key=lambda x: -x[1]
)

plt.figure(figsize=(10, 6))
perm_features = [x[0] for x in perm_ranking]
perm_means = [x[1] for x in perm_ranking]
perm_stds = [x[2] for x in perm_ranking]
plt.barh(perm_features[::-1], perm_means[::-1], xerr=[s for s in perm_stds[::-1]], color='#FF9800')
plt.title('Permutation Importance (Macro F1 drop)')
plt.xlabel('Mean F1 Drop')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "permutation_importance.png"), dpi=150)
plt.close()

logging.info(f"Permutation ranking: {[(f, round(m, 4)) for f, m, _ in perm_ranking]}")

# ============================================================
# 6. CONFIDENCE ANALYSIS
# ============================================================
logging.info("Confidence Analysis...")
max_proba = y_proba.max(axis=1)
confidence_stats = {
    "mean": round(float(max_proba.mean()), 4),
    "median": round(float(np.median(max_proba)), 4),
    "min": round(float(max_proba.min()), 4),
    "max": round(float(max_proba.max()), 4),
    "std": round(float(max_proba.std()), 4),
    "pct_below_60": round(float((max_proba < 0.6).mean() * 100), 2),
    "pct_below_80": round(float((max_proba < 0.8).mean() * 100), 2),
}

# Per-class confidence
per_class_conf = {}
for cls in CLASSES:
    mask = y_val == cls
    cls_conf = max_proba[mask.values]
    per_class_conf[cls] = {
        "mean": round(float(cls_conf.mean()), 4),
        "min": round(float(cls_conf.min()), 4),
        "median": round(float(np.median(cls_conf)), 4),
        "pct_below_60": round(float((cls_conf < 0.6).mean() * 100), 2),
    }

# Overall histogram
plt.figure(figsize=(10, 6))
plt.hist(max_proba, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
plt.axvline(x=0.6, color='red', linestyle='--', label='60% threshold')
plt.axvline(x=0.8, color='orange', linestyle='--', label='80% threshold')
plt.title('Prediction Confidence Distribution (LightGBM_v1)')
plt.xlabel('Max Predicted Probability')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confidence_histogram.png"), dpi=150)
plt.close()

# Per-class histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, cls in enumerate(CLASSES):
    ax = axes[i // 2][i % 2]
    mask = y_val == cls
    cls_conf = max_proba[mask.values]
    ax.hist(cls_conf, bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
    ax.axvline(x=0.6, color='red', linestyle='--')
    ax.set_title(f'Confidence: {cls} (n={mask.sum()})')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
plt.suptitle('Per-Class Confidence Distributions')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confidence_per_class.png"), dpi=150)
plt.close()

logging.info(f"Confidence stats: {confidence_stats}")

# ============================================================
# 7. ERROR ANALYSIS
# ============================================================
logging.info("Error Analysis...")
misclassified_mask = y_pred != y_val.values
misclassified_indices = np.where(misclassified_mask)[0]
n_errors = len(misclassified_indices)
logging.info(f"Total misclassifications: {n_errors} / {len(y_val)} ({n_errors/len(y_val)*100:.2f}%)")

error_records = []
for idx in misclassified_indices:
    true_cls = y_val.iloc[idx]
    pred_cls = y_pred[idx]
    conf = max_proba[idx]
    
    # Feature values
    feat_vals = {f: round(float(X_val.iloc[idx][f]), 4) for f in FEATURE_NAMES}
    
    # Top SHAP features for this prediction (if available)
    top_shap = {}
    if shap_results and isinstance(shap_values, list):
        pred_class_idx = list(model.classes_).index(pred_cls)
        sample_shap = shap_values[pred_class_idx][idx]
        top_idx = np.argsort(np.abs(sample_shap))[::-1][:3]
        top_shap = {FEATURE_NAMES[j]: round(float(sample_shap[j]), 4) for j in top_idx}
    
    error_records.append({
        "idx": int(idx),
        "true_class": true_cls,
        "predicted_class": pred_cls,
        "confidence": round(float(conf), 4),
        "feature_values": feat_vals,
        "top_shap_features": top_shap
    })

# Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=CLASSES)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix: LightGBM_v1 (Frozen Baseline)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cm_LightGBM_v1_frozen.png"), dpi=150)
plt.close()

# Error pattern analysis: which class pairs are most confused
error_pairs = {}
for rec in error_records:
    pair = f"{rec['true_class']} → {rec['predicted_class']}"
    error_pairs[pair] = error_pairs.get(pair, 0) + 1
error_pairs = dict(sorted(error_pairs.items(), key=lambda x: -x[1]))

# ============================================================
# 8. LEAKAGE AUDIT
# ============================================================
logging.info("Leakage Audit...")
leakage_checks = {
    "nodeid_in_features": "nodeid" in FEATURE_NAMES,
    "source_file_in_features": "source_file" in FEATURE_NAMES,
    "energy_in_features": "energy" in FEATURE_NAMES,
    "timestamp_in_features": any("timestamp" in f for f in FEATURE_NAMES),
}

# Proxy leakage check: train a model with ONLY voltage + frequency
# If F1 is suspiciously high (>0.7), something is leaking
from lightgbm import LGBMClassifier as LGBMProxy
proxy_X_train = X_train[['voltage', 'frequency']]
proxy_X_val = X_val[['voltage', 'frequency']]
proxy_model = LGBMProxy(random_state=42, n_jobs=-1, verbose=-1)
proxy_model.fit(proxy_X_train, y_train)
proxy_pred = proxy_model.predict(proxy_X_val)
proxy_f1 = f1_score(y_val, proxy_pred, average='macro')
leakage_checks["proxy_voltage_freq_f1"] = round(proxy_f1, 4)
leakage_checks["proxy_leakage_detected"] = proxy_f1 > 0.70

logging.info(f"Leakage checks: {leakage_checks}")

# ============================================================
# 9. SAVE ALL RESULTS
# ============================================================
audit_results = {
    "model_metrics": metadata,
    "shap_global_ranking": shap_results["global_ranking"] if shap_results else None,
    "shap_local_explanations": shap_results["local_explanations"] if shap_results else None,
    "feature_dominance": dominance,
    "is_single_feature_dominant": is_dominant,
    "dominant_feature": dominant_feature[0],
    "lgbm_ranking": [(f, int(i)) for f, i in lgbm_ranking],
    "permutation_ranking": [(f, round(m, 4), round(s, 4)) for f, m, s in perm_ranking],
    "confidence_stats": confidence_stats,
    "per_class_confidence": per_class_conf,
    "n_errors": n_errors,
    "error_rate_pct": round(n_errors / len(y_val) * 100, 2),
    "error_pairs": error_pairs,
    "error_records": error_records[:50],  # Cap at 50 for readability
    "leakage_checks": leakage_checks,
    "classification_report": classification_report(y_val, y_pred, target_names=CLASSES, output_dict=True)
}

results_path = os.path.join(REPORTS_DIR, "baseline_audit_results.json")
with open(results_path, "w") as f:
    json.dump(audit_results, f, indent=4, default=str)

logging.info(f"Audit results saved to {results_path}")

# ============================================================
# STOP GATE SUMMARY
# ============================================================
print("\n" + "="*60)
print("STOP GATE ASSESSMENT")
print("="*60)
print(f"1. Macro F1: {macro_f1:.4f}")
print(f"2. Leakage detected: {any(leakage_checks[k] for k in ['nodeid_in_features', 'source_file_in_features', 'energy_in_features', 'timestamp_in_features'])}")
print(f"3. Proxy leakage (voltage+freq only F1): {proxy_f1:.4f} ({'SUSPICIOUS' if proxy_f1 > 0.7 else 'OK'})")
print(f"4. Single feature dominance: {dominant_feature[0]} at {dominance[dominant_feature[0]]}%")
print(f"5. Low confidence predictions (<60%): {confidence_stats['pct_below_60']}%")
print(f"6. Error rate: {n_errors}/{len(y_val)} ({n_errors/len(y_val)*100:.2f}%)")
print(f"7. Most confused pair: {list(error_pairs.items())[0] if error_pairs else 'None'}")
print("="*60)

logging.info("Baseline audit complete.")
