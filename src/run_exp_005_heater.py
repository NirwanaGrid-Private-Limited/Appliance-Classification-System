"""
EXP-005: Heater Integration Experiment
Trains LightGBM_v5_combined using identical settings to LightGBM_v4_combined,
but on cleaned_dataset_v2.csv to evaluate the addition of the Heater class.
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = config.DATA_PROCESSED_DIR / "cleaned_dataset_v2.csv"
MODEL_DIR = os.path.join(config.BASE_DIR, "outputs", "models")
PLOTS_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "stage_b")
REPORTS_DIR = config.REPORTS_DIR
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Baseline for Comparison (LightGBM_v4_combined metrics approx)
# ============================================================
# Since we don't have the exact JSON of v4 in memory, we will just log v5
# and let the report compare manually, but we can also load v4 metadata if it exists.
try:
    with open(os.path.join(MODEL_DIR, "LightGBM_v4_combined_metadata.json"), "r") as f:
        v4_meta = json.load(f)
except:
    v4_meta = None

def main():
    logging.info("Loading v2 dataset...")
    df = pd.read_csv(DATA_PATH)
    df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
    df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist_dt'], format='mixed')

    # Engineer exact same 15 features
    logging.info("Engineering 15 features exactly as v4...")
    # Time features
    df['hour'] = df['timestamp_ist_dt'].dt.hour
    df['day_of_week'] = df['timestamp_ist_dt'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Electrical features
    df['apparent_power'] = df['voltage'] * df['current']
    df['reactive_power'] = np.sqrt(np.maximum(df['apparent_power']**2 - df['power']**2, 0))
    df['current_pf'] = df['current'] * df['pf']
    df['power_per_voltage'] = df['power'] / df['voltage'].replace(0, np.nan)
    df['power_per_voltage'] = df['power_per_voltage'].fillna(0)

    features = [
        'voltage', 'current', 'power', 'pf', 'frequency', 'relay',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
        'apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage'
    ]
    
    X = df[features]
    y = df['label']
    CLASSES = sorted(y.unique().tolist())
    
    logging.info(f"Classes found: {CLASSES}")

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(df, y))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train model (identical hyperparams - Default LightGBM)
    logging.info("Training LightGBM_v5_combined...")
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Inference
    t0 = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - t0

    # Evaluation
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    weighted_f1 = f1_score(y_val, y_pred, average='weighted')
    macro_prec = precision_score(y_val, y_pred, average='macro')
    macro_rec = recall_score(y_val, y_pred, average='macro')
    
    per_class_rec = recall_score(y_val, y_pred, average=None, labels=CLASSES)
    per_class_prec = precision_score(y_val, y_pred, average=None, labels=CLASSES)
    per_class_f1 = f1_score(y_val, y_pred, average=None, labels=CLASSES)
    
    logging.info(f"Macro F1: {macro_f1:.4f}")
    logging.info(f"Weighted F1: {weighted_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=CLASSES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix: LightGBM_v5_combined')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cm_LightGBM_v5_combined.png"), dpi=150)
    plt.close()

    # Feature importances
    feat_imp = dict(zip(features, model.feature_importances_.tolist()))
    sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])
    plt.figure(figsize=(12, 6))
    plt.barh([x[0] for x in sorted_imp][::-1], [x[1] for x in sorted_imp][::-1], color='#4CAF50')
    plt.title('Feature Importance: LightGBM_v5_combined')
    plt.xlabel('Split Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feat_imp_LightGBM_v5_combined.png"), dpi=150)
    plt.close()

    # Save model
    model_path = os.path.join(MODEL_DIR, "LightGBM_v5_combined.joblib")
    joblib.dump(model, model_path)
    
    # Save metadata
    meta = {
        "version": "LightGBM_v5_combined",
        "training_date": "2026-07-30",
        "features": features,
        "hyperparameters": "Default LightGBM",
        "validation_strategy": "Stratified 80/20",
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class_metrics": {
            c: {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4)
            } for c, p, r, f in zip(CLASSES, per_class_prec, per_class_rec, per_class_f1)
        },
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "feature_importance": {k: int(v) for k, v in sorted_imp}
    }
    with open(os.path.join(MODEL_DIR, "LightGBM_v5_combined_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    logging.info("EXP-005 Complete. Artifacts saved.")

if __name__ == "__main__":
    main()
