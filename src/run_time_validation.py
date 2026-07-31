"""
Phase 2 Final Validation Before Phase 3: Temporal Generalization & Ablation Study
Runs:
1. Time Generalization Validation (Leave-One-Day-Out on major days + Pooled LODO, Weekday vs Weekend, Time-of-Day)
2. Time Feature Ablation (Model A: Electrical Only, Model B: Electrical + Time, Model C: Time Only)
"""
import pandas as pd
import numpy as np
import os
import json
import time
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r"c:\Users\saury\Desktop\nirvana_grid\data\processed\cleaned_dataset.csv"
PLOTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\plots\time_validation"
REPORTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\reports"
RANDOM_STATE = 42
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Load Data
logging.info("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist_dt'], format='mixed')

# Feature engineering
df['hour'] = df['timestamp_ist_dt'].dt.hour
df['minute'] = df['timestamp_ist_dt'].dt.minute
df['day_of_week'] = df['timestamp_ist_dt'].dt.dayofweek
df['date'] = df['timestamp_ist_dt'].dt.date
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['apparent_power'] = df['voltage'] * df['current']
df['reactive_power'] = np.sqrt(np.maximum(df['apparent_power']**2 - df['power']**2, 0))
df['current_pf'] = df['current'] * df['pf']
df['power_per_voltage'] = (df['power'] / df['voltage'].replace(0, np.nan)).fillna(0)

CLASSES = sorted(df['label'].unique().tolist())
logging.info(f"Classes: {CLASSES}, Total Records: {len(df)}")

base_features = ['voltage', 'current', 'power', 'pf', 'frequency', 'relay']
elec_features = ['apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage']
time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']

MODEL_A_FEAT = base_features + elec_features                      # Electrical Only (10 features)
MODEL_B_FEAT = base_features + elec_features + time_features      # Electrical + Time (15 features - Production Candidate)
MODEL_C_FEAT = time_features                                      # Time Only (5 features)

# ============================================================
# Helper function to evaluate split
# ============================================================
def eval_split(train_mask, test_mask, feature_cols, split_name):
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'label']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'label']
    
    if len(X_train) == 0 or len(X_test) == 0:
        return None
    
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_rec = recall_score(y_test, y_pred, average='macro')
    per_class_rec = recall_score(y_test, y_pred, average=None, labels=CLASSES)
    
    n_errors = int((y_test != y_pred).sum())
    error_rate = float(n_errors / len(y_test))
    
    rec_dict = {}
    for c, r in zip(CLASSES, per_class_rec):
        rec_dict[c] = float(r)
        
    return {
        "split_name": split_name,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "macro_f1": round(float(macro_f1), 4),
        "macro_recall": round(float(macro_rec), 4),
        "per_class_recall": {k: round(v, 4) for k, v in rec_dict.items()},
        "n_errors": n_errors,
        "error_rate_pct": round(error_rate * 100, 2)
    }

# ============================================================
# EXPERIMENT 1: TIME GENERALIZATION VALIDATION
# ============================================================
logging.info("=" * 60)
logging.info("EXPERIMENT 1: TIME GENERALIZATION VALIDATION")
logging.info("=" * 60)

exp1_results = {
    "lodo_major_days": [],
    "pooled_lodo": {},
    "weekday_weekend": [],
    "time_of_day": []
}

# 1. Leave-One-Day-Out (LODO) Validation on Major Days (>= 500 records)
date_counts = df['date'].value_counts()
major_dates = sorted([d for d, cnt in date_counts.items() if cnt >= 500])
logging.info(f"Running LODO across {len(major_dates)} major full-day captures...")

lodo_f1s = []
for d in major_dates:
    test_mask = (df['date'] == d)
    train_mask = ~test_mask
    res = eval_split(train_mask, test_mask, MODEL_B_FEAT, f"Holdout Day: {d}")
    if res:
        exp1_results["lodo_major_days"].append(res)
        lodo_f1s.append(res["macro_f1"])
        logging.info(f"  Holdout {d} (N={res['test_size']}): Macro F1={res['macro_f1']:.4f}, Errors={res['n_errors']} ({res['error_rate_pct']}%)")

mean_lodo_f1 = float(np.mean(lodo_f1s)) if lodo_f1s else 0.0
logging.info(f"Mean LODO Macro F1 across major days: {mean_lodo_f1:.4f}")

# 2. Weekday vs Weekend Validation
logging.info("Running Weekday vs Weekend Validation...")
weekday_mask = (df['is_weekend'] == 0)
weekend_mask = (df['is_weekend'] == 1)

res_train_wkday = eval_split(weekday_mask, weekend_mask, MODEL_B_FEAT, "Train Weekdays -> Test Weekends")
res_train_wkend = eval_split(weekend_mask, weekday_mask, MODEL_B_FEAT, "Train Weekends -> Test Weekdays")
if res_train_wkday:
    exp1_results["weekday_weekend"].append(res_train_wkday)
    logging.info(f"  Train Weekdays -> Test Weekends: F1={res_train_wkday['macro_f1']:.4f}, Errors={res_train_wkday['error_rate_pct']}%")
if res_train_wkend:
    exp1_results["weekday_weekend"].append(res_train_wkend)
    logging.info(f"  Train Weekends -> Test Weekdays: F1={res_train_wkend['macro_f1']:.4f}, Errors={res_train_wkend['error_rate_pct']}%")

# 3. Time-of-Day Validation
logging.info("Running Time-of-Day Validation...")
daytime_mask = (df['hour'] >= 6) & (df['hour'] < 18)
nighttime_mask = ~daytime_mask

res_day_to_night = eval_split(daytime_mask, nighttime_mask, MODEL_B_FEAT, "Train Day (06-18) -> Test Night (18-06)")
res_night_to_day = eval_split(nighttime_mask, daytime_mask, MODEL_B_FEAT, "Train Night (18-06) -> Test Day (06-18)")
if res_day_to_night:
    exp1_results["time_of_day"].append(res_day_to_night)
    logging.info(f"  Train Day -> Test Night: F1={res_day_to_night['macro_f1']:.4f}, Errors={res_day_to_night['error_rate_pct']}%")
if res_night_to_day:
    exp1_results["time_of_day"].append(res_night_to_day)
    logging.info(f"  Train Night -> Test Day: F1={res_night_to_day['macro_f1']:.4f}, Errors={res_night_to_day['error_rate_pct']}%")

# ============================================================
# EXPERIMENT 2: TIME FEATURE ABLATION (Model A vs Model B vs Model C)
# ============================================================
logging.info("=" * 60)
logging.info("EXPERIMENT 2: TIME FEATURE ABLATION (IDENTICAL VALIDATION DATA)")
logging.info("=" * 60)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(sss.split(df, df['label']))

def train_ablation_model(model_name, feature_cols):
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['label']
    X_val = df.iloc[val_idx][feature_cols]
    y_val = df.iloc[val_idx]['label']
    
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    macro_rec = recall_score(y_val, y_pred, average='macro')
    per_class_rec = recall_score(y_val, y_pred, average=None, labels=CLASSES)
    
    cm = confusion_matrix(y_val, y_pred, labels=CLASSES)
    n_errors = int((y_val != y_pred).sum())
    
    feat_imp = dict(zip(feature_cols, [int(x) for x in model.feature_importances_.tolist()]))
    sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix: {model_name}\n(Macro F1 = {macro_f1:.4f})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"), dpi=150)
    plt.close()
    
    return {
        "model_name": model_name,
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "macro_f1": round(float(macro_f1), 4),
        "macro_recall": round(float(macro_rec), 4),
        "per_class_recall": {c: round(float(r), 4) for c, r in zip(CLASSES, per_class_rec)},
        "n_errors": n_errors,
        "error_rate_pct": round(n_errors / len(y_val) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {k: int(v) for k, v in sorted_imp}
    }

model_a = train_ablation_model("Model A (Electrical Only)", MODEL_A_FEAT)
model_b = train_ablation_model("Model B (Electrical + Time)", MODEL_B_FEAT)
model_c = train_ablation_model("Model C (Time Only)", MODEL_C_FEAT)

logging.info(f"Model A (Electrical Only):   Macro F1 = {model_a['macro_f1']:.4f}, Errors = {model_a['n_errors']}")
logging.info(f"Model B (Electrical + Time): Macro F1 = {model_b['macro_f1']:.4f}, Errors = {model_b['n_errors']}")
logging.info(f"Model C (Time Only):         Macro F1 = {model_c['macro_f1']:.4f}, Errors = {model_c['n_errors']}")

# Save all results to JSON
all_results = {
    "experiment_1_time_generalization": exp1_results,
    "mean_lodo_macro_f1": round(mean_lodo_f1, 4),
    "experiment_2_ablation": {
        "Model_A_Electrical_Only": model_a,
        "Model_B_Electrical_Time": model_b,
        "Model_C_Time_Only": model_c
    }
}

out_path = os.path.join(REPORTS_DIR, "time_validation_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=4)

logging.info(f"Saved complete validation results to {out_path}")
