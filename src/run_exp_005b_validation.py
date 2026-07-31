"""
EXP-005B: Validation Experiments for Root Cause Analysis
=========================================================
Experiment 1 - Time Feature Ablation:
    Reuses EXACT v5 hyperparameters, dataset, preprocessing, split, and random seed.
    ONLY removes the 5 temporal features (hour_sin, hour_cos, dow_sin, dow_cos, is_weekend).

Experiment 2 - Node Generalization (GroupKFold):
    Reuses EXACT v5 hyperparameters and full 15 features.
    Replaces Stratified 80/20 with GroupKFold grouped by nodeid.

No existing models or reports are overwritten.
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
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# Paths
# ============================================================
DATA_PATH = config.DATA_PROCESSED_DIR / "cleaned_dataset_v2.csv"
MODEL_DIR = os.path.join(config.BASE_DIR, "outputs", "models")
PLOTS_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "exp005b")
REPORTS_DIR = config.REPORTS_DIR
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Exact v5 feature lists
# ============================================================
ALL_15_FEATURES = [
    'voltage', 'current', 'power', 'pf', 'frequency', 'relay',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage'
]

TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']

ELECTRICAL_ONLY_FEATURES = [f for f in ALL_15_FEATURES if f not in TEMPORAL_FEATURES]


def prepare_dataset():
    """Load and engineer features identically to run_exp_005_heater.py."""
    logging.info("Loading cleaned_dataset_v2.csv...")
    df = pd.read_csv(DATA_PATH)
    df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
    df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist_dt'], format='mixed')

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

    return df


def get_v5_model():
    """Return an LGBMClassifier with EXACT v5 hyperparameters."""
    return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def evaluate_model(y_true, y_pred, classes):
    """Return a dict of per-class and macro metrics."""
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=classes)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=classes)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=classes)

    per_class = {}
    for c, p, r, f in zip(classes, per_class_prec, per_class_rec, per_class_f1):
        per_class[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": classes
    }


def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


# ============================================================
# EXPERIMENT 1: TIME FEATURE ABLATION
# ============================================================
def run_experiment_1(df):
    logging.info("=" * 60)
    logging.info("EXPERIMENT 1: TIME FEATURE ABLATION")
    logging.info("=" * 60)
    logging.info(f"Using {len(ELECTRICAL_ONLY_FEATURES)} electrical-only features: {ELECTRICAL_ONLY_FEATURES}")
    logging.info(f"Removed temporal features: {TEMPORAL_FEATURES}")

    X = df[ELECTRICAL_ONLY_FEATURES]
    y = df['label']
    CLASSES = sorted(y.unique().tolist())

    # Exact same Stratified 80/20 split with same random_state
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(df, y))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train with EXACT v5 hyperparameters
    model = get_v5_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - t0

    metrics = evaluate_model(y_val, y_pred, CLASSES)
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["inference_time_sec"] = round(inference_time, 4)
    metrics["features_used"] = ELECTRICAL_ONLY_FEATURES
    metrics["features_removed"] = TEMPORAL_FEATURES

    # Feature importance
    feat_imp = dict(zip(ELECTRICAL_ONLY_FEATURES, model.feature_importances_.tolist()))
    metrics["feature_importance"] = {k: int(v) for k, v in sorted(feat_imp.items(), key=lambda x: -x[1])}

    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, CLASSES, "Confusion Matrix: EXP-005B Time Ablation", "cm_exp005b_ablation.png")

    # Save model (new artifact, not overwriting anything)
    joblib.dump(model, os.path.join(MODEL_DIR, "LightGBM_v5B_Ablation.joblib"))

    logging.info(f"EXP-1 Macro F1: {metrics['macro_f1']}")
    logging.info(f"EXP-1 Per-class: {json.dumps(metrics['per_class_metrics'], indent=2)}")

    return metrics


# ============================================================
# EXPERIMENT 2: NODE GENERALIZATION (GroupKFold)
# ============================================================
def run_experiment_2(df):
    logging.info("=" * 60)
    logging.info("EXPERIMENT 2: NODE GENERALIZATION (GroupKFold)")
    logging.info("=" * 60)

    X = df[ALL_15_FEATURES]
    y = df['label']
    groups = df['nodeid']
    CLASSES = sorted(y.unique().tolist())
    unique_nodes = sorted(df['nodeid'].unique().tolist())

    logging.info(f"Unique nodes: {unique_nodes}")
    logging.info(f"Using all 15 features, GroupKFold with {len(unique_nodes)} groups")

    gkf = GroupKFold(n_splits=len(unique_nodes))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_nodes = sorted(df.iloc[train_idx]['nodeid'].unique().tolist())
        test_nodes = sorted(df.iloc[val_idx]['nodeid'].unique().tolist())

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Exact v5 hyperparameters
        model = get_v5_model()
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_val)
        metrics = evaluate_model(y_val, y_pred, CLASSES)
        metrics["fold"] = fold_idx + 1
        metrics["train_nodes"] = train_nodes
        metrics["test_nodes"] = test_nodes
        metrics["train_samples"] = len(train_idx)
        metrics["test_samples"] = len(val_idx)
        metrics["train_time_sec"] = round(train_time, 4)

        # Check which classes are actually present in test set
        test_classes_present = sorted(y_val.unique().tolist())
        metrics["test_classes_present"] = test_classes_present

        logging.info(f"Fold {fold_idx+1}: Train={train_nodes}, Test={test_nodes}, "
                     f"Macro F1={metrics['macro_f1']}, "
                     f"Test classes: {test_classes_present}")

        # Plot per-fold confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, CLASSES,
                              f"CM Fold {fold_idx+1}: Test={test_nodes}",
                              f"cm_groupkfold_fold{fold_idx+1}.png")

        fold_results.append(metrics)

    return fold_results


def main():
    df = prepare_dataset()

    results = {}

    # Experiment 1
    results["experiment_1_ablation"] = run_experiment_1(df)

    # Experiment 2
    results["experiment_2_groupkfold"] = run_experiment_2(df)

    # Save all metrics
    output_path = os.path.join(REPORTS_DIR, "exp005b_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"All metrics saved to {output_path}")
    logging.info("EXP-005B Validation Complete!")


if __name__ == "__main__":
    main()
