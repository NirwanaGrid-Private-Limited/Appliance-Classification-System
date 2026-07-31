"""
EXP-005C: Node Feature Ablation Experiment
============================================
Hypothesis: The model's high Stratified 80/20 performance is primarily driven
by node-specific electrical baseline features (voltage, frequency, pf, and
their derived features).

This experiment removes those features and evaluates using BOTH:
  1. Stratified 80/20 (same as v5)
  2. GroupKFold (grouped by nodeid)

Uses EXACT v5 hyperparameters, dataset, preprocessing, and random seed.
Only independent variable: removal of node-baseline features.

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
    confusion_matrix
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
PLOTS_DIR = os.path.join(config.BASE_DIR, "outputs", "plots", "exp005c")
REPORTS_DIR = config.REPORTS_DIR
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Feature definitions
# ============================================================
# Node-baseline features to REMOVE
NODE_BASELINE_FEATURES = [
    'voltage',       # Grid voltage varies by household
    'frequency',     # Grid frequency varies by location
    'pf',            # Power factor partially reflects node wiring
    'apparent_power', # = voltage * current (derived from voltage)
    'reactive_power', # = sqrt(apparent_power^2 - power^2) (derived from voltage)
    'current_pf',    # = current * pf (derived from pf)
    'power_per_voltage'  # = power / voltage (derived from voltage)
]

# Appliance-behavior features to RETAIN
RETAINED_FEATURES = [
    'current',       # Driven by appliance load
    'power',         # Driven by appliance consumption
    'relay',         # On/off state
    'hour_sin',      # Temporal
    'hour_cos',      # Temporal
    'dow_sin',       # Temporal
    'dow_cos',       # Temporal
    'is_weekend'     # Temporal
]

ALL_15_FEATURES = [
    'voltage', 'current', 'power', 'pf', 'frequency', 'relay',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage'
]


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

    # Electrical features (still computed for completeness, but not all used)
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
        per_class[c] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4)
        }

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
# PART 1: STRATIFIED 80/20 EVALUATION
# ============================================================
def run_stratified_evaluation(df):
    logging.info("=" * 60)
    logging.info("PART 1: STRATIFIED 80/20 WITH NODE-BASELINE ABLATION")
    logging.info("=" * 60)
    logging.info(f"Retained features ({len(RETAINED_FEATURES)}): {RETAINED_FEATURES}")
    logging.info(f"Removed features ({len(NODE_BASELINE_FEATURES)}): {NODE_BASELINE_FEATURES}")

    X = df[RETAINED_FEATURES]
    y = df['label']
    CLASSES = sorted(y.unique().tolist())

    # Exact same split as v5
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(df, y))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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
    metrics["features_used"] = RETAINED_FEATURES
    metrics["features_removed"] = NODE_BASELINE_FEATURES

    # Feature importance
    feat_imp = dict(zip(RETAINED_FEATURES, model.feature_importances_.tolist()))
    metrics["feature_importance"] = {
        k: int(v) for k, v in sorted(feat_imp.items(), key=lambda x: -x[1])
    }

    # SHAP via pred_contrib
    logging.info("Computing SHAP (pred_contrib) for stratified split...")
    X_val_arr = X_val.values
    contribs = model.predict(X_val_arr, pred_contrib=True)
    num_classes = len(CLASSES)
    num_features = len(RETAINED_FEATURES)

    shap_per_class = {}
    for i, cls in enumerate(model.classes_):
        start = i * (num_features + 1)
        end = start + num_features
        class_contribs = contribs[:, start:end]
        mean_abs_shap = np.abs(class_contribs).mean(axis=0)
        shap_per_class[cls] = {
            feat: round(float(val), 6)
            for feat, val in zip(RETAINED_FEATURES, mean_abs_shap)
        }
    metrics["shap_importance"] = shap_per_class

    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, CLASSES,
                          "Confusion Matrix: EXP-005C Stratified (Node-Baseline Ablation)",
                          "cm_exp005c_stratified.png")

    # Feature importance plot
    sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])
    plt.figure(figsize=(10, 5))
    plt.barh([x[0] for x in sorted_imp][::-1],
             [x[1] for x in sorted_imp][::-1], color='#E91E63')
    plt.title('Feature Importance: EXP-005C (Node-Baseline Ablation)')
    plt.xlabel('Split Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feat_imp_exp005c.png"), dpi=150)
    plt.close()

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "LightGBM_v5C_NodeAblation.joblib"))

    logging.info(f"Stratified Macro F1: {metrics['macro_f1']}")
    for c in CLASSES:
        m = metrics['per_class_metrics'][c]
        logging.info(f"  {c}: P={m['precision']} R={m['recall']} F1={m['f1']}")

    return metrics


# ============================================================
# PART 2: GROUPKFOLD EVALUATION
# ============================================================
def run_groupkfold_evaluation(df):
    logging.info("=" * 60)
    logging.info("PART 2: GROUPKFOLD WITH NODE-BASELINE ABLATION")
    logging.info("=" * 60)

    X = df[RETAINED_FEATURES]
    y = df['label']
    groups = df['nodeid']
    CLASSES = sorted(y.unique().tolist())
    unique_nodes = sorted(df['nodeid'].unique().tolist())

    logging.info(f"Unique nodes: {unique_nodes}")

    gkf = GroupKFold(n_splits=len(unique_nodes))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_nodes = sorted(df.iloc[train_idx]['nodeid'].unique().tolist())
        test_nodes = sorted(df.iloc[val_idx]['nodeid'].unique().tolist())

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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
        metrics["test_classes_present"] = sorted(y_val.unique().tolist())

        logging.info(f"Fold {fold_idx+1}: Train={train_nodes}, Test={test_nodes}, "
                     f"Macro F1={metrics['macro_f1']}, "
                     f"Test classes: {metrics['test_classes_present']}")

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, CLASSES,
                              f"CM Fold {fold_idx+1} (Node Ablation): Test={test_nodes}",
                              f"cm_exp005c_gkf_fold{fold_idx+1}.png")

        fold_results.append(metrics)

    return fold_results


def main():
    df = prepare_dataset()

    results = {
        "experiment": "EXP-005C_Node_Feature_Ablation",
        "features_retained": RETAINED_FEATURES,
        "features_removed": NODE_BASELINE_FEATURES,
        "stratified_evaluation": run_stratified_evaluation(df),
        "groupkfold_evaluation": run_groupkfold_evaluation(df)
    }

    output_path = os.path.join(REPORTS_DIR, "exp005c_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"All metrics saved to {output_path}")
    logging.info("EXP-005C Node Feature Ablation Complete!")


if __name__ == "__main__":
    main()
