import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from collections import defaultdict
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = r"c:\Users\saury\Desktop\nirvana_grid\data\processed\cleaned_dataset_v2.csv"
MODEL_PATH = r"c:\Users\saury\Desktop\nirvana_grid\outputs\models\LightGBM_v5_combined.joblib"
PLOTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\plots\rca"
REPORTS_DIR = r"c:\Users\saury\Desktop\nirvana_grid\outputs\reports"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURES = [
    'voltage', 'current', 'power', 'pf', 'frequency', 'relay',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage'
]
RANDOM_STATE = 42

def task1_distribution_analysis(df, results):
    logging.info("Task 1: Dataset Distribution Analysis")
    stats = {}
    for c in FEATURES:
        stats[c] = {
            "mean": float(df[c].mean()), "median": float(df[c].median()),
            "std": float(df[c].std()), "min": float(df[c].min()), "max": float(df[c].max()),
            "25%": float(df[c].quantile(0.25)), "50%": float(df[c].quantile(0.5)),
            "75%": float(df[c].quantile(0.75)), "missing": int(df[c].isna().sum())
        }
    results['feature_stats'] = stats
    
    # Plotting distributions for Heater vs Geyser
    subset = df[df['label'].isin(['Heater', 'Geyser'])]
    for feat in ['voltage', 'current', 'power', 'pf', 'reactive_power', 'apparent_power']:
        plt.figure(figsize=(8,4))
        sns.boxplot(data=subset, x='label', y=feat)
        plt.title(f"{feat} Boxplot: Heater vs Geyser")
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{feat}_heater_geyser.png"))
        plt.close()

def task2_feature_separability(df):
    logging.info("Task 2: Feature Separability")
    subset = df.sample(min(5000, len(df)), random_state=RANDOM_STATE)
    X = subset[FEATURES].fillna(0)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=subset['label'], palette="tab10", s=10)
    plt.title("PCA of 15 Features")
    plt.savefig(os.path.join(PLOTS_DIR, "pca_plot.png"))
    plt.close()
    
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, max_iter=300)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=subset['label'], palette="tab10", s=10)
    plt.title("t-SNE of 15 Features")
    plt.savefig(os.path.join(PLOTS_DIR, "tsne_plot.png"))
    plt.close()

def task3_5_confusion_and_probs(model, df, val_idx, results):
    logging.info("Task 3 & 5: Confusion, Probabilities, and Task 1A: Misclassifications")
    X_val = df.iloc[val_idx][FEATURES]
    y_val = df.iloc[val_idx]['label'].values
    
    probs = model.predict_proba(X_val)
    preds = model.classes_[np.argmax(probs, axis=1)]
    confidences = np.max(probs, axis=1)
    
    df_val = df.iloc[val_idx].copy()
    df_val['pred'] = preds
    df_val['confidence'] = confidences
    
    classes = model.classes_
    results['confusion_matrix'] = {}
    
    for c1 in classes:
        for c2 in classes:
            mask = (df_val['label'] == c1) & (df_val['pred'] == c2)
            cnt = mask.sum()
            avg_conf = df_val.loc[mask, 'confidence'].mean() if cnt > 0 else 0
            results['confusion_matrix'][f"{c1}_to_{c2}"] = {
                "count": int(cnt),
                "avg_confidence": float(avg_conf)
            }
            
    # Task 1A Misclassified Samples (Heater vs Geyser)
    mask_gh = (df_val['label'] == 'Geyser') & (df_val['pred'] == 'Heater')
    mask_hg = (df_val['label'] == 'Heater') & (df_val['pred'] == 'Geyser')
    results['misclassified'] = {
        'Geyser_to_Heater': df_val.loc[mask_gh, ['label', 'pred', 'confidence'] + FEATURES].head(5).to_dict(orient='records'),
        'Heater_to_Geyser': df_val.loc[mask_hg, ['label', 'pred', 'confidence'] + FEATURES].head(5).to_dict(orient='records')
    }
    
    # Task 5: Probability distribution
    plt.figure(figsize=(8,4))
    sns.histplot(df_val[df_val['label'] != df_val['pred']]['confidence'], bins=20, color='red', alpha=0.5, label='Incorrect')
    sns.histplot(df_val[df_val['label'] == df_val['pred']]['confidence'], bins=20, color='green', alpha=0.5, label='Correct')
    plt.title("Prediction Confidence Distribution")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "confidence_dist.png"))
    plt.close()

def task4_shap(model, df, val_idx, results):
    logging.info("Task 4: SHAP Analysis via LightGBM pred_contrib")
    X_val = df.iloc[val_idx][FEATURES]
    contribs = model.predict(X_val, pred_contrib=True)
    
    # In LightGBM multiclass, contribs is shape (n_samples, num_classes * (num_features + 1))
    num_classes = len(model.classes_)
    num_features = len(FEATURES)
    
    shap_results = {}
    for i, cls in enumerate(model.classes_):
        start = i * (num_features + 1)
        end = start + num_features
        class_contribs = contribs[:, start:end]
        mean_abs_shap = np.abs(class_contribs).mean(axis=0)
        shap_results[cls] = dict(zip(FEATURES, mean_abs_shap))
        
    results['shap_importance'] = shap_results

def task6_class_imbalance(df, results):
    logging.info("Task 6: Class Imbalance")
    counts = df['label'].value_counts()
    results['class_counts'] = counts.to_dict()
    results['class_percentages'] = (counts / len(df) * 100).to_dict()

def task7_model_capacity(model, results):
    logging.info("Task 7: Model Capacity")
    booster = model.booster_
    results['model_capacity'] = {
        'num_trees': booster.num_trees(),
        'num_leaves_approx': booster.num_trees() * model.num_leaves,
    }

def task8a_node_generalization(df, train_idx, val_idx, results):
    logging.info("Task 8A: Node-Level Generalization Audit")
    if 'nodeid' in df.columns:
        node_counts = df['nodeid'].value_counts().to_dict()
        train_nodes = df.iloc[train_idx]['nodeid'].value_counts().to_dict()
        val_nodes = df.iloc[val_idx]['nodeid'].value_counts().to_dict()
        
        # Check where Heater comes from
        heater_nodes = df[df['label'] == 'Heater']['nodeid'].value_counts().to_dict()
        results['node_audit'] = {
            'total_samples_per_node': node_counts,
            'train_samples_per_node': train_nodes,
            'val_samples_per_node': val_nodes,
            'heater_nodes': heater_nodes
        }

def task9_label_audit(df, results):
    logging.info("Task 9: Label Quality Audit for Heater")
    heater_df = df[df['label'] == 'Heater']
    duplicates = heater_df.duplicated(subset=FEATURES).sum()
    results['label_audit_heater'] = {
        'total': len(heater_df),
        'exact_duplicates_in_features': int(duplicates)
    }

def main():
    logging.info("Loading Data and Model...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist_dt'], format='mixed')
    df['hour'] = df['timestamp_ist_dt'].dt.hour
    df['day_of_week'] = df['timestamp_ist_dt'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['apparent_power'] = df['voltage'] * df['current']
    df['reactive_power'] = np.sqrt(np.maximum(df['apparent_power']**2 - df['power']**2, 0))
    df['current_pf'] = df['current'] * df['pf']
    df['power_per_voltage'] = df['power'] / df['voltage'].replace(0, np.nan)
    df['power_per_voltage'] = df['power_per_voltage'].fillna(0)
    df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
    
    model = joblib.load(MODEL_PATH)
    
    y = df['label']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(df, y))
    
    results = {}
    
    task1_distribution_analysis(df, results)
    task2_feature_separability(df)
    task3_5_confusion_and_probs(model, df, val_idx, results)
    task4_shap(model, df, val_idx, results)
    task6_class_imbalance(df, results)
    task7_model_capacity(model, results)
    task8a_node_generalization(df, train_idx, val_idx, results)
    task9_label_audit(df, results)
    
    with open(os.path.join(REPORTS_DIR, "rca_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    logging.info("RCA Script Execution Complete!")

if __name__ == "__main__":
    main()
