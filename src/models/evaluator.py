from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_predictions(y_true, y_pred, label_encoder=None):
    """
    Computes standard classification metrics.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_recall": per_class_recall.tolist()
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, model_name, experiment_name, output_dir):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name} ({experiment_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/cm_{model_name.replace(' ', '_')}_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

def plot_feature_importances(importances, feature_names, model_name, experiment_name, output_dir):
    """
    Generates and saves a feature importance plot.
    """
    if importances is None:
        return
        
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances: {model_name} ({experiment_name})')
    plt.bar(range(len(importances)), sorted_importances, align="center")
    plt.xticks(range(len(importances)), sorted_features, rotation=45, ha='right')
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/feat_imp_{model_name.replace(' ', '_')}_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
