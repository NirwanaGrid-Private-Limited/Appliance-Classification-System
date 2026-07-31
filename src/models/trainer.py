import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from src.models.baselines import get_baseline_models
from src.models.evaluator import evaluate_predictions, plot_confusion_matrix, plot_feature_importances
import logging

def check_group_kfold_class_coverage(X, y, groups, n_splits=5):
    """
    Checks if all folds in GroupKFold contain all classes in both train and val.
    """
    gkf = GroupKFold(n_splits=n_splits)
    total_classes = set(y)
    missing_classes_detected = False
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_classes = set(y.iloc[train_idx])
        val_classes = set(y.iloc[val_idx])
        
        if len(train_classes) < len(total_classes) or len(val_classes) < len(total_classes):
            missing_classes_detected = True
            logging.warning(f"Fold {fold}: Missing classes detected!")
            logging.warning(f"  Train classes: {train_classes}")
            logging.warning(f"  Val classes: {val_classes}")
            
    return not missing_classes_detected

def train_and_evaluate(X, y, groups, experiment_name, output_dir, use_class_weight=False, cv_strategy="groupkfold"):
    """
    Trains baseline models and evaluates them using the specified CV strategy.
    """
    logging.info(f"--- Starting Experiment: {experiment_name} | CV Strategy: {cv_strategy} ---")
    
    class_weight = 'balanced' if use_class_weight else None
    models = get_baseline_models(class_weight=class_weight)
    
    results = {}
    classes = sorted(list(set(y)))
    
    if cv_strategy == "groupkfold":
        # Check coverage
        coverage_ok = check_group_kfold_class_coverage(X, y, groups)
        if not coverage_ok:
            logging.warning("GroupKFold does not cover all classes in all folds. This is a severe limitation for multi-class.")
        cv = GroupKFold(n_splits=5)
        splits = list(cv.split(X, y, groups))
    elif cv_strategy == "stratified_80_20":
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        splits = list(cv.split(X, y))
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")

    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        
        train_time_total = 0
        inference_time_total = 0
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Note: For GroupKFold, some folds might missing classes in y_train.
            # Tree models might survive (just never predicting the missing class), 
            # but LogisticRegression might complain or fail. We'll handle it gracefully if possible.
            
            t0 = time.time()
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logging.error(f"Model {name} failed to train on fold {fold}: {e}")
                continue
            train_time_total += (time.time() - t0)
            
            t0 = time.time()
            y_pred = model.predict(X_val)
            inference_time_total += (time.time() - t0)
            
            # Accumulate predictions for a global confusion matrix
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            
            metrics = evaluate_predictions(y_val, y_pred)
            fold_metrics.append(metrics)
            
        if len(fold_metrics) == 0:
            logging.error(f"{name} completely failed.")
            continue
            
        # Aggregate metrics
        avg_metrics = {
            "macro_f1": np.mean([m["macro_f1"] for m in fold_metrics]),
            "macro_precision": np.mean([m["macro_precision"] for m in fold_metrics]),
            "macro_recall": np.mean([m["macro_recall"] for m in fold_metrics]),
            "train_time_sec": train_time_total / len(splits),
            "inference_time_sec": inference_time_total / len(splits)
        }
        
        # Per class recall (average across folds)
        per_class_recalls = np.mean([m["per_class_recall"] for m in fold_metrics], axis=0)
        avg_metrics["per_class_recall"] = dict(zip(classes, per_class_recalls))
        
        results[name] = avg_metrics
        
        # Plot confusion matrix on aggregated predictions
        plot_confusion_matrix(all_y_true, all_y_pred, classes, name, experiment_name, output_dir)
        
        # Feature importances (take from the last fold if available)
        if hasattr(model, "feature_importances_"):
            plot_feature_importances(model.feature_importances_, X.columns, name, experiment_name, output_dir)
            
    return results
