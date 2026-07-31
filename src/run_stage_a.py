import pandas as pd
import numpy as np
import logging
import os
import json
from src.models.trainer import train_and_evaluate, check_group_kfold_class_coverage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    
    # Encode relay
    df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
    
    # Target and Groups
    y = df['label']
    groups = df['nodeid']
    
    # Drop leakage and non-feature columns
    cols_to_drop = ['label', 'nodeid', 'source_file', 'timestamp_ist', 'timestamp_ist_dt']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return X, y, groups

def generate_report(all_results, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Baseline Model Benchmark Report (Stage A)\n\n")
        f.write("This report presents the Phase 2 Stage A baseline metrics.\n\n")
        
        for exp_name, results in all_results.items():
            f.write(f"## Experiment: {exp_name}\n")
            
            # Markdown table
            f.write("| Model | Macro F1 | Macro Precision | Macro Recall | Train Time (s) | Inference Time (s) |\n")
            f.write("|-------|----------|-----------------|--------------|----------------|--------------------|\n")
            
            for model_name, metrics in results.items():
                f.write(f"| {model_name} | {metrics['macro_f1']:.4f} | {metrics['macro_precision']:.4f} | "
                        f"{metrics['macro_recall']:.4f} | {metrics['train_time_sec']:.4f} | {metrics['inference_time_sec']:.4f} |\n")
            
            f.write("\n### Per-Class Recall\n")
            f.write("| Model | ")
            # Get classes from first model
            first_model = list(results.keys())[0]
            classes = list(results[first_model]['per_class_recall'].keys())
            f.write(" | ".join(classes) + " |\n")
            f.write("|-------|" + "|".join(["---"] * len(classes)) + "|\n")
            
            for model_name, metrics in results.items():
                recalls = [f"{metrics['per_class_recall'][c]:.4f}" for c in classes]
                f.write(f"| {model_name} | " + " | ".join(recalls) + " |\n")
            
            f.write("\n---\n")

def main():
    data_path = r"c:\Users\saury\Desktop\nirvana_grid\data\processed\cleaned_dataset.csv"
    output_dir = r"c:\Users\saury\Desktop\nirvana_grid\outputs\plots\models"
    report_path = r"c:\Users\saury\Desktop\nirvana_grid\outputs\reports\baseline_benchmark.md"
    
    X, y, groups = load_and_preprocess(data_path)
    all_results = {}
    
    logging.info("Starting Phase 2 Stage A...")
    
    # 1. GroupKFold Default
    logging.info("Checking GroupKFold class coverage...")
    coverage_ok = check_group_kfold_class_coverage(X, y, groups)
    
    if coverage_ok:
        res_gkf = train_and_evaluate(X, y, groups, "GroupKFold_Default", output_dir, use_class_weight=False, cv_strategy="groupkfold")
        all_results["GroupKFold_Default"] = res_gkf
    else:
        logging.warning("GroupKFold cannot be used as primary due to missing classes. Logging limitation and using Stratified 80/20.")
        all_results["GroupKFold_Limitation"] = {"Note": "GroupKFold misses classes in training splits. Fallback to Stratified 80/20."}
        
    # 2. Stratified 80/20 Default
    res_strat = train_and_evaluate(X, y, groups, "Stratified_Default", output_dir, use_class_weight=False, cv_strategy="stratified_80_20")
    all_results["Stratified_Default"] = res_strat
    
    # Check minority recall to see if class weighting is needed
    minority_classes = ['Geyser', 'LED']
    needs_weighting = False
    for model, metrics in res_strat.items():
        for mc in minority_classes:
            if metrics['per_class_recall'][mc] < 0.60:  # arbitrary threshold for 'inadequate'
                needs_weighting = True
                
    if needs_weighting:
        logging.info("Minority class performance inadequate. Running class-weighted benchmark...")
        res_weighted = train_and_evaluate(X, y, groups, "Stratified_Weighted", output_dir, use_class_weight=True, cv_strategy="stratified_80_20")
        all_results["Stratified_Weighted"] = res_weighted
    else:
        logging.info("Default model minority performance is adequate. No weighting necessary.")

    # 3. Energy Ablation Study
    logging.info("Running Energy Ablation Study...")
    X_no_energy = X.drop(columns=['energy'])
    res_no_energy = train_and_evaluate(X_no_energy, y, groups, "Stratified_NoEnergy", output_dir, use_class_weight=needs_weighting, cv_strategy="stratified_80_20")
    all_results["Stratified_NoEnergy"] = res_no_energy

    # Generate Report
    # Clean up non-dict results for report generation
    report_data = {k: v for k, v in all_results.items() if isinstance(v, dict) and "Note" not in v}
    generate_report(report_data, report_path)
    
    logging.info(f"Phase 2 Stage A complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
