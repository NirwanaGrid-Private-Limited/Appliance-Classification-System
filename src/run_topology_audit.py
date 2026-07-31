"""
Dataset Topology Audit for EXP-005
===================================
Generates Node x Appliance matrix and verifies GroupKFold class coverage.
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.model_selection import GroupKFold
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = config.DATA_PROCESSED_DIR / "cleaned_dataset_v2.csv"
REPORTS_DIR = config.REPORTS_DIR

def main():
    df = pd.read_csv(DATA_PATH)
    
    # ============================================================
    # 1. Node x Appliance Matrix
    # ============================================================
    logging.info("=== NODE x APPLIANCE MATRIX ===")
    pivot = pd.crosstab(df['nodeid'], df['label'])
    logging.info(f"\n{pivot.to_string()}")
    
    # Per-node totals
    node_totals = df['nodeid'].value_counts().sort_index()
    logging.info(f"\nPer-node totals:\n{node_totals.to_string()}")
    
    # Per-appliance: how many unique nodes
    appliance_nodes = df.groupby('label')['nodeid'].nunique()
    logging.info(f"\nUnique nodes per appliance:\n{appliance_nodes.to_string()}")
    
    # Detailed: which specific nodes per appliance
    appliance_node_detail = df.groupby('label')['nodeid'].apply(lambda x: sorted(x.unique().tolist()))
    logging.info(f"\nSpecific nodes per appliance:\n{appliance_node_detail.to_string()}")
    
    # ============================================================
    # 2. Confounding Analysis
    # ============================================================
    logging.info("\n=== CONFOUNDING ANALYSIS ===")
    # For each appliance, check if it appears on >1 node
    for appliance in sorted(df['label'].unique()):
        nodes = sorted(df[df['label'] == appliance]['nodeid'].unique())
        counts = df[df['label'] == appliance].groupby('nodeid').size().to_dict()
        is_confounded = len(nodes) == 1
        logging.info(f"{appliance}: {len(nodes)} node(s) {nodes} | Counts: {counts} | "
                     f"Confounded with NodeID: {is_confounded}")
    
    # ============================================================
    # 3. GroupKFold Class Coverage Audit
    # ============================================================
    logging.info("\n=== GROUPKFOLD CLASS COVERAGE AUDIT ===")
    y = df['label']
    groups = df['nodeid']
    unique_nodes = sorted(df['nodeid'].unique())
    gkf = GroupKFold(n_splits=len(unique_nodes))
    
    fold_audit = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        train_nodes = sorted(df.iloc[train_idx]['nodeid'].unique().tolist())
        test_nodes = sorted(df.iloc[val_idx]['nodeid'].unique().tolist())
        train_classes = sorted(df.iloc[train_idx]['label'].unique().tolist())
        test_classes = sorted(df.iloc[val_idx]['label'].unique().tolist())
        
        # Key question: are the test classes represented in training?
        missing_in_train = [c for c in test_classes if c not in train_classes]
        present_in_train = [c for c in test_classes if c in train_classes]
        
        # For classes present in train, how many training samples?
        train_class_counts = df.iloc[train_idx]['label'].value_counts().to_dict()
        test_class_counts = df.iloc[val_idx]['label'].value_counts().to_dict()
        
        fold_info = {
            "fold": fold_idx + 1,
            "train_nodes": train_nodes,
            "test_nodes": test_nodes,
            "train_classes": train_classes,
            "test_classes": test_classes,
            "missing_classes_in_train": missing_in_train,
            "classes_present_in_both": present_in_train,
            "train_class_counts": train_class_counts,
            "test_class_counts": test_class_counts
        }
        fold_audit.append(fold_info)
        
        logging.info(f"\nFold {fold_idx+1}:")
        logging.info(f"  Train Nodes: {train_nodes}")
        logging.info(f"  Test Nodes:  {test_nodes}")
        logging.info(f"  Train Classes: {train_classes} (counts: {train_class_counts})")
        logging.info(f"  Test Classes:  {test_classes} (counts: {test_class_counts})")
        if missing_in_train:
            logging.info(f"  *** MISSING IN TRAIN: {missing_in_train} ***")
            logging.info(f"      => Test classes have ZERO training examples. "
                         f"Performance collapse is expected (not a generalization failure).")
        else:
            logging.info(f"  All test classes present in training.")
            logging.info(f"      => Performance collapse IS a genuine generalization failure.")
    
    # ============================================================
    # 4. Save results
    # ============================================================
    results = {
        "node_appliance_matrix": pivot.to_dict(),
        "node_totals": node_totals.to_dict(),
        "unique_nodes_per_appliance": appliance_nodes.to_dict(),
        "specific_nodes_per_appliance": {k: v for k, v in appliance_node_detail.items()},
        "groupkfold_class_coverage_audit": fold_audit
    }
    
    output_path = os.path.join(REPORTS_DIR, "dataset_topology_audit.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"\nAudit saved to {output_path}")
    logging.info("Dataset Topology Audit Complete!")

if __name__ == "__main__":
    main()
