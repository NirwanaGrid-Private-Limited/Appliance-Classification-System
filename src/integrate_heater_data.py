import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data_file
from cleaner import clean_data
from merger import merge_datasets
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("heater_integration")

def main():
    heater_raw_path = config.DATA_RAW_DIR / "Heater_NODE_40.csv"
    master_path = config.DATA_PROCESSED_DIR / "cleaned_dataset.csv"
    output_path = config.DATA_PROCESSED_DIR / "cleaned_dataset_v2.csv"
    
    logger.info("Loading Heater data...")
    heater_df = load_data_file(heater_raw_path)
    if heater_df is None:
        logger.error("Failed to load heater data.")
        return
        
    logger.info("Cleaning Heater data...")
    cleaned_heater_df, invalid_rows, stats = clean_data(heater_df, heater_raw_path.name)
    logger.info(f"Heater cleaning stats: {stats}")
    
    logger.info("Loading previous master dataset...")
    master_df = pd.read_csv(master_path)
    
    # Check if 'label' exists in master_df (it should)
    if 'label' not in master_df.columns:
        logger.error("No label column in master_df!")
        return
        
    # Merge datasets
    logger.info("Merging datasets...")
    # Convert master_df timestamp_ist_dt to datetime if it's string
    master_df['timestamp_ist_dt'] = pd.to_datetime(master_df['timestamp_ist_dt'], errors='coerce')
    
    v2_df = merge_datasets([master_df, cleaned_heater_df])
    
    logger.info("Saving v2 dataset...")
    v2_df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned_dataset_v2.csv with {len(v2_df)} rows.")
    
    # Class distribution analysis
    print("\n" + "=" * 50)
    print("DATASET v2 CLASS DISTRIBUTION")
    print("=" * 50)
    counts = v2_df['label'].value_counts()
    total = len(v2_df)
    
    print(f"{'Appliance':20} | {'Samples':10} | {'Percentage':10}")
    print("-" * 50)
    for appliance, count in counts.items():
        pct = (count / total) * 100
        print(f"{appliance:20} | {count:<10} | {pct:>5.2f}%")
    print("=" * 50)
    
    # Save the distribution for the report
    with open(config.REPORTS_DIR / "class_distribution_v2.md", "w") as f:
        f.write("## Dataset v2 Class Distribution\n\n")
        f.write("| Appliance | Samples | Percentage |\n")
        f.write("|---|---|---|\n")
        for appliance, count in counts.items():
            pct = (count / total) * 100
            f.write(f"| {appliance} | {count} | {pct:.2f}% |\n")

if __name__ == "__main__":
    main()
