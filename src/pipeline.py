import pandas as pd
import config
from data_loader import load_all_raw_data
from cleaner import clean_data
from validator import validate_timestamps
from profiler import generate_overall_profile, generate_column_stats, generate_appliance_stats, format_markdown_report
from eda import generate_all_plots
from merger import merge_datasets, save_master_dataset
from datetime import datetime

logger = config.setup_logger("pipeline", config.LOGS_DIR / "pipeline.log")

def run_pipeline():
    logger.info("Starting Data Engineering Pipeline...")
    
    # 1. Load Data
    raw_dataframes = load_all_raw_data()
    if not raw_dataframes:
        logger.error("No valid datasets loaded. Exiting.")
        return
        
    cleaned_dfs = []
    all_invalid_rows = []
    cleaning_stats = {}
    ts_stats_all = {}
    
    # 2. Clean and Validate
    for filename, df in raw_dataframes.items():
        cleaned_df, invalid_rows, stats = clean_data(df, filename)
        cleaned_dfs.append(cleaned_df)
        if not invalid_rows.empty:
            all_invalid_rows.append(invalid_rows)
            
        cleaning_stats[filename] = stats
        
        # Timestamp validation on cleaned data
        ts_stats_all[filename] = validate_timestamps(cleaned_df)
        
    # 3. Merge
    master_df = merge_datasets(cleaned_dfs)
    if master_df.empty:
        logger.error("Master dataset is empty. Exiting.")
        return
        
    save_master_dataset(master_df)
    
    # Save all invalid rows
    if all_invalid_rows:
        invalid_df = pd.concat(all_invalid_rows, ignore_index=True)
        invalid_df.to_csv(config.DATA_PROCESSED_DIR / "invalid_rows.csv", index=False)
        logger.info(f"Saved {len(invalid_df)} invalid rows to invalid_rows.csv")
        
    # 4. Generate timestamped reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # A. Cleaning Summary
    cleaning_summary_path = config.REPORTS_DIR / f"cleaning_summary_{timestamp}.md"
    with open(cleaning_summary_path, 'w') as f:
        f.write(f"# Cleaning Summary\n\nGenerated on: {timestamp}\n\n")
        f.write("## File-wise Cleaning Statistics\n\n")
        for filename, stats in cleaning_stats.items():
            f.write(f"### {filename}\n")
            for k, v in stats.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")
            
    # B. Profiling and Appliance-wise Stats
    profile = generate_overall_profile(master_df)
    col_stats = generate_column_stats(master_df)
    ts_overall = validate_timestamps(master_df)
    
    profile_md = format_markdown_report(profile, col_stats, ts_overall, "Master Dataset")
    profile_path = config.REPORTS_DIR / f"data_profile_{timestamp}.md"
    with open(profile_path, 'w') as f:
        f.write(profile_md)
        
    profile_csv_path = config.REPORTS_DIR / f"data_profile_{timestamp}.csv"
    col_stats.to_csv(profile_csv_path, index=False)
    
    appliance_stats = generate_appliance_stats(master_df)
    appliance_stats_path = config.REPORTS_DIR / f"appliance_stats_{timestamp}.csv"
    if not appliance_stats.empty:
        appliance_stats.to_csv(appliance_stats_path, index=False)
        logger.info(f"Appliance statistics saved to {appliance_stats_path.name}")
        
    # C. Data Quality Report
    dq_report_path = config.REPORTS_DIR / f"data_quality_report_{timestamp}.md"
    with open(dq_report_path, 'w') as f:
        f.write(f"# Data Quality Report\n\nGenerated on: {timestamp}\n\n")
        f.write("## 1. Summary\n")
        f.write(f"The final merged dataset contains **{len(master_df)} rows** and **{len(master_df.columns)} columns**.\n")
        total_invalid = sum(s.get('total_removed', 0) for s in cleaning_stats.values())
        f.write(f"A total of **{total_invalid} rows** were removed across all files during cleaning.\n\n")
        
        f.write("## 2. Issues Discovered\n")
        f.write("- **Missing/Invalid Values**: Out-of-bound electrical readings (e.g. negative power, PF > 1), and missing data were flagged.\n")
        f.write("- **Timestamps**: Checked for broken sequences and duplicate timestamps. Exact gaps detailed in Data Profile report.\n")
        f.write("- **Duplicates**: Exact row duplicates were removed.\n\n")
        
        f.write("## 3. Cleaning Performed\n")
        f.write("- Schema validation enforced strictly before load.\n")
        f.write("- Timestamps parsed into standard `datetime`.\n")
        f.write("- Duplicate rows dropped.\n")
        f.write("- Rows with out-of-bounds parameters removed (configured in `config.py`).\n")
        f.write("- All removed rows saved immutably to `invalid_rows.csv` with reject reasons.\n\n")
        
        f.write("## 4. Remaining Concerns\n")
        f.write("- Large time gaps may still exist in the time-series which could affect window-based feature engineering later.\n")
        f.write("- Potential sensor noise close to the boundary limits.\n\n")
        
        f.write("## 5. Recommendations before Feature Engineering\n")
        f.write("- Implement time-series imputation (forward fill or interpolation) for small missing intervals.\n")
        f.write("- Address major time gaps by splitting series into separate contiguous chunks or episodes.\n\n")
        
        f.write("## 6. Columns that may cause leakage\n")
        f.write("- **`nodeid`**: May perfectly correlate with appliances if one node is permanently attached to one appliance type. Needs careful cross-validation strategy (e.g. GroupKFold by nodeid) to ensure generalization to new households/nodes.\n")
        f.write("- **`source_file`**: Explicitly contains the class name, must be dropped.\n")
        f.write("- **`timestamp_ist_dt`**: Absolute time may cause overfitting to specific days/months. Extract relative features (hour of day, day of week) instead.\n\n")
        
        f.write("## 7. Columns that should be investigated further\n")
        f.write("- **`relay`**: Does the relay status correlate 1:1 with power draw? Can it act as a proxy label?\n")
        f.write("- **`energy`**: Cumulative energy might drift. Check if it resets or monotonically increases.\n")
        
    logger.info("All reports generated successfully.")
    
    # 5. Exploratory Data Analysis
    generate_all_plots(master_df)
    
    logger.info("Pipeline execution complete.")

if __name__ == "__main__":
    run_pipeline()
