import pandas as pd
import config
import logging

logger = config.setup_logger("merger", config.LOGS_DIR / "merger.log")

def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges a list of cleaned dataframes into one master dataset."""
    logger.info(f"Merging {len(dfs)} datasets...")
    if not dfs:
        logger.warning("No datasets to merge!")
        return pd.DataFrame()
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Final sort just in case
    if 'nodeid' in master_df.columns and 'timestamp_ist_dt' in master_df.columns:
        master_df = master_df.sort_values(by=['nodeid', 'timestamp_ist_dt'])
        
    # Drop intermediate datetime if we want to save space, but it's useful.
    # We will keep it.
    
    logger.info(f"Master dataset created with {len(master_df)} rows.")
    return master_df

def save_master_dataset(df: pd.DataFrame):
    output_path = config.DATA_PROCESSED_DIR / "cleaned_dataset.csv"
    logger.info(f"Saving master dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Save complete.")
