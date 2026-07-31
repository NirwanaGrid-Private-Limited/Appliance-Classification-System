import pandas as pd
import config
from validator import get_invalid_masks

logger = config.setup_logger("cleaner", config.LOGS_DIR / "cleaner.log")

def clean_data(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Cleans the dataframe by applying constraints.
    Returns:
        - cleaned DataFrame
        - invalid rows DataFrame
        - dictionary of cleaning stats
    """
    initial_rows = len(df)
    
    # Pre-parse timestamps to datetime safely to check format
    df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist'], errors='coerce')
    
    # We treat invalid timestamps as missing values for the timestamp_ist_dt column
    # We will flag rows where timestamp parsing failed as out_of_bounds or missing.
    masks = get_invalid_masks(df)
    masks['invalid_timestamp'] = df['timestamp_ist_dt'].isna()
    
    invalid_mask = pd.Series(False, index=df.index)
    
    stats = {}
    
    if config.REMOVE_DUPLICATES:
        invalid_mask = invalid_mask | masks['duplicates'] | masks['duplicate_timestamps']
        stats['removed_duplicates'] = (masks['duplicates'] | masks['duplicate_timestamps']).sum()
        
    if config.REMOVE_MISSING:
        invalid_mask = invalid_mask | masks['missing_values'] | masks['invalid_timestamp']
        stats['removed_missing'] = (masks['missing_values'] | masks['invalid_timestamp']).sum()
        
    if config.REMOVE_OUT_OF_BOUNDS:
        invalid_mask = invalid_mask | masks['out_of_bounds']
        stats['removed_out_of_bounds'] = masks['out_of_bounds'].sum()
        
    invalid_rows = df[invalid_mask].copy()
    
    # Add a reason column to invalid rows
    reasons = []
    for idx, row in invalid_rows.iterrows():
        r = []
        if masks['duplicates'].loc[idx] or masks['duplicate_timestamps'].loc[idx]: r.append("duplicate")
        if masks['missing_values'].loc[idx]: r.append("missing_value")
        if masks['invalid_timestamp'].loc[idx]: r.append("invalid_timestamp")
        if masks['out_of_bounds'].loc[idx]: r.append("out_of_bounds")
        reasons.append("|".join(r))
        
    invalid_rows['reject_reason'] = reasons
    
    # Clean df
    cleaned_df = df[~invalid_mask].copy()
    
    # Sort chronologically
    if not cleaned_df.empty:
        cleaned_df = cleaned_df.sort_values(by=['nodeid', 'timestamp_ist_dt'])
        
    stats['initial_rows'] = initial_rows
    stats['final_rows'] = len(cleaned_df)
    stats['total_removed'] = initial_rows - len(cleaned_df)
    
    logger.info(f"{filename}: Removed {stats['total_removed']} rows out of {initial_rows}.")
    
    return cleaned_df, invalid_rows, stats
