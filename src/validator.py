import pandas as pd
import numpy as np
import config

def get_invalid_masks(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary of boolean masks indicating which rows are INVALID 
    for various reasons. True means the row is INVALID for that reason.
    """
    masks = {}
    
    # 1. Missing values in critical columns
    # 'label' and 'source_file' shouldn't have missing values.
    # We check the original EXPECTED_COLUMNS
    critical_cols = [c for c in config.EXPECTED_COLUMNS if c in df.columns]
    masks['missing_values'] = df[critical_cols].isna().any(axis=1)
    
    # 2. Impossible electrical values
    m_voltage = (df['voltage'] < config.MIN_VOLTAGE)
    m_current = (df['current'] < config.MIN_CURRENT)
    m_power = (df['power'] < config.MIN_POWER)
    m_pf = (df['pf'] < config.MIN_PF) | (df['pf'] > config.MAX_PF)
    m_freq = (df['frequency'] < config.MIN_FREQ) | (df['frequency'] > config.MAX_FREQ)
    
    # Replace NaNs in these masks with False so they don't break logic
    m_voltage = m_voltage.fillna(False)
    m_current = m_current.fillna(False)
    m_power = m_power.fillna(False)
    m_pf = m_pf.fillna(False)
    m_freq = m_freq.fillna(False)
    
    masks['out_of_bounds'] = m_voltage | m_current | m_power | m_pf | m_freq
    
    # 3. Duplicate rows (ignoring timestamp gaps for duplicate row check)
    masks['duplicates'] = df.duplicated(subset=critical_cols, keep='first')
    
    # 4. Duplicate timestamps (multiple readings at exact same time for same node)
    if 'timestamp_ist' in df.columns and 'nodeid' in df.columns:
        masks['duplicate_timestamps'] = df.duplicated(subset=['nodeid', 'timestamp_ist'], keep='first')
    else:
        masks['duplicate_timestamps'] = pd.Series(False, index=df.index)
        
    return masks

def validate_timestamps(df: pd.DataFrame) -> dict:
    """
    Check for broken sequences and time gaps.
    Assumes df is sorted by timestamp_ist.
    Returns statistics about timestamps.
    """
    if 'timestamp_ist_dt' not in df.columns:
        return {}
        
    # Group by nodeid to calculate diffs
    diffs = df.groupby('nodeid')['timestamp_ist_dt'].diff().dt.total_seconds()
    
    # Identify gaps > (interval + tolerance)
    expected = config.EXPECTED_SAMPLING_INTERVAL_SEC
    tolerance = config.SAMPLING_INTERVAL_TOLERANCE_SEC
    
    gaps = diffs > (expected + tolerance)
    short_intervals = (diffs > 0) & (diffs < (expected - tolerance))
    
    return {
        'total_gaps': gaps.sum(),
        'total_short_intervals': short_intervals.sum(),
        'max_gap_seconds': diffs.max() if not diffs.isna().all() else 0
    }
