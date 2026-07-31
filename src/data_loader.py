import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import config

logger = config.setup_logger("data_loader", config.LOGS_DIR / "data_loader.log")

def infer_appliance_label(filename: str) -> str:
    """Infer appliance label from filename."""
    name_lower = filename.lower()
    
    # Clean mapping of substring to Label
    label_mapping = {
        "ac": "AC",
        "geyser": "Geyser",
        "refrig": "Refrigerator",
        "led": "LED",
        "heater": "Heater"
    }
    
    for key, label in label_mapping.items():
        if key in name_lower:
            return label
            
    return "Unknown"

def validate_schema(df: pd.DataFrame, filename: str) -> bool:
    """
    Validates:
    - required columns
    - column names
    - missing columns
    - unexpected extra columns
    - column order
    """
    expected_cols = config.EXPECTED_COLUMNS
    actual_cols = df.columns.tolist()
    
    # Check for exact match including order
    if actual_cols != expected_cols:
        missing = [c for c in expected_cols if c not in actual_cols]
        extra = [c for c in actual_cols if c not in expected_cols]
        if missing or extra:
            logger.error(f"Schema mismatch in {filename}. Missing: {missing}, Extra: {extra}")
        else:
            logger.error(f"Schema order mismatch in {filename}.")
        return False
        
    return True

def load_data_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Loads a single CSV file, validates its schema, and casts types.
    """
    logger.info(f"Loading {filepath.name}")
    try:
        # Load everything as strings initially to validate raw columns
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        logger.error(f"Failed to read {filepath.name}: {e}")
        return None
        
    if not validate_schema(df, filepath.name):
        return None
        
    # Cast to expected data types
    try:
        for col, dtype in config.SCHEMA_TYPES.items():
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # We don't drop rows with coercion NaNs here, we let the cleaner handle it.
    except Exception as e:
        logger.error(f"Data type conversion failed for {filepath.name}: {e}")
        return None
        
    # Infer label
    df['label'] = infer_appliance_label(filepath.name)
    df['source_file'] = filepath.name
    
    logger.info(f"Successfully loaded {filepath.name} with {len(df)} rows.")
    return df

def load_all_raw_data() -> dict:
    """
    Iterates through raw data directory and loads all valid CSVs.
    Returns a dict mapping filename to DataFrame.
    """
    dataframes = {}
    for f in config.DATA_RAW_DIR.glob("*.csv"):
        df = load_data_file(f)
        if df is not None:
            dataframes[f.name] = df
    return dataframes
