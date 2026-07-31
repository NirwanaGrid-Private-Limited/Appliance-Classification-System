import os
import logging
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Expected Schema
EXPECTED_COLUMNS = [
    "nodeid", "voltage", "current", "power", "energy", 
    "pf", "frequency", "relay", "timestamp_ist"
]

SCHEMA_TYPES = {
    "nodeid": "object",
    "voltage": "float64",
    "current": "float64",
    "power": "float64",
    "energy": "float64",
    "pf": "float64",
    "frequency": "float64",
    "relay": "object",
    "timestamp_ist": "object" # Will be parsed to datetime later
}

# Validation constraints
MIN_VOLTAGE = 0.0
MIN_POWER = 0.0
MIN_CURRENT = 0.0
MIN_PF = 0.0
MAX_PF = 1.0

# Typical Grid frequency is 50Hz or 60Hz. Assuming India (IST in timestamp), so 50Hz. Allow 45-55 Hz.
MIN_FREQ = 45.0
MAX_FREQ = 55.0

# Expected Sampling Interval in seconds
EXPECTED_SAMPLING_INTERVAL_SEC = 60
SAMPLING_INTERVAL_TOLERANCE_SEC = 5

# Cleaning Configurations
REMOVE_DUPLICATES = True
REMOVE_MISSING = True
REMOVE_OUT_OF_BOUNDS = True

import math

# Live Production Inference Configurations
WINDOW_SIZE_SEC = 60
POLLING_INTERVAL_SEC = 10
MIN_WINDOW_SAMPLES = math.ceil(WINDOW_SIZE_SEC / POLLING_INTERVAL_SEC)  # Automatically computed minimum accumulated samples before inference begins

def setup_logger(name, log_file, level=logging.INFO):
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
